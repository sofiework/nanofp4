import torch
from torch import nn

from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.qwen3 import Qwen3MLP
from nanovllm.utils.context import set_context


class Eagle3Midlayer(nn.Module):
    # llama-style decoder layer, 双流输入: qkv 输入 4096 = concat(embed, feature), 无 q_norm/k_norm

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 6144,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1e6,
        max_position: int = 40960,
    ) -> None:
        
        super().__init__()
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # embed & feature norm
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)   # norm embed stream
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)       # norm feature stream
        
        # qkv dim = 4096 = concat(embed, feature)
        self.qkv_proj = QKVParallelLinear(2 * hidden_size, head_dim, num_heads, num_kv_heads, bias=False)
        self.o_proj = RowParallelLinear(num_heads * head_dim, hidden_size, bias=False)
        self.rotary_emb = get_rope(head_dim, rotary_dim=head_dim, max_position=max_position, base=rope_theta)
        self.attn = Attention(num_heads, head_dim, head_dim ** -0.5, num_kv_heads)  # k/v_cache 由 allocate_kv_cache 挂
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, "silu")

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # norm embed & feature
        e = self.input_layernorm(embeds)
        residual = feature                      # residual = pre-norm feature
        h = self.hidden_norm(feature)

        # concact embed & feature
        qkv = self.qkv_proj(torch.cat([e, h], dim=-1))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # RoPE
        q, k = self.rotary_emb(positions, q, k)
        # draft KV cache + attention
        o = self.attn(q, k, v)                 
        x = self.o_proj(o.flatten(1, -1))

        # fused add + norm
        # x: normed, for MLP
        # residual: un-normed
        x, residual = self.post_attention_layernorm(x, residual)

        # MLP
        x = self.mlp(x)

        # second residual add 留给 Proposer 的 self.norm
        return x, residual                      


class Eagle3Proposer(nn.Module):

    def __init__(self, embed_tokens: nn.Module, draft_vocab_size: int = 32000, hidden_size: int = 2048) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens        # 共享 target 的 embedding, 不复制权重
        self.fc = nn.Linear(3 * hidden_size, hidden_size, bias=False)
        self.midlayer = Eagle3Midlayer(hidden_size=hidden_size)
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)

        self.register_buffer("d2t", torch.zeros(draft_vocab_size, dtype=torch.int64))

    def load_weights(self, path: str):
        # head 是 pytorch_model.bin (fp16, key 带 midlayer. 前缀), 手动映射; copy_ 自动转 dtype/device
        sd = torch.load(f"{path}/pytorch_model.bin", map_location="cpu", weights_only=True)
        m = self.midlayer
        m.qkv_proj.weight.data.copy_(torch.cat(
            [sd["midlayer.self_attn.q_proj.weight"],
             sd["midlayer.self_attn.k_proj.weight"],
             sd["midlayer.self_attn.v_proj.weight"]], dim=0))
        m.mlp.gate_up_proj.weight.data.copy_(torch.cat(
            [sd["midlayer.mlp.gate_proj.weight"],
             sd["midlayer.mlp.up_proj.weight"]], dim=0))       # 顺序对照 Qwen3MLP: gate 前 up 后
        m.o_proj.weight.data.copy_(sd["midlayer.self_attn.o_proj.weight"])
        m.mlp.down_proj.weight.data.copy_(sd["midlayer.mlp.down_proj.weight"])
        m.input_layernorm.weight.data.copy_(sd["midlayer.input_layernorm.weight"])
        m.hidden_norm.weight.data.copy_(sd["midlayer.hidden_norm.weight"])
        m.post_attention_layernorm.weight.data.copy_(sd["midlayer.post_attention_layernorm.weight"])
        self.fc.weight.data.copy_(sd["fc.weight"])
        self.norm.weight.data.copy_(sd["norm.weight"])
        self.lm_head.weight.data.copy_(sd["lm_head.weight"])
        self.d2t.copy_(sd["d2t"])               # t2d 跳过, greedy 用不到

    def _sample(self, hidden: torch.Tensor) -> int:
        # hidden [1, 2048] -> draft vocab argmax -> target vocab id
        draft_id = self.lm_head(hidden).argmax(dim=-1)
        return int(draft_id + self.d2t[draft_id])

    @torch.inference_mode()
    def propose(
        self,
        input_ids: torch.Tensor,      # [n_use] shifted window tokens
        positions: torch.Tensor,      # [n_use]
        features: torch.Tensor,       # [n_use, 6144] fc 前原料
        num_draft: int,               # 0 = 只填 draft KV (非最后 prefill chunk)
        steps: list[tuple[int, int]], # single-token pass 每步的 (slot, context_len)
        block_tables: torch.Tensor,
    ) -> list[int]:
        
        # window pass, fc projection
        feat = self.fc(features)

        # x: MLP output, residual: pre-normed
        x, residual = self.midlayer(positions, self.embed_tokens(input_ids), feat)
        
        # fused add residual2 + norm
        # hidden -> for lm head, prenorm -> next feature
        hidden, prenorm = self.norm(x, residual)

        if num_draft == 0:
            return []

        # hidden[-1].shape = [1, 2048]
        out = [self._sample(hidden[-1:])]       # first draft token
        feat = prenorm[-1:]                     # last token -> next feature

        # k-1 次 single-token pass (decode 路径)
        for slot, ctx_len in steps:
            # is_prefill = False -> decode attn
            # slot_mapping = [slot] -> KV
            # context_lens = [ctx_len] -> KV read length
            set_context(False,
                        slot_mapping=torch.tensor([slot], dtype=torch.int32, device="cuda"),
                        context_lens=torch.tensor([ctx_len], dtype=torch.int32, device="cuda"),
                        block_tables=block_tables)
            pos = torch.tensor([ctx_len - 1], dtype=torch.int64, device="cuda")  
            tok = torch.tensor([out[-1]], dtype=torch.int64, device="cuda")

            x, residual = self.midlayer(pos, self.embed_tokens(tok), feat)
            hidden, feat = self.norm(x, residual)

            out.append(self._sample(hidden))
        return out

