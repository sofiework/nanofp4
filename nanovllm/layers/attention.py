import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_tree:
            return self.tree_forward(q, k, v, context)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o

    # tree-masked attention (vLLM tree_attn style): each of the N tree nodes attends to the whole
    # committed context (paged flash, no causal) + its ancestors in the tree (in-hand masked SDPA),
    # merged by online softmax. Query layout is [S*N] tokens, seq-major; N = context.tree_q_len.
    def tree_forward(self, q, k, v, context):
        N = context.tree_q_len
        S = q.size(0) // N
        Hq, Hkv, Dh = self.num_heads, self.num_kv_heads, self.head_dim
        q4 = q.view(S, N, Hq, Dh)

        # prefix: committed context via paged flash (causal=False -> all nodes see all context), keep the LSE
        pre_out, pre_lse = flash_attn_with_kvcache(
            q4, self.k_cache, self.v_cache,
            cache_seqlens=context.context_lens, block_table=context.block_tables,
            softmax_scale=self.scale, causal=False, return_softmax_lse=True)   # [S,N,Hq,Dh], [S,Hq,N]

        # suffix: in-hand masked attention over the N tree nodes' k/v (tiny; explicit for the LSE)
        g = Hq // Hkv
        kt = k.view(S, N, Hkv, Dh).repeat_interleave(g, dim=2).float()   # [S,N,Hq,Dh]
        vt = v.view(S, N, Hkv, Dh).repeat_interleave(g, dim=2).float()
        scores = torch.einsum("snhd,smhd->shnm", q4.float(), kt) * self.scale   # [S,Hq,N,N]
        scores = scores + context.tree_bias[None, None]
        suf_lse = torch.logsumexp(scores, dim=-1)                              # [S,Hq,N]
        suf_out = torch.einsum("shnm,smhd->snhd", scores.softmax(-1), vt)      # [S,N,Hq,Dh]

        # merge the two disjoint key-sets (context vs tree) by online softmax
        pre_lse = pre_lse.transpose(1, 2).float()   # [S,N,Hq]
        suf_lse = suf_lse.transpose(1, 2).float()
        m = torch.maximum(pre_lse, suf_lse)
        p = torch.exp(pre_lse - m)
        s = torch.exp(suf_lse - m)
        o = (pre_out.float() * p[..., None] + suf_out * s[..., None]) / (p + s)[..., None]
        return o.reshape(S * N, Hq, Dh).to(q.dtype)
