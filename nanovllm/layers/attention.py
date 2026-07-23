import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.utils.kv_quant import fake_quant_kv
from nanovllm.layers.paged_attn import paged_attn, store_kv_quant


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


def _varlen_query_map(context):
    """Per query row: (seq index, causal length) from cu_seqlens. ctx = absolute position + 1."""
    cq, ck = context.cu_seqlens_q, context.cu_seqlens_k
    sq = (cq[1:] - cq[:-1]).to(torch.int32)
    sk = (ck[1:] - ck[:-1]).to(torch.int32)
    B = sq.numel()
    Nq = int(cq[-1].item())
    q_seq = torch.repeat_interleave(torch.arange(B, dtype=torch.int32, device=cq.device), sq)
    local = torch.arange(Nq, dtype=torch.int32, device=cq.device) - cq[:-1].to(torch.int32)[q_seq]
    q_ctx = (sk - sq)[q_seq] + local + 1
    return q_seq, q_ctx


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
        # real KV quant: format ("" | "fp8" | "fp4") + per-token(-head/-block) scale caches;
        # all set per-layer by ModelRunner.allocate_kv_cache.
        self.kv_quant_fmt = ""
        self.k_scale = self.v_scale = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        fmt = self.kv_quant_fmt
        k_cache, v_cache = self.k_cache, self.v_cache
        has_cache = k_cache.numel() and v_cache.numel()

        if has_cache:
            if fmt:
                store_kv_quant(k, v, k_cache, v_cache, self.k_scale, self.v_scale, context.slot_mapping, fmt)
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if fmt and context.block_tables is not None:
                # chunked-prefill continuation / prefix cache / spec-verify: read quantized paged cache
                q_seq, q_ctx = _varlen_query_map(context)
                o = paged_attn(q, k_cache, v_cache, context.block_tables, q_seq, q_ctx, self.scale,
                               fmt, self.k_scale, self.v_scale)
            else:
                if fmt:    # fresh full prefill: attend on in-hand k,v (fake-quant to match stored cache)
                    k, v = fake_quant_kv(k, fmt), fake_quant_kv(v, fmt)
                elif context.block_tables is not None:    # bf16 prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True,
                                           block_table=None if fmt else context.block_tables)
        else:    # decode
            if fmt:
                B = q.size(0)
                q_seq = torch.arange(B, dtype=torch.int32, device=q.device)
                o = paged_attn(q, k_cache, v_cache, context.block_tables, q_seq, context.context_lens,
                               self.scale, fmt, self.k_scale, self.v_scale)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
        return o
