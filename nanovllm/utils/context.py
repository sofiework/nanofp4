from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    # spec decoding: verify forward keeps logits for every query token, not just the last
    is_verify: bool = False

    # tree spec decoding: tree-masked attention (prefix paged-flash over context + in-hand masked suffix)
    is_tree: bool = False
    tree_bias: torch.Tensor | None = None   # [N, N] additive mask (0 attend, -inf mask); N = tree_q_len
    tree_q_len: int = 0                      # nodes per seq in the current tree forward

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, is_verify=False, is_tree=False, tree_bias=None, tree_q_len=0):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, is_verify, is_tree, tree_bias, tree_q_len)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
