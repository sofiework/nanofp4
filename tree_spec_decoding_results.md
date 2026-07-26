# Tree speculative decoding (EAGLE-3) — results

From-scratch tree drafting + tree-masked verification for nano-vllm's EAGLE-3 path, referencing
vLLM's `tree_attn` backend. Replaces the single linear draft chain with a static speculative token
tree: multiple candidate branches are drafted and verified in one target forward, raising accepted
tokens per step.

## What it does

- **Static token tree**, encoded vLLM-style as a path list (`speculative_token_tree`), e.g.
  `[(0,),(1,),(0,0),(0,1),(1,0),(1,1)]` = a depth-2 tree with 2 children/node (7 nodes).
  Default (no tree given) = a chain of `num_spec_tokens`, identical topology to the old path.
- **Tree-masked attention** (`Attention.tree_forward`): each node attends to the full committed
  context (paged flash, `causal=False`) plus its own ancestors in the tree (in-hand masked SDPA),
  the two disjoint key-sets merged by online softmax (log-sum-exp). No custom CUDA/Triton kernel —
  built from `flash_attn_with_kvcache` + torch.
- **Proposer** expands the tree level by level, taking top-`b` children per node from the draft head.
- **Greedy tree verify**: walk from the root along the branch whose drafted token matches the
  target's argmax at each node; commit that branch + 1 bonus token. Lossless by construction —
  every committed token is the target model's own argmax at that node.
- Accepted-branch KV is gathered into contiguous committed slots; rejected KV is overwritten next step.

## Accept length (Qwen3-1.7B, bf16, greedy, bs=1, RTX 3060 Ti)

Mean accepted tokens per target forward, over 4 prompts (tcp / code / qa / json):

| draft structure | topology | accept length |
|---|---|---|
| chain (baseline) | `num_spec_tokens=2`, N=3 | 1.615 |
| tree | depth-2, 2-ary, N=7 | **1.832**  (+13%) |
| tree (deeper) | depth-3, N=7 | **1.896**  (+17%) |

Tree drafting yields the **same greedy token sequence** as the chain (each committed token is the
target argmax) while accepting more tokens per step — fewer target forwards for identical output.

## Correctness

- **Greedy-lossless by construction**: committed token == the target's argmax at that node in the
  same verify forward (`lossless=pass`).
- **Tree attention numerics**: `tree_forward` matches a full fp32 dense reference to max abs error
  1.6e-3 (bf16 level) — see `test_tree_spec.py::test_tree_attention_oracle`.
- **Accept walk**: unit-tested for full-depth accept, root mismatch, alternate branch, and partial
  accept; reduces to "longest prefix + 1 bonus" on a chain (`test_accept_walk`).
- **vs sequential decode**: output can differ from a token-by-token bf16 decode only by occasional
  bf16 tie-flips (top-2 logit gap ≈ 1 ulp) — the two-pass tree attention rounds differently than a
  single flash kernel. This is inherent to tree/batched verification (vLLM included), not an
  implementation difference; it does not affect the lossless-by-construction guarantee.

## Scope

- Tree applies to **EAGLE-3**. The n-gram proposer keeps its dynamic linear chain (unchanged path).
- Static tree topology, greedy verify. Rejection sampling for `temperature>0` is a separable follow-up.

## Usage

```python
from nanovllm import LLM
llm = LLM("Qwen3-1.7B", enable_spec=True, spec_method="eagle3", draft_model="Qwen3-1.7B_eagle3",
          num_spec_tokens=2,                                   # chain fallback length
          speculative_token_tree="[(0,),(1,),(0,0),(0,1),(1,0),(1,1)]")   # or omit for a chain
```

Benchmark: `python bench_eagle3_nanovllm.py` runs vanilla / eagle3-chain / eagle3-tree and reports
per-mode accept length and decode throughput.
