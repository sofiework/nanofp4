# nano-vllm EAGLE-3 Speculative Decoding Benchmark (Qwen3-1.7B)

from-scratch EAGLE-3 on nano-vllm (verify / KV rollback / scheduler 机制与 n-gram 共享, proposer 换成 EAGLE-3 head)。同 schema 对照 vLLM real-EAGLE3 ([`vllm_vanilla_eagle3_benchmark.md`](vllm_vanilla_eagle3_benchmark.md)) 与 n-gram ([`nanovllm_ngram_benchmark.md`](nanovllm_ngram_benchmark.md))。

- Raw: [`nanovllm_eagle3.jsonl`](nanovllm_eagle3.jsonl) — 30 runs
- git_commit `407279f`, 2026-07-26T05:54:16

## Config

| model | Qwen3-1.7B / bf16 |
|---|---|
| engine | nano-vllm (from-scratch) |
| proposer | EAGLE-3 head (`AngelSlim/Qwen3-1.7B_eagle3`), 1 midlayer + fc + draft lm_head, 共享 target embedding |
| k (num_spec_tokens) | 2 (vanilla: null) |
| draft KV | target paged KV pool +1 层, 同 block_table |
| decode | greedy (verify=argmax); temperature=0.6 仅影响 prefill 首 token |
| bs / max_tokens / seed | 1 / 1024 / 0 |
| thinking | on (template 默认) |
| gpu | RTX 3060 Ti 8GB, enforce_eager, mixbatch off |

## Results (per run)

accept_len = (gen-1) / target_forward_count, 上界 k+1 = 3。
draft_fwd = midlayer forward 数 (每 verify 步 window 1 次 + single-token 1 次)。

| prompt | vanilla tok/s | eagle3 tok/s | speedup | accept_len | gen | tgt_fwd | draft_fwd | lossless |
|---|---|---|---|---|---|---|---|---|
| code_01 | 31.15 | 33.38 | 1.07× | 1.8106 | 1024 | 565 | 1695 | pass |
| code_02 | 31.74 | 32.76 | 1.03× | 1.7699 | 1024 | 578 | 1734 | pass |
| json_01 | 31.77 | 30.89 | 0.97× | 1.6667 | 276 | 165 | 495 | pass |
| json_02 | 31.68 | 36.78 | 1.16× | 1.9868 | 452 | 227 | 681 | pass |
| qa_01 | 31.75 | 32.05 | 1.01× | 1.728 | 1024 | 592 | 1776 | pass |
| qa_02 | 31.69 | 33.42 | 1.05× | 1.8011 | 1024 | 568 | 1704 | pass |
| story_01 | 31.77 | 28.26 | 0.89× | 1.5227 | 942 | 618 | 1854 | pass |
| story_02 | 31.69 | 29.95 | 0.95× | 1.616 | 948 | 586 | 1758 | pass |
| trans_01 | 31.77 | 31.4 | 0.99× | 1.6918 | 495 | 292 | 876 | pass |
| trans_02 | 31.67 | 28.14 | 0.89× | 1.5183 | 997 | 656 | 1968 | pass |

## Category mean

| category | vanilla | eagle3 | speedup | accept_len |
|---|---|---|---|---|
| code | 31.4 | 33.1 | 1.05× | 1.79 |
| json | 31.7 | 33.8 | 1.07× | 1.83 |
| qa | 31.7 | 32.7 | 1.03× | 1.76 |
| story | 31.7 | 29.1 | 0.92× | 1.57 |
| translation | 31.7 | 29.8 | 0.94× | 1.61 |
| **mean** | **31.7** | **31.7** | **1.00×** | **1.71** |

## 对比 (同 schema JSONL, 各自 vanilla 基线)

| engine / proposer | k | tok/s (spec) | speedup | accept_len |
|---|---|---|---|---|
| nano-vllm / eagle3 | 2 | 31.7 | 1.00× | 1.71 |

## Notes

- lossless = pass by construction (committed token == 同一次 verify forward 的 target argmax)。跨 mode 逐 token 对比时, bf16 近平局 (top-2 logit gap = 1 ulp) 可因 verify q_len 不同翻转 argmax, 非实现差异。
- draft 每步含 1 次 CPU↔GPU sync (`_sample` 的 int()), bs=1 下未做 cross-request batching / cuda graph。
