# nano-vllm EAGLE-3 Speculative Decoding Benchmark (Qwen3-1.7B)

from-scratch EAGLE-3 on nano-vllm (verify / KV rollback / scheduler 机制与 n-gram 共享, proposer 换成 EAGLE-3 head)。同 schema 对照 vLLM real-EAGLE3 ([`vllm_vanilla_eagle3_benchmark.md`](vllm_vanilla_eagle3_benchmark.md)) 与 n-gram ([`nanovllm_ngram_benchmark.md`](nanovllm_ngram_benchmark.md))。

- Raw: [`nanovllm_eagle3.jsonl`](nanovllm_eagle3.jsonl) — 20 runs
- git_commit `bb823b3`, 2026-07-08T23:17:51

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
| code_01 | 30.85 | 43.62 | 1.41× | 1.7608 | 1024 | 581 | 1162 | pass |
| code_02 | 31.64 | 42.44 | 1.34× | 1.7022 | 1024 | 601 | 1202 | pass |
| json_01 | 31.68 | 42.23 | 1.33× | 1.6887 | 256 | 151 | 302 | pass |
| json_02 | 31.66 | 49.86 | 1.57× | 2.0 | 447 | 223 | 446 | pass |
| qa_01 | 31.71 | 43.26 | 1.36× | 1.7368 | 1024 | 589 | 1178 | pass |
| qa_02 | 31.65 | 44.59 | 1.41× | 1.7916 | 1024 | 571 | 1142 | pass |
| story_01 | 31.73 | 38.58 | 1.22× | 1.5453 | 1024 | 662 | 1324 | pass |
| story_02 | 31.65 | 39.1 | 1.24× | 1.5662 | 900 | 574 | 1148 | pass |
| trans_01 | 31.68 | 42.02 | 1.33× | 1.6918 | 495 | 292 | 584 | pass |
| trans_02 | 31.66 | 38.01 | 1.20× | 1.5275 | 612 | 400 | 800 | pass |

## Category mean

| category | vanilla | eagle3 | speedup | accept_len |
|---|---|---|---|---|
| code | 31.2 | 43.0 | 1.38× | 1.73 |
| json | 31.7 | 46.0 | 1.45× | 1.84 |
| qa | 31.7 | 43.9 | 1.39× | 1.76 |
| story | 31.7 | 38.8 | 1.23× | 1.56 |
| translation | 31.7 | 40.0 | 1.26× | 1.61 |
| **mean** | **31.6** | **42.4** | **1.34×** | **1.70** |

## 对比 (同 schema JSONL, 各自 vanilla 基线)

| engine / proposer | k | tok/s (spec) | speedup | accept_len |
|---|---|---|---|---|
| nano-vllm / eagle3 | 2 | 42.4 | 1.34× | 1.70 |
| nano-vllm / ngram | 4 | 34.7 | 1.14× | 1.17 |
| vLLM / eagle3 | 2 | 99.7 | 1.52× | 1.71 |

## Notes

- lossless = pass by construction (committed token == 同一次 verify forward 的 target argmax)。跨 mode 逐 token 对比时, bf16 近平局 (top-2 logit gap = 1 ulp) 可因 verify q_len 不同翻转 argmax, 非实现差异。
- draft 每步含 1 次 CPU↔GPU sync (`_sample` 的 int()), bs=1 下未做 cross-request batching / cuda graph。
- **baseline 口径**: vanilla 与 eagle3 同为 enforce_eager, speedup 是 eager-vs-eager。
  cuda graph 开启的 vanilla decode 为 97.3 tok/s (同 prompt 实测) — spec 路径 (verify 变长)
  未接入 cuda graph, 相对默认配置 vanilla 为 0.44×。加速比中硬件无关的部分是
  target forward 数减少 41% (= 1 − 1/accept_len)。
