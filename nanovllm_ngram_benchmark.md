# nano-vllm n-gram Speculative Decoding Benchmark (Qwen3-1.7B)

from-scratch n-gram spec decoding on nano-vllm. 同 schema 对照 vLLM baseline ([`vllm_vanilla_eagle3_benchmark.md`](vllm_vanilla_eagle3_benchmark.md))。
EAGLE-3 接上后用同一份 JSONL 真比 accept_len。

- Raw: [`nanovllm_baseline.jsonl`](nanovllm_baseline.jsonl) — 20 runs
- git_commit `bb823b3`, 2026-07-02T01:17:12

## Config

| model | Qwen3-1.7B / bf16 |
|---|---|
| engine | nano-vllm (from-scratch) |
| proposer | n-gram / prompt-lookup (max_n=3), draft 纯 CPU |
| k (num_spec_tokens) | 4 (vanilla: null) |
| decode | greedy (verify=argmax); temperature=0.6 仅影响 prefill 首 token |
| bs / max_tokens / seed | 1 / 1024 / 0 |
| thinking | on (template 默认) |
| gpu | RTX 3060 Ti 8GB, enforce_eager, mixbatch off |

## nano-vllm_ngram Results (per run)

accept_len = (gen-1) / target_forward_count；n-gram draft 无 GPU forward → draft_fwd=0。

| prompt | vanilla tok/s | ngram tok/s | speedup | accept_len | gen | tgt_fwd | lossless |
|---|---|---|---|---|---|---|---|
| code_01 | 29.97 | 40.0 | 1.33× | 1.3843 | 1024 | 739 | pass |
| code_02 | 30.36 | 35.03 | 1.15× | 1.1772 | 1024 | 869 | pass |
| json_01 | 30.45 | 32.96 | 1.08× | 1.1008 | 285 | 258 | pass |
| json_02 | 30.36 | 41.84 | 1.38× | 1.4094 | 452 | 320 | pass |
| qa_01 | 30.38 | 33.74 | 1.11× | 1.1329 | 1024 | 903 | pass |
| qa_02 | 30.35 | 31.85 | 1.05× | 1.069 | 1024 | 957 | pass |
| story_01 | 30.35 | 30.14 | 0.99× | 1.0095 | 955 | 945 | pass |
| story_02 | 30.35 | 31.18 | 1.03× | 1.0471 | 1024 | 977 | pass |
| trans_01 | 30.37 | 35.22 | 1.16× | 1.1801 | 368 | 311 | pass |
| trans_02 | 30.36 | 34.56 | 1.14× | 1.163 | 736 | 632 | pass |

## Category mean

| category | vanilla | ngram | speedup | accept_len |
|---|---|---|---|---|
| code | 30.2 | 37.5 | 1.24× | 1.28 |
| json | 30.4 | 37.4 | 1.23× | 1.26 |
| qa | 30.4 | 32.8 | 1.08× | 1.10 |
| story | 30.4 | 30.7 | 1.01× | 1.03 |
| translation | 30.4 | 34.9 | 1.15× | 1.17 |
| **mean** | **30.3** | **34.7** | **1.14×** | **1.17** |

## Notes

- n-gram accept_len 上界 = k+1 = 5；对 code/json 等重复结构命中高, story/translation 低。
- lossless = pass by construction (verify accepted token == target argmax 同一次 forward)。
- 对比 vLLM EAGLE3 baseline 时注意 k 不同 (此处 4 vs eagle3 的 2) 且 proposer 不同。
