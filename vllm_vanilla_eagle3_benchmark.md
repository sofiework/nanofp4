# vLLM Vanilla vs EAGLE3 Baseline (Qwen3-1.7B)

Reference baseline measured on **real vLLM**, used as the target for our from-scratch
EAGLE-3 implementation in nano-vllm. Two baselines recorded:

1. **`vllm_vanilla`** — vLLM standard autoregressive decode (no spec).
2. **`vllm_eagle3`** — vLLM EAGLE3 speculative decode, `num_speculative_tokens=2`.

- Raw data: [`vllm_baseline.jsonl`](vllm_baseline.jsonl) — one JSON object per run (prompt × mode), 20 runs.
- Script: [`bench_eagle3_vllm.py`](bench_eagle3_vllm.py)
- Read with pandas: `df = pd.read_json("vllm_baseline.jsonl", lines=True)`

---

## Fixed configuration (identical across all runs)

| field | value |
|---|---|
| engine | vLLM `0.23.1rc1.dev` (nightly), torch `2.11.0+cu130` |
| model (target) | Qwen3-1.7B, dtype `bf16` |
| eagle3 head | AngelSlim/Qwen3-1.7B_eagle3, `head_rev=94441b48acc5` |
| k (num_speculative_tokens) | **2** (vanilla: null) — same as AngelSlim paper table |
| temperature | 0.0 (greedy) |
| max_tokens | 1024 |
| bs | 1 (single stream) |
| seed | 0 |
| template | qwen3-chat (`apply_chat_template`, `add_generation_prompt=True`) |
| thinking | **on** (Qwen3 template default, not overridden) |
| gpu | RTX 3060 Ti 8GB |
| max_model_len | 2048, gpu_memory_utilization 0.85, enforce_eager, prefix_caching off |
| git_commit (nano-vllm) | bb823b3 |

---

## Prompt data (5 categories × 2)

| prompt_id | category | text |
|---|---|---|
| code_01 | code | Write a Python function that checks whether a string is a palindrome. |
| code_02 | code | Implement quicksort in Python with comments. |
| json_01 | json | Return a JSON object describing a person: name, age, and a list of 3 hobbies. Only output JSON. |
| json_02 | json | Give a JSON array of the first 5 US presidents, each with name and start year. Only output JSON. |
| qa_01 | qa | What causes the four seasons on Earth? |
| qa_02 | qa | Explain the difference between TCP and UDP. |
| story_01 | story | Write a short story about a robot who discovers music. |
| story_02 | story | Write a short story about a lighthouse keeper on the last night before automation. |
| trans_01 | translation | Translate to French: 'The weather is nice today, let us go for a walk in the park.' |
| trans_02 | translation | Translate to Chinese: 'Speculative decoding verifies multiple draft tokens in parallel.' |

---

## JSONL schema (columns)

**Identity:** `run_id`, `engine`, `mode`, `prompt_category`, `prompt_text`, `max_tokens`
**Config:** `model`, `dtype`, `k`, `temperature`, `seed`, `bs`, `gpu`, `git_commit`, `head_rev`, `thinking`, `template`, `timestamp`
**Performance:** `decode_tok_s` (decode-only, excludes prefill), `ttft_ms` (prefill+1 token probe), `avg_accept_len` (= total_generated_tokens ÷ target_forward_count), `total_generated_tokens`
**Diagnostic:** `target_forward_count` (verify/decode steps), `draft_forward_count` (eagle-head forwards = target_forward × k), `lossless` (`pass` / `diverge@N` / `ref`)

---

## Results (per run)

decode_tok_s is single-stream, decode phase only (prefill excluded).

| prompt | vanilla tok/s | eagle3 tok/s | speedup | accept_len | gen | tgt_fwd | draft_fwd | lossless |
|---|---|---|---|---|---|---|---|---|
| code_01 | 66.5 | 105.1 | 1.58× | 1.80 | 1024 | 570 | 1140 | diverge@34 |
| code_02 | 66.8 | 104.8 | 1.57× | 1.80 | 1024 | 569 | 1138 | diverge@232 |
| json_01 | 66.9 | 101.1 | 1.51× | 1.75 | 277 | 158 | 316 | diverge@100 |
| json_02 | 66.7 | 115.4 | 1.73× | 1.99 | 452 | 227 | 454 | **pass** |
| qa_01 | 64.2 | 94.5 | 1.47× | 1.62 | 1024 | 632 | 1264 | diverge@4 |
| qa_02 | 64.5 | 99.0 | 1.53× | 1.70 | 1024 | 602 | 1204 | diverge@137 |
| story_01 | 65.1 | 93.0 | 1.43× | 1.60 | 959 | 599 | 1198 | diverge@121 |
| story_02 | 65.1 | 94.0 | 1.44× | 1.61 | 1024 | 636 | 1272 | diverge@360 |
| trans_01 | 65.0 | 98.5 | 1.51× | 1.70 | 495 | 292 | 584 | **pass** |
| trans_02 | 65.1 | 92.0 | 1.41× | 1.58 | 1024 | 650 | 1300 | diverge@184 |

### Category summary

| category | vanilla tok/s | eagle3 tok/s | speedup | avg accept_len |
|---|---|---|---|---|
| code | 66.6 | 104.9 | 1.58× | 1.80 |
| json | 66.8 | 108.3 | 1.62× | 1.87 |
| qa | 64.3 | 96.7 | 1.50× | 1.66 |
| story | 65.1 | 93.5 | 1.44× | 1.61 |
| translation | 65.0 | 95.2 | 1.46× | 1.64 |
| **mean** | **65.6** | **99.7** | **1.52×** | **1.72** |

---



## Reproduce

```bash
conda activate vllm          # vLLM 0.23 nightly cu130 env (separate from nanovllm env)
cd ~/nano-vllm
python bench_eagle3_vllm.py   # writes vllm_baseline.jsonl (20 runs)
```
