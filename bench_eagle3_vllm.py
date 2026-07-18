"""
EAGLE3 vs vanilla baseline on real vLLM -> dump JSONL (一行一个 run = 一个 prompt x 一个 mode).
pandas: df = pd.read_json("vllm_baseline.jsonl", lines=True)

规格 (memory spec-decoding-benchmark-spec): k=2 (AngelSlim 论文同), bs=1, max_tokens=1024,
greedy(temperature=0), thinking=on(qwen3 template 默认), 5 类 prompt 各 2 条.
硬件: 3060Ti 8GB, target=Qwen3-1.7B, eagle3=AngelSlim/Qwen3-1.7B_eagle3.
用法 (vllm conda env): python bench_eagle3_vllm.py
"""
import os, json, time, subprocess, datetime
from vllm import LLM, SamplingParams

TARGET = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
EAGLE3 = os.path.expanduser("~/huggingface/Qwen3-1.7B_eagle3/")
OUT_JSONL = os.path.expanduser("~/nano-vllm/vllm_baseline.jsonl")

MAX_TOKENS, K, TEMPERATURE, SEED, BS = 1024, 2, 0.0, 0, 1
GPU_UTIL, MAX_MODEL_LEN = 0.85, 2048
GPU = "RTX 3060 Ti 8GB"
MODEL_NAME, DTYPE, TEMPLATE, THINKING = "Qwen3-1.7B", "bf16", "qwen3-chat", True

# (prompt_id, category, text)  category in code/json/qa/story/translation
PROMPTS = [
    ("code_01", "code", "Write a Python function that checks whether a string is a palindrome."),
    ("code_02", "code", "Implement quicksort in Python with comments."),
    ("json_01", "json", "Return a JSON object describing a person: name, age, and a list of 3 hobbies. Only output JSON."),
    ("json_02", "json", "Give a JSON array of the first 5 US presidents, each with name and start year. Only output JSON."),
    ("qa_01",   "qa",   "What causes the four seasons on Earth?"),
    ("qa_02",   "qa",   "Explain the difference between TCP and UDP."),
    ("story_01","story","Write a short story about a robot who discovers music."),
    ("story_02","story","Write a short story about a lighthouse keeper on the last night before automation."),
    ("trans_01","translation","Translate to French: 'The weather is nice today, let us go for a walk in the park.'"),
    ("trans_02","translation","Translate to Chinese: 'Speculative decoding verifies multiple draft tokens in parallel.'"),
]


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.expanduser("~/nano-vllm"),
                          capture_output=True, text=True).stdout.strip() or "unknown"


def head_rev():
    try:  # eagle3 head 权重版本
        from huggingface_hub import HfApi
        return HfApi().model_info("AngelSlim/Qwen3-1.7B_eagle3").sha[:12]
    except Exception:
        return "AngelSlim/Qwen3-1.7B_eagle3"


def spec_counts(llm):
    d = dt = a = 0
    for x in llm.llm_engine.get_metrics():
        n = getattr(x, "name", "")
        if n == "vllm:spec_decode_num_drafts": d = x.value
        elif n == "vllm:spec_decode_num_draft_tokens": dt = x.value
        elif n == "vllm:spec_decode_num_accepted_tokens": a = x.value
    return d, dt, a


def prefix_match(a, b):
    m = 0
    for x, y in zip(a, b):
        if x != y: break
        m += 1
    return m


def run_mode(mode, gc, hr, ts, vanilla_ids=None):
    kw = dict(model=TARGET, gpu_memory_utilization=GPU_UTIL, max_model_len=MAX_MODEL_LEN,
              enforce_eager=True, seed=SEED, enable_prefix_caching=False, disable_log_stats=False)
    if mode == "eagle3":
        kw["speculative_config"] = {"method": "eagle3", "model": EAGLE3, "num_speculative_tokens": K}
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    probe = SamplingParams(temperature=TEMPERATURE, max_tokens=1)
    full = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    rows, out_map = [], {}
    for pid, cat, text in PROMPTS:
        prompt = tok.apply_chat_template([{"role": "user", "content": text}],
                                         tokenize=False, add_generation_prompt=True)
        # TTFT probe (prefill + 1 token)
        t0 = time.perf_counter(); llm.generate([prompt], probe, use_tqdm=False)
        ttft_ms = (time.perf_counter() - t0) * 1000
        d0, dt0, a0 = spec_counts(llm)
        # full generation
        t0 = time.perf_counter(); out = llm.generate([prompt], full, use_tqdm=False)[0]
        e2e_ms = (time.perf_counter() - t0) * 1000
        d1, dt1, a1 = spec_counts(llm)
        ids = list(out.outputs[0].token_ids)
        n = len(ids); out_map[pid] = ids
        decode_ms = max(e2e_ms - ttft_ms, 1e-6)
        decode_tok_s = (n - 1) / (decode_ms / 1000) if n > 1 else 0.0
        if mode == "eagle3":
            tfc = d1 - d0                      # target forward = verify steps
            dfc = dt1 - dt0                    # draft head forwards
            acc = n / tfc if tfc else float("nan")
            div = prefix_match(vanilla_ids.get(pid, []), ids)
            lossless = "pass" if vanilla_ids.get(pid) == ids else f"diverge@{div}"
        else:
            tfc, dfc, acc, lossless = n, None, 1.0, "ref"
        rows.append(dict(
            run_id=f"vllm_{mode}_{pid}", engine="vllm", mode=mode, prompt_category=cat,
            prompt_text=text, max_tokens=MAX_TOKENS,
            model=MODEL_NAME, dtype=DTYPE, k=(K if mode == "eagle3" else None),
            temperature=TEMPERATURE, seed=SEED, bs=BS, gpu=GPU,
            git_commit=gc, head_rev=hr, thinking=THINKING, template=TEMPLATE, timestamp=ts,
            decode_tok_s=round(decode_tok_s, 2), ttft_ms=round(ttft_ms, 2),
            avg_accept_len=round(acc, 4) if acc == acc else None, total_generated_tokens=n,
            target_forward_count=tfc, draft_forward_count=dfc, lossless=lossless,
        ))
    del llm
    return rows, out_map


def main():
    gc, hr, ts = git_commit(), head_rev(), datetime.datetime.now().isoformat(timespec="seconds")
    van_rows, van_ids = run_mode("vanilla", gc, hr, ts)
    eag_rows, _ = run_mode("eagle3", gc, hr, ts, vanilla_ids=van_ids)
    with open(OUT_JSONL, "w") as f:
        for r in van_rows + eag_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(van_rows) + len(eag_rows)} runs -> {OUT_JSONL}")
    for r in eag_rows:
        print(f"  {r['run_id']:<24} tok/s={r['decode_tok_s']:<7} accept={r['avg_accept_len']} "
              f"tgt_fwd={r['target_forward_count']} lossless={r['lossless']}")


if __name__ == "__main__":
    main()
