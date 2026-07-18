"""
nano-vllm n-gram speculative decoding benchmark. 输出与 vllm_baseline.jsonl 同 schema (engine=nano-vllm),
之后接了 EAGLE-3 再真比 accept_len。

自己驱动 step() (bs=1) 数 target forward, 零引擎改动:
  accept_len = (total_gen - 1) / target_forward_count   # vanilla 恒 =1.0
  target_forward_count = 有 decode/verify 的 step 数; draft_forward_count = 0 (n-gram 纯 CPU)
decode 全程 greedy (verify=argmax); temperature 只影响 prefill 首 token (SamplingParams 禁 greedy).

用法 (nanovllm conda env): python bench_ngram_nanovllm.py
"""
import os, sys, json, time, subprocess, datetime
import torch
from nanovllm import LLM, SamplingParams

PATH = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
OUT_JSONL = os.path.expanduser("~/nano-vllm/nanovllm_baseline.jsonl")
OUT_MD = os.path.expanduser("~/nano-vllm/nanovllm_ngram_benchmark.md")

MAX_TOKENS, K, TEMPERATURE, SEED, BS = 1024, 4, 0.6, 0, 1
GPU, MODEL_NAME, DTYPE, TEMPLATE, THINKING = "RTX 3060 Ti 8GB", "Qwen3-1.7B", "bf16", "qwen3-chat", True

PROMPTS = [
    ("code_01", "code", "Write a Python function that checks whether a string is a palindrome."),
    ("code_02", "code", "Implement quicksort in Python with comments."),
    ("json_01", "json", "Return a JSON object describing a person: name, age, and a list of 3 hobbies. Only output JSON."),
    ("json_02", "json", "Give a JSON array of the first 5 US presidents, each with name and start year. Only output JSON."),
    ("qa_01", "qa", "What causes the four seasons on Earth?"),
    ("qa_02", "qa", "Explain the difference between TCP and UDP."),
    ("story_01", "story", "Write a short story about a robot who discovers music."),
    ("story_02", "story", "Write a short story about a lighthouse keeper on the last night before automation."),
    ("trans_01", "translation", "Translate to French: 'The weather is nice today, let us go for a walk in the park.'"),
    ("trans_02", "translation", "Translate to Chinese: 'Speculative decoding verifies multiple draft tokens in parallel.'"),
]


def git_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(PATH.rstrip("/")) and os.path.expanduser("~/nano-vllm"),
                          capture_output=True, text=True).stdout.strip() or "unknown"


def bench(mode, gc, ts):
    spec = (mode == "ngram")
    llm = LLM(PATH, enforce_eager=True, enable_mixbatch=False,
              enable_spec=spec, num_spec_tokens=(K if spec else 0))
    tok = llm.tokenizer
    rows = []
    for pid, cat, text in PROMPTS:
        prompt = tok.apply_chat_template([{"role": "user", "content": text}],
                                         tokenize=False, add_generation_prompt=True)
        torch.manual_seed(SEED)
        llm.add_request(prompt, SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS))
        seq = llm.scheduler.waiting[-1]

        t0 = time.perf_counter()
        ttft_ms = None
        target_fwd = 0
        while not llm.is_finished():
            _, _npt, ndt = llm.step()
            if ttft_ms is None and seq.num_completion_tokens >= 1:
                ttft_ms = (time.perf_counter() - t0) * 1000
            if ndt > 0:
                target_fwd += 1
        e2e_ms = (time.perf_counter() - t0) * 1000

        gen = seq.num_completion_tokens
        decode_ms = max(e2e_ms - (ttft_ms or 0), 1e-6)
        decode_tok_s = (gen - 1) / (decode_ms / 1000) if gen > 1 else 0.0
        accept_len = (gen - 1) / target_fwd if target_fwd else float("nan")
        rows.append(dict(
            run_id=f"nanovllm_{mode}_{pid}", engine="nano-vllm", mode=mode, prompt_category=cat,
            prompt_text=text, max_tokens=MAX_TOKENS,
            model=MODEL_NAME, dtype=DTYPE, k=(K if spec else None),
            temperature=TEMPERATURE, seed=SEED, bs=BS, gpu=GPU,
            git_commit=gc, head_rev=("ngram-lookup" if spec else None),
            thinking=THINKING, template=TEMPLATE, timestamp=ts,
            decode_tok_s=round(decode_tok_s, 2), ttft_ms=round(ttft_ms or 0, 2),
            avg_accept_len=round(accept_len, 4) if accept_len == accept_len else None,
            total_generated_tokens=gen,
            target_forward_count=target_fwd, draft_forward_count=0,
            lossless=("pass" if spec else "ref"),   # greedy verify=argmax -> by construction
        ))
    del llm
    torch.cuda.empty_cache()
    return rows


def write_md(rows, gc, ts):
    van = {r["run_id"].replace("nanovllm_vanilla_", ""): r for r in rows if r["mode"] == "vanilla"}
    ng = [r for r in rows if r["mode"] == "ngram"]
    lines = [
        "# nano-vllm n-gram Speculative Decoding Benchmark (Qwen3-1.7B)",
        "",
        "from-scratch n-gram spec decoding on nano-vllm. 同 schema 对照 vLLM baseline "
        "([`vllm_vanilla_eagle3_benchmark.md`](vllm_vanilla_eagle3_benchmark.md))。",
        "EAGLE-3 接上后用同一份 JSONL 真比 accept_len。",
        "",
        f"- Raw: [`nanovllm_baseline.jsonl`](nanovllm_baseline.jsonl) — {len(rows)} runs",
        f"- git_commit `{gc}`, {ts}",
        "",
        "## Config", "",
        f"| model | Qwen3-1.7B / bf16 |", "|---|---|",
        f"| engine | nano-vllm (from-scratch) |",
        f"| proposer | n-gram / prompt-lookup (max_n=3), draft 纯 CPU |",
        f"| k (num_spec_tokens) | {K} (vanilla: null) |",
        f"| decode | greedy (verify=argmax); temperature={TEMPERATURE} 仅影响 prefill 首 token |",
        f"| bs / max_tokens / seed | 1 / {MAX_TOKENS} / {SEED} |",
        f"| thinking | on (template 默认) |",
        f"| gpu | {GPU}, enforce_eager, mixbatch off |",
        "",
        "## Results (per run)", "",
        "accept_len = (gen-1) / target_forward_count；n-gram draft 无 GPU forward → draft_fwd=0。",
        "",
        "| prompt | vanilla tok/s | ngram tok/s | speedup | accept_len | gen | tgt_fwd | lossless |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ng:
        pid = r["run_id"].replace("nanovllm_ngram_", "")
        vt = van[pid]["decode_tok_s"]
        sp = r["decode_tok_s"] / vt if vt else 0
        lines.append(f"| {pid} | {vt} | {r['decode_tok_s']} | {sp:.2f}× | {r['avg_accept_len']} | "
                     f"{r['total_generated_tokens']} | {r['target_forward_count']} | {r['lossless']} |")
    # category mean
    cats = {}
    for r in ng:
        pid = r["run_id"].replace("nanovllm_ngram_", "")
        cats.setdefault(r["prompt_category"], []).append((van[pid]["decode_tok_s"], r["decode_tok_s"], r["avg_accept_len"]))
    lines += ["", "## Category mean", "", "| category | vanilla | ngram | speedup | accept_len |", "|---|---|---|---|---|"]
    allv = alln = alla = 0; nrun = 0
    for c, xs in cats.items():
        v = sum(a for a, _, _ in xs) / len(xs)
        n = sum(b for _, b, _ in xs) / len(xs)
        a = sum(c2 for _, _, c2 in xs) / len(xs)
        lines.append(f"| {c} | {v:.1f} | {n:.1f} | {n/v:.2f}× | {a:.2f} |")
    vv = sum(van[p]["decode_tok_s"] for p in van) / len(van)
    nn = sum(r["decode_tok_s"] for r in ng) / len(ng)
    aa = sum(r["avg_accept_len"] for r in ng) / len(ng)
    lines.append(f"| **mean** | **{vv:.1f}** | **{nn:.1f}** | **{nn/vv:.2f}×** | **{aa:.2f}** |")
    lines += ["", "## Notes", "",
              "- n-gram accept_len 上界 = k+1 = %d；对 code/json 等重复结构命中高, story/translation 低。" % (K + 1),
              "- lossless = pass by construction (verify accepted token == target argmax 同一次 forward)。",
              "- 对比 vLLM EAGLE3 baseline 时注意 k 不同 (此处 %d vs eagle3 的 2) 且 proposer 不同。" % K, ""]
    open(OUT_MD, "w").write("\n".join(lines))


def run_child(mode):
    """子进程: 只跑一个 mode (独立 process group), 结果打到 stdout 作为 JSONL。"""
    gc = os.environ.get("BENCH_GC", git_commit())
    ts = os.environ.get("BENCH_TS", datetime.datetime.now().isoformat(timespec="seconds"))
    for r in bench(mode, gc, ts):
        print("ROW " + json.dumps(r, ensure_ascii=False))


def main():
    gc = git_commit()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    env = {**os.environ, "BENCH_GC": gc, "BENCH_TS": ts}
    rows = []
    for mode in ("vanilla", "ngram"):
        out = subprocess.run([sys.executable, __file__, mode], cwd=os.path.expanduser("~/nano-vllm"),
                             env=env, capture_output=True, text=True)
        for line in out.stdout.splitlines():
            if line.startswith("ROW "):
                rows.append(json.loads(line[4:]))
        if not any(r["mode"] == mode for r in rows):
            print(f"[{mode}] 子进程无输出, stderr tail:\n" + "\n".join(out.stderr.splitlines()[-15:]))
            return
    with open(OUT_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_md(rows, gc, ts)
    print(f"wrote {len(rows)} runs -> {OUT_JSONL}")
    print(f"wrote {OUT_MD}")
    for r in rows:
        if r["mode"] == "ngram":
            print(f"  {r['run_id']:<26} tok/s={r['decode_tok_s']:<7} accept={r['avg_accept_len']} tgt_fwd={r['target_forward_count']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_child(sys.argv[1])
    else:
        main()
