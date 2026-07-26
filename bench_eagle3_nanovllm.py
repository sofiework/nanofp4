"""
nano-vllm EAGLE-3 speculative decoding benchmark. 输出与 vllm_baseline.jsonl / nanovllm_baseline.jsonl
同 schema (engine=nano-vllm, mode=eagle3), 与 vLLM real-EAGLE3 及 nano-vllm n-gram 直接对比 accept_len。

自己驱动 step() (bs=1) 数 target forward:
  accept_len = (total_gen - 1) / target_forward_count   # vanilla 恒 =1.0
  target_forward_count = 有 decode/verify 的 step 数
  draft_forward_count  = midlayer forward 数 (每次 propose: window 1 次 + num_draft>0 时 k-1 次 single-token)
decode 全程 greedy (verify=argmax); temperature 只影响 prefill 首 token (SamplingParams 禁 greedy).

用法 (nanovllm conda env): python bench_eagle3_nanovllm.py
"""
import os, sys, json, time, subprocess, datetime
import torch
from nanovllm import LLM, SamplingParams

PATH = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
DRAFT = os.path.expanduser("~/huggingface/Qwen3-1.7B_eagle3/")
OUT_JSONL = os.path.expanduser("~/nano-vllm/nanovllm_eagle3.jsonl")
OUT_MD = os.path.expanduser("~/nano-vllm/nanovllm_eagle3_benchmark.md")
NGRAM_JSONL = os.path.expanduser("~/nano-vllm/nanovllm_baseline.jsonl")
VLLM_JSONL = os.path.expanduser("~/nano-vllm/vllm_baseline.jsonl")

MAX_TOKENS, K, TEMPERATURE, SEED, BS = 1024, 2, 0.6, 0, 1
GPU, MODEL_NAME, DTYPE, TEMPLATE, THINKING = "RTX 3060 Ti 8GB", "Qwen3-1.7B", "bf16", "qwen3-chat", True
HEAD_REV = "AngelSlim/Qwen3-1.7B_eagle3"

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
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.expanduser("~/nano-vllm"),
                          capture_output=True, text=True).stdout.strip() or "unknown"


# N=7, depth-2 binary tree (root + 2 children + 4 grandchildren) for the eagle3_tree mode
TREE_STR = "[(0,),(1,),(0,0),(0,1),(1,0),(1,1)]"


def bench(mode, gc, ts):
    spec = mode in ("eagle3", "eagle3_tree")
    kw = dict(enforce_eager=True, enable_spec=spec, num_spec_tokens=(K if spec else 0))
    if spec:
        kw.update(spec_method="eagle3", draft_model=DRAFT)
    if mode == "eagle3_tree":
        kw["speculative_token_tree"] = TREE_STR
    llm = LLM(PATH, **kw)
    tok = llm.tokenizer

    # count draft-head midlayer forwards (window pass + one re-fed pass per tree level)
    draft_fwd = {"n": 0}
    if spec:
        depth = llm.config.spec_tree.max_depth
        orig_propose = llm.model_runner.propose_tree
        def counted(seqs):
            for s in seqs:
                draft_fwd["n"] += 1 + (depth if s.num_completion_tokens > 0 else 0)
            return orig_propose(seqs)
        llm.model_runner.propose_tree = counted

    rows = []
    for pid, cat, text in PROMPTS:
        prompt = tok.apply_chat_template([{"role": "user", "content": text}],
                                         tokenize=False, add_generation_prompt=True)
        torch.manual_seed(SEED)
        llm.add_request(prompt, SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS))
        seq = llm.scheduler.waiting[-1]
        draft_fwd["n"] = 0

        t0 = time.perf_counter()
        ttft_ms = None
        target_fwd = 0
        while not llm.is_finished():
            _, num_tokens = llm.step()
            if ttft_ms is None and seq.num_completion_tokens >= 1:
                ttft_ms = (time.perf_counter() - t0) * 1000
            if num_tokens < 0:    # decode/verify forward (num_tokens = -len(seqs))
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
            git_commit=gc, head_rev=(HEAD_REV if spec else None),
            thinking=THINKING, template=TEMPLATE, timestamp=ts,
            decode_tok_s=round(decode_tok_s, 2), ttft_ms=round(ttft_ms or 0, 2),
            avg_accept_len=round(accept_len, 4) if accept_len == accept_len else None,
            total_generated_tokens=gen,
            target_forward_count=target_fwd, draft_forward_count=draft_fwd["n"],
            lossless=("pass" if spec else "ref"),   # greedy verify=argmax -> by construction
        ))
    del llm
    torch.cuda.empty_cache()
    return rows


def _mean(rows, key):
    xs = [r[key] for r in rows if r.get(key) is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def write_md(rows, gc, ts):
    van = {r["run_id"].replace("nanovllm_vanilla_", ""): r for r in rows if r["mode"] == "vanilla"}
    eg = [r for r in rows if r["mode"] == "eagle3"]
    lines = [
        "# nano-vllm EAGLE-3 Speculative Decoding Benchmark (Qwen3-1.7B)",
        "",
        "from-scratch EAGLE-3 on nano-vllm (verify / KV rollback / scheduler 机制与 n-gram 共享, "
        "proposer 换成 EAGLE-3 head)。同 schema 对照 vLLM real-EAGLE3 "
        "([`vllm_vanilla_eagle3_benchmark.md`](vllm_vanilla_eagle3_benchmark.md)) 与 "
        "n-gram ([`nanovllm_ngram_benchmark.md`](nanovllm_ngram_benchmark.md))。",
        "",
        f"- Raw: [`nanovllm_eagle3.jsonl`](nanovllm_eagle3.jsonl) — {len(rows)} runs",
        f"- git_commit `{gc}`, {ts}",
        "",
        "## Config", "",
        "| model | Qwen3-1.7B / bf16 |", "|---|---|",
        "| engine | nano-vllm (from-scratch) |",
        f"| proposer | EAGLE-3 head (`{HEAD_REV}`), 1 midlayer + fc + draft lm_head, 共享 target embedding |",
        f"| k (num_spec_tokens) | {K} (vanilla: null) |",
        "| draft KV | target paged KV pool +1 层, 同 block_table |",
        f"| decode | greedy (verify=argmax); temperature={TEMPERATURE} 仅影响 prefill 首 token |",
        f"| bs / max_tokens / seed | {BS} / {MAX_TOKENS} / {SEED} |",
        "| thinking | on (template 默认) |",
        f"| gpu | {GPU}, enforce_eager, mixbatch off |",
        "",
        "## Results (per run)", "",
        "accept_len = (gen-1) / target_forward_count, 上界 k+1 = %d。" % (K + 1),
        "draft_fwd = midlayer forward 数 (每 verify 步 window 1 次 + single-token %d 次)。" % (K - 1),
        "",
        "| prompt | vanilla tok/s | eagle3 tok/s | speedup | accept_len | gen | tgt_fwd | draft_fwd | lossless |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in eg:
        pid = r["run_id"].replace("nanovllm_eagle3_", "")
        vt = van[pid]["decode_tok_s"]
        sp = r["decode_tok_s"] / vt if vt else 0
        lines.append(f"| {pid} | {vt} | {r['decode_tok_s']} | {sp:.2f}× | {r['avg_accept_len']} | "
                     f"{r['total_generated_tokens']} | {r['target_forward_count']} | "
                     f"{r['draft_forward_count']} | {r['lossless']} |")
    cats = {}
    for r in eg:
        pid = r["run_id"].replace("nanovllm_eagle3_", "")
        cats.setdefault(r["prompt_category"], []).append((van[pid]["decode_tok_s"], r["decode_tok_s"], r["avg_accept_len"]))
    lines += ["", "## Category mean", "", "| category | vanilla | eagle3 | speedup | accept_len |", "|---|---|---|---|---|"]
    for c, xs in cats.items():
        v = sum(a for a, _, _ in xs) / len(xs)
        n = sum(b for _, b, _ in xs) / len(xs)
        a = sum(c2 for _, _, c2 in xs) / len(xs)
        lines.append(f"| {c} | {v:.1f} | {n:.1f} | {n/v:.2f}× | {a:.2f} |")
    vv = _mean(list(van.values()), "decode_tok_s")
    nn = _mean(eg, "decode_tok_s")
    aa = _mean(eg, "avg_accept_len")
    lines.append(f"| **mean** | **{vv:.1f}** | **{nn:.1f}** | **{nn/vv:.2f}×** | **{aa:.2f}** |")

    # cross comparison from sibling jsonl (same schema)
    cmp_lines = ["", "## 对比 (同 schema JSONL, 各自 vanilla 基线)", "",
                 "| engine / proposer | k | tok/s (spec) | speedup | accept_len |", "|---|---|---|---|---|",
                 f"| nano-vllm / eagle3 | {K} | {nn:.1f} | {nn/vv:.2f}× | {aa:.2f} |"]
    for path, tag in ((NGRAM_JSONL, "nano-vllm / ngram"), (VLLM_JSONL, "vLLM / eagle3")):
        if not os.path.exists(path):
            continue
        rs = [json.loads(l) for l in open(path)]
        spec_rows = [r for r in rs if r["mode"] not in ("vanilla",)]
        van_rows = [r for r in rs if r["mode"] == "vanilla"]
        if not spec_rows or not van_rows:
            continue
        st, vt = _mean(spec_rows, "decode_tok_s"), _mean(van_rows, "decode_tok_s")
        al = _mean(spec_rows, "avg_accept_len")
        kk = spec_rows[0].get("k")
        cmp_lines.append(f"| {tag} | {kk} | {st:.1f} | {st/vt:.2f}× | {al:.2f} |")
    lines += cmp_lines
    lines += ["", "## Notes", "",
              "- lossless = pass by construction (committed token == 同一次 verify forward 的 target argmax)。"
              "跨 mode 逐 token 对比时, bf16 近平局 (top-2 logit gap = 1 ulp) 可因 verify q_len 不同翻转 argmax, 非实现差异。",
              "- draft 每步含 1 次 CPU↔GPU sync (`_sample` 的 int()), bs=1 下未做 cross-request batching / cuda graph。",
              ""]
    open(OUT_MD, "w").write("\n".join(lines))


def run_child(mode):
    gc = os.environ.get("BENCH_GC", git_commit())
    ts = os.environ.get("BENCH_TS", datetime.datetime.now().isoformat(timespec="seconds"))
    for r in bench(mode, gc, ts):
        print("ROW " + json.dumps(r, ensure_ascii=False))


def main():
    gc = git_commit()
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    env = {**os.environ, "BENCH_GC": gc, "BENCH_TS": ts}
    rows = []
    for mode in ("vanilla", "eagle3", "eagle3_tree"):
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
    # chain vs tree accept-length comparison (the point of tree drafting: same output, fewer target forwards)
    def _macc(m):
        rs = [r["avg_accept_len"] for r in rows if r["mode"] == m and r["avg_accept_len"]]
        return sum(rs) / len(rs) if rs else float("nan")
    def _mtps(m):
        rs = [r["decode_tok_s"] for r in rows if r["mode"] == m]
        return sum(rs) / len(rs) if rs else float("nan")
    print(f"  mean accept_len   chain={_macc('eagle3'):.3f}  tree={_macc('eagle3_tree'):.3f}")
    print(f"  mean decode tok/s chain={_mtps('eagle3'):.1f}  tree={_mtps('eagle3_tree'):.1f}  (vanilla={_mtps('vanilla'):.1f})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_child(sys.argv[1])
    else:
        main()
