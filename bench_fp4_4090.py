"""Benchmark: Qwen3-32B NVFP4 weights + fp4 KV, EAGLE-3 vs no-spec, on a 4090 (sm_89).

Metrics per config (bs=1, self-timed by driving step()):
  - throughput  : decode tokens / decode wall-time (tok/s)
  - TTFT        : time to first token (ms)
  - ITL         : inter-token latency p50/p99 (ms)
  - accept_len  : (gen-1) / target_forwards   (eagle3 only; 1.0 for no-spec)

Configs (bf16-draft-KV dropped per Step 0 — draft-KV quant is free, mixed pool not worth it):
  - no_spec   : nvfp4 weights + fp4 KV, no speculation      -> baseline throughput/TTFT/ITL
  - eagle3    : nvfp4 weights + fp4 KV + EAGLE-3            -> the full system

Real kernels are wired in (locally validated on a 3060 Ti):
  * NVFP4 weight-only fused dequant GEMM (layers/nvfp4_gemm.py) — packed uint8 weights stay resident
    so the 32B fits; dequant to fp16 happens inside the GEMM.
  * fp4/fp8 paged KV: quant store + a dequant-in-loop paged-attention read kernel for decode AND
    verify/prefill (layers/paged_attn.py).
Prereqs on the 4090 (see fp4_spec_4090_plan.md):
  * Download the PACKED nvfp4 checkpoint (weights are uint8, loaded directly — no on-the-fly packing):
        huggingface-cli download AngelSlim/Qwen3-32B_nvfp4  --local-dir ~/huggingface/Qwen3-32B_nvfp4
        huggingface-cli download AngelSlim/Qwen3-32B_eagle3 --local-dir ~/huggingface/Qwen3-32B_eagle3
  * fp4 KV runs on any arch; fp8 (e4m3) KV needs sm_89 (the 4090) — Triton can't compile e4m3 on sm_86.
  * VRAM is tight: ~18 GB packed weights + ~3 GB embed/lm_head + eagle3 head + KV. If OOM, lower
    gpu_memory_utilization / max_model_len (already 4096), or run no_spec (no head) first.
Run:  python bench_fp4_4090.py
"""
import os, json, time, statistics
import multiprocessing as mp
import torch
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-32B_nvfp4/")   # AngelSlim packed NVFP4 (W4A4, weight-only here)
DRAFT = os.path.expanduser("~/huggingface/Qwen3-32B_eagle3/")
K, MAX_TOKENS, TEMP, SEED = 2, 512, 0.6, 0

PROMPTS = [
    ("code",  "Write a Python function that merges two sorted lists into one sorted list, with tests."),
    ("qa",    "Explain how a transformer attention mechanism works, step by step."),
    ("math",  "A train travels 60 km in 45 minutes. What is its average speed in km/h? Show your work."),
    ("story", "Write a short story about a lighthouse keeper who discovers a message in a bottle."),
]

# weight_scale_fp8=True: store the NVFP4 per-group scale as fp8_e4m3 (native ckpt dtype) not fp16 ->
# ~2 GB less weight VRAM (32B weights ~21->19 GB), lossless. Needed to fit eagle3's draft head on 24 GB.
CONFIGS = {
    "no_spec": dict(weight_quant="nvfp4", weight_scale_fp8=True, kv_quant_target="fp4"),
    "eagle3":  dict(weight_quant="nvfp4", weight_scale_fp8=True, kv_quant_target="fp4", kv_quant_draft="fp4",
                    enable_spec=True, num_spec_tokens=K, spec_method="eagle3", draft_model=DRAFT),
}


def pct(xs, q):
    return statistics.quantiles(xs, n=100)[q - 1] if len(xs) > 1 else (xs[0] if xs else 0.0)


def run(cfg_name):
    cfg = CONFIGS[cfg_name]
    # 24 GB is tight: max_num_batched_tokens caps the warmup/prefill activation spike (default 16384
    # OOMs); fp8 weight scales (in CONFIGS) free ~2 GB so eagle3's head fits. Raise gpu_memory_utilization
    # or lower max_model_len if KV still won't allocate; drop max_num_batched_tokens to 1024 if warmup OOMs.
    llm = LLM(MODEL, enforce_eager=True, max_model_len=2048, max_num_batched_tokens=2048, **cfg)
    tok = llm.tokenizer
    rows = []
    for pid, text in PROMPTS:
        prompt = tok.apply_chat_template([{"role": "user", "content": text}],
                                         tokenize=False, add_generation_prompt=True)
        torch.manual_seed(SEED)
        llm.add_request(prompt, SamplingParams(temperature=TEMP, max_tokens=MAX_TOKENS))
        seq = llm.scheduler.waiting[-1]

        t0 = time.perf_counter()
        ttft = None
        stamps = []            # wall-clock at each step that produced >=1 new token
        target_fwd = 0
        last_n = 0
        while not llm.is_finished():
            _, num_tokens = llm.step()
            if num_tokens < 0:                       # decode/verify forward
                target_fwd += 1
            n = seq.num_completion_tokens
            if n > last_n:
                now = time.perf_counter()
                if ttft is None:
                    ttft = (now - t0) * 1000
                stamps.append(now)
                last_n = n
        gen = seq.num_completion_tokens
        # ITL from the decode-phase stamps (skip the first = prefill/TTFT boundary)
        itl = [(stamps[i] - stamps[i - 1]) * 1000 for i in range(1, len(stamps))]
        decode_s = (stamps[-1] - stamps[0]) if len(stamps) > 1 else 1e-9
        rows.append(dict(
            pid=pid, gen=gen,
            throughput_tok_s=round((gen - 1) / decode_s, 2),
            ttft_ms=round(ttft or 0, 2),
            itl_p50_ms=round(pct(itl, 50), 2), itl_p99_ms=round(pct(itl, 99), 2),
            accept_len=round((gen - 1) / target_fwd, 4) if target_fwd else 1.0,
        ))
        print(f"  [{cfg_name}/{pid}] tok/s={rows[-1]['throughput_tok_s']} "
              f"ttft={rows[-1]['ttft_ms']}ms itl_p50={rows[-1]['itl_p50_ms']}ms "
              f"accept={rows[-1]['accept_len']}")
    del llm
    return rows


def _worker(name, q):
    q.put(run(name))


def main():
    # Each config runs in its own spawned process: a second LLM in the same process would re-init the
    # torch process group ("initialize... twice") and, worse, the first 20 GB model's VRAM would linger.
    ctx = mp.get_context("spawn")
    results = {}
    for name in CONFIGS:
        print(f"=== {name} ===")
        q = ctx.Queue()
        p = ctx.Process(target=_worker, args=(name, q))
        p.start()
        p.join()
        if p.exitcode == 0 and not q.empty():
            results[name] = q.get()
        else:
            print(f"  !! {name} failed (exit {p.exitcode}); see traceback above — skipping")
            results[name] = []
    results = {k: v for k, v in results.items() if v}
    # aggregate
    def agg(rows, key):
        return round(statistics.mean(r[key] for r in rows), 2)
    lines = ["# Qwen3-32B NVFP4 + fp4 KV + EAGLE-3 (4090)", "",
             "| config | tok/s | TTFT ms | ITL p50 | ITL p99 | accept_len |",
             "|---|---|---|---|---|---|"]
    for name, rows in results.items():
        lines.append(f"| {name} | {agg(rows,'throughput_tok_s')} | {agg(rows,'ttft_ms')} | "
                     f"{agg(rows,'itl_p50_ms')} | {agg(rows,'itl_p99_ms')} | {agg(rows,'accept_len')} |")
    if "no_spec" in results and "eagle3" in results:
        sp = agg(results["eagle3"], "throughput_tok_s") / max(agg(results["no_spec"], "throughput_tok_s"), 1e-9)
        lines += ["", f"**EAGLE-3 speedup: {sp:.2f}x** (decode throughput vs no-spec, same nvfp4+fp4-KV target)."]
    md = "\n".join(lines)
    print("\n" + md)
    open("fp4_4090_benchmark.md", "w").write(md + "\n")
    json.dump(results, open("fp4_4090_benchmark.jsonl", "w"), indent=2)
    print("\nsaved fp4_4090_benchmark.md + .jsonl")


if __name__ == "__main__":
    main()
