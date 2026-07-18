"""
Benchmark: priority scheduling (fast vs normal).  独立于 bench_mixedbatch.py, 不影响 mixbatch 测试.

scenario:
  - 1000 prompts, 长度三档 (短/中/长), online Poisson arrival -> waiting 排队 (priority 才有意义)
  - 随机 20% 标为 fast(interactive), 其余 normal(batch); mask 固定, A/B 共用同一批请求
  - A=baseline: 全部 priority=0 (无分级)
  - B=priority : fast 子集 priority=1, 其余 0
  - 对齐比同一批 fast 请求的 TTFT (A 当普通排队 vs B 插队); 看 normal 是否被 deadline 保底不饿死

用法 (建议先锁频 sudo nvidia-smi -lgc <freq>):
    python3 bench_priority.py                 # rate=30, fast=20%, reps=1
    python3 bench_priority.py --rate 40 --reps 3
    python3 bench_priority.py --compare-only  # 复用 base.json/prio.json 只出指标+图
"""
import os
import json
import time
import argparse
from random import randint, seed, shuffle, Random

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
MAX_TOKENS = 128
N_PROMPTS = 1000
BUCKETS = [(400, (40, 120)),      # 短
           (400, (200, 700)),     # 中
           (200, (1200, 2000))]   # 长
CHUNK = 2048                       # mixbatch 测出的好配置
DEADLINE = 10.0                    # normal 防饿死阈值 (秒); 传给 config.priority_deadline


def gen_prompts():
    seed(0)
    prompts = []
    for count, (lo, hi) in BUCKETS:
        prompts += [[randint(0, 10000) for _ in range(randint(lo, hi))] for _ in range(count)]
    shuffle(prompts)
    return prompts


def gen_fast_mask(n, frac, seed_val=2):
    """随机选 frac 比例为 fast(True). 固定, A/B 共用."""
    rng = Random(seed_val)
    mask = [rng.random() < frac for _ in range(n)]
    return mask


def gen_arrivals(n, rate, seed_val=1):
    if rate <= 0:
        return [0.0] * n
    rng = Random(seed_val)
    t, arr = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        arr.append(t)
    return arr


def _run_once(llm, prompts, sp, arrivals):
    llm.collect_metrics = True
    llm.reset_metrics()
    n = len(prompts)
    idx = 0
    t0 = time.perf_counter()
    while idx < n or not llm.is_finished():
        now = time.perf_counter() - t0
        while idx < n and arrivals[idx] <= now:
            llm.add_request(prompts[idx], sp[idx])
            idx += 1
        if llm.is_finished():
            if idx < n:
                time.sleep(max(0.0, arrivals[idx] - (time.perf_counter() - t0)))
            continue
        llm.step()
    wall = time.perf_counter() - t0
    ids = sorted(llm.token_times.keys())        # = 提交顺序 = prompt index
    return {
        "wall": wall,
        "submit": [llm.submit_time[i] for i in ids],
        "token_times": [llm.token_times[i] for i in ids],
    }


def _worker(use_priority, prompts, fast_mask, arrivals, reps, chunk, deadline, out_path):
    """一个独立进程跑一个 mode. use_priority=False -> 全 normal; True -> fast 子集 priority=1."""
    from nanovllm import LLM, SamplingParams
    sp = []
    for is_fast in fast_mask:
        prio = 1 if (use_priority and is_fast) else 0
        sp.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=MAX_TOKENS, priority=prio))
    llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=4096,
              enable_mixbatch=True, max_num_batched_tokens=chunk, priority_deadline=deadline)

    llm.collect_metrics = False
    llm.generate(prompts, sp, use_tqdm=False)   # 热身丢弃

    reps_data = [_run_once(llm, prompts, sp, arrivals) for _ in range(reps)]
    for r, rep in enumerate(reps_data):
        print(f"  [priority={use_priority}] rep {r}: wall {rep['wall']:.2f}s")
    json.dump({"use_priority": use_priority, "max_tokens": MAX_TOKENS,
               "fast_mask": fast_mask, "reps": reps_data}, open(out_path, "w"))


def _pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _tier(rep, fast_mask, want_fast):
    """取某一档(fast/normal) 的 TTFT 和 ITL (ms)"""
    ttfts, itls = [], []
    for is_fast, sub, tt in zip(fast_mask, rep["submit"], rep["token_times"]):
        if is_fast != want_fast or not tt:
            continue
        ttfts.append((tt[0] - sub) * 1000)
        itls += [(b - a) * 1000 for a, b in zip(tt, tt[1:])]
    return ttfts, itls


def _collect(fn):
    d = json.load(open(fn))
    fm = d["fast_mask"]
    out = {}
    for tier, want in [("fast", True), ("normal", False)]:
        ttfts, itls = [], []
        for rep in d["reps"]:
            t, i = _tier(rep, fm, want)
            ttfts += t; itls += i
        out[tier] = dict(ttft=ttfts, itl=itls)
    return out


def compare(base_file, prio_file):
    A = _collect(base_file)   # baseline (无分级)
    B = _collect(prio_file)   # priority

    def line(metric, tier):
        a, b = A[tier][metric], B[tier][metric]
        chg = (_pct(b, 99) / _pct(a, 99) - 1) * 100 if _pct(a, 99) else 0
        print(f"  [{tier:6}] {metric.upper()} p50: {_pct(a,50):8.1f} -> {_pct(b,50):8.1f}   "
              f"p99: {_pct(a,99):8.1f} -> {_pct(b,99):8.1f}  ({chg:+.1f}%)")

    print("\n=== TTFT (ms): baseline -> priority ===")
    line("ttft", "fast"); line("ttft", "normal")
    print("=== ITL (ms): baseline -> priority ===")
    line("itl", "fast"); line("itl", "normal")
    print("=== starvation check (normal TTFT max, ms) ===")
    print(f"  baseline {max(A['normal']['ttft']):.0f}  ->  priority {max(B['normal']['ttft']):.0f}   "
          f"(deadline={DEADLINE*1000:.0f}ms)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib 没装 -> 跳过画图, 指标已完整)")
        return

    for tier, fname in [("fast", "prio_ttft_fast.png"), ("normal", "prio_ttft_normal.png")]:
        plt.figure()
        for lbl, D in [("baseline", A), ("priority", B)]:
            x = sorted(D[tier]["ttft"]); y = [i / len(x) for i in range(len(x))]
            plt.plot(x, y, label=lbl)
        plt.xlabel("TTFT (ms)"); plt.ylabel("CDF"); plt.legend()
        plt.title(f"TTFT CDF — {tier} tier"); plt.grid(alpha=0.3)
        plt.savefig(fname, dpi=120); plt.close()
    print("\nsaved: prio_ttft_fast.png, prio_ttft_normal.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=30.0, help="到达速率 req/s; 要够高才会排队")
    ap.add_argument("--fast-frac", type=float, default=0.2, help="fast 请求比例")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--deadline", type=float, default=DEADLINE)
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    if args.compare_only:
        compare("base.json", "prio.json")
        return

    print("[reminder] lock GPU clock: sudo nvidia-smi -lgc <freq>")
    import torch.multiprocessing as mp
    prompts = gen_prompts()
    fast_mask = gen_fast_mask(len(prompts), args.fast_frac)
    arrivals = gen_arrivals(len(prompts), args.rate)
    print(f"prompts={len(prompts)}, fast={sum(fast_mask)}, rate={args.rate}, chunk={args.chunk}, deadline={args.deadline}")
    ctx = mp.get_context("spawn")
    for use_prio, out in [(False, "base.json"), (True, "prio.json")]:
        p = ctx.Process(target=_worker, args=(use_prio, prompts, fast_mask, arrivals,
                                              args.reps, args.chunk, args.deadline, out))
        p.start(); p.join()
        assert p.exitcode == 0, f"worker (priority={use_prio}) failed"
    compare("base.json", "prio.json")


if __name__ == "__main__":
    main()
