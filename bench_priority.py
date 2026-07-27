"""
Benchmark: 3-tier priority scheduling + anti-starvation aging.

tiers (SamplingParams.priority, higher served first):
  P2 premium 10%  | P1 standard 30%  | P0 batch 60%     (traffic mix, fixed across all runs)

scenario:
  - online Poisson arrivals -> the waiting queue builds up (priority only matters under contention)
  - each run compares, on the SAME request set + arrivals, two schedulers:
      baseline : every request priority=0 (pure FCFS by arrival)
      priority : request gets its tier priority; aging bumps a starving seq up one level every
                 `deadline` seconds (scheduler._priority), capped at the top tier
  - sweep over several arrival rates; one extra run at the top rate with aging OFF to expose the
    P0 starvation that aging prevents.

metrics, per tier: TTFT p50/p99, ITL p50; plus P0 max-TTFT (the starvation indicator) vs 2*deadline.
outputs: prio_sweep_raw.json (all reps) -> prio_sweep.jsonl (one row per rate/scheduler/tier) + prio_sweep.md.

usage (lock GPU clock first: sudo nvidia-smi -lgc <freq>):
    python3 bench_priority.py --rates 1,2,4 --deadline 4        # run the sweep
    python3 bench_priority.py --compare-only                    # rebuild table from prio_sweep_raw.json
"""
import os
import json
import time
import argparse
from random import randint, seed, shuffle, Random

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
MAX_TOKENS = 128
BUCKETS = [(200, (40, 120)),       # short
           (150, (200, 700)),      # medium
           (50, (800, 1400))]      # long
CHUNK = 2048
FRACS = (0.60, 0.30, 0.10)         # P0 batch, P1 standard, P2 premium
TIER_NAME = {0: "P0-batch", 1: "P1-std", 2: "P2-prem"}


def gen_prompts():
    seed(0)
    prompts = []
    for count, (lo, hi) in BUCKETS:
        prompts += [[randint(0, 10000) for _ in range(randint(lo, hi))] for _ in range(count)]
    shuffle(prompts)
    return prompts


def gen_class(n, fracs=FRACS, seed_val=2):
    """Assign each request a tier in {0,1,2} by cumulative fracs. Fixed seed -> shared across runs."""
    rng = Random(seed_val)
    c0, c1 = fracs[0], fracs[0] + fracs[1]
    cls = []
    for _ in range(n):
        r = rng.random()
        cls.append(0 if r < c0 else 1 if r < c1 else 2)
    return cls


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
    ids = sorted(llm.token_times.keys())        # = submission order = prompt index
    return {
        "wall": wall,
        "submit": [llm.submit_time[i] for i in ids],
        "token_times": [llm.token_times[i] for i in ids],
    }


def _sp_for(mode, cls):
    """Build the SamplingParams list. baseline -> all priority 0; priority -> per-tier priority."""
    from nanovllm import SamplingParams
    out = []
    for c in cls:
        prio = c if mode == "priority" else 0
        out.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=MAX_TOKENS, priority=prio))
    return out


def _worker(runs, prompts, cls, reps, chunk, out_path):
    """Single process, model loaded once; each run mutates scheduler.priority_deadline + sp list."""
    from nanovllm import LLM
    llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=4096,
              max_num_batched_tokens=chunk, max_num_seqs=64)
    llm.collect_metrics = False
    llm.generate(prompts, _sp_for("baseline", cls), use_tqdm=False)   # warm-up (discarded)

    results = []
    for run in runs:
        llm.scheduler.priority_deadline = run["deadline"]
        sp = _sp_for(run["mode"], cls)
        arrivals = gen_arrivals(len(prompts), run["rate"])
        reps_data = [_run_once(llm, prompts, sp, arrivals) for _ in range(reps)]
        for r, rep in enumerate(reps_data):
            print(f"  [{run['tag']}] rep {r}: wall {rep['wall']:.2f}s", flush=True)
        results.append({**run, "reps": reps_data})
    json.dump({"max_tokens": MAX_TOKENS, "cls": cls, "runs": results}, open(out_path, "w"))


def _pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _tier_metrics(run, cls, want):
    """Collect TTFT/ITL (ms) for tier `want` across all reps of a run."""
    ttfts, itls = [], []
    for rep in run["reps"]:
        for c, sub, tt in zip(cls, rep["submit"], rep["token_times"]):
            if c != want or not tt:
                continue
            ttfts.append((tt[0] - sub) * 1000)
            itls += [(b - a) * 1000 for a, b in zip(tt, tt[1:])]
    return ttfts, itls


def _throughput(run):
    """Aggregate decode throughput = total generated tokens / mean wall over reps."""
    toks = sum(sum(len(tt) for tt in rep["token_times"]) for rep in run["reps"])
    wall = sum(rep["wall"] for rep in run["reps"])
    return toks / wall if wall else 0.0


def compare(raw_path):
    d = json.load(open(raw_path))
    cls = d["cls"]
    rows = []
    for run in d["runs"]:
        for tier in (2, 1, 0):
            ttft, itl = _tier_metrics(run, cls, tier)
            rows.append({
                "rate": run["rate"], "scheduler": run["tag"], "deadline": run["deadline"],
                "tier": TIER_NAME[tier], "n": len(ttft),
                "ttft_p50": round(_pct(ttft, 50), 1), "ttft_p99": round(_pct(ttft, 99), 1),
                "ttft_max": round(max(ttft), 1) if ttft else 0.0,
                "itl_p50": round(_pct(itl, 50), 1),
                "throughput": round(_throughput(run), 1),
            })
    with open("prio_sweep.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # markdown table + headline (P2 p99 reduction priority-vs-baseline per rate; P0 starvation)
    lines = ["# Priority scheduling sweep (Qwen3-1.7B, 3060 Ti)", "",
             f"tiers: P2 premium {FRACS[2]:.0%} / P1 standard {FRACS[1]:.0%} / P0 batch {FRACS[0]:.0%}; "
             f"prompts={sum(c for c, _ in BUCKETS)}, max_tokens={MAX_TOKENS}", "",
             "| rate | scheduler | tier | TTFT p50 | TTFT p99 | TTFT max | ITL p50 | thrpt tok/s |",
             "|------|-----------|------|----------|----------|----------|---------|-------------|"]
    for r in rows:
        lines.append(f"| {r['rate']} | {r['scheduler']} | {r['tier']} | {r['ttft_p50']} | "
                     f"{r['ttft_p99']} | {r['ttft_max']} | {r['itl_p50']} | {r['throughput']} |")

    lines += ["", "## Headline", ""]
    def get(rate, sched, tier, field):
        for r in rows:
            if r["rate"] == rate and r["scheduler"] == sched and r["tier"] == tier:
                return r[field]
        return None
    rates = sorted({r["rate"] for r in rows})
    for rate in rates:
        base = get(rate, "baseline", "P2-prem", "ttft_p99")
        prio = get(rate, "priority", "P2-prem", "ttft_p99")
        if base and prio:
            chg = (prio / base - 1) * 100
            lines.append(f"- rate {rate}: **P2 premium p99 TTFT {base:.0f} -> {prio:.0f} ms "
                         f"({chg:+.1f}%)** (baseline FCFS -> priority+aging)")
    # starvation: P0 max under aging vs no-aging at the top rate
    top = max(rates)
    p0_aging = get(top, "priority", "P0-batch", "ttft_max")
    p0_noage = get(top, "no-aging", "P0-batch", "ttft_max")
    if p0_aging is not None:
        lines.append(f"- rate {top}: **P0 batch max-TTFT** aging {p0_aging:.0f} ms"
                     + (f" vs no-aging {p0_noage:.0f} ms" if p0_noage is not None else "")
                     + " (starvation bound)")

    open("prio_sweep.md", "w").write("\n".join(lines))
    print("\n".join(lines))
    print("\nsaved: prio_sweep.jsonl, prio_sweep.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=str, default="1,2,4", help="comma-separated arrival rates req/s")
    ap.add_argument("--fast-frac", type=float, default=None, help="unused (kept for compat)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--deadline", type=float, default=4.0, help="aging step (s) -> config.priority_deadline")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    if args.compare_only:
        compare("prio_sweep_raw.json")
        return

    print("[reminder] lock GPU clock: sudo nvidia-smi -lgc <freq>")
    import torch.multiprocessing as mp
    prompts = gen_prompts()
    cls = gen_class(len(prompts))
    counts = {TIER_NAME[t]: cls.count(t) for t in (2, 1, 0)}
    rates = [float(x) for x in args.rates.split(",")]

    # permutation matrix: each rate gets baseline (FCFS) + priority (aging on);
    # the top rate additionally gets a no-aging run to expose P0 starvation.
    runs = []
    for rate in rates:
        runs.append(dict(tag="baseline", mode="baseline", rate=rate, deadline=args.deadline))
        runs.append(dict(tag="priority", mode="priority", rate=rate, deadline=args.deadline))
    top = max(rates)
    runs.append(dict(tag="no-aging", mode="priority", rate=top, deadline=0.0))

    print(f"prompts={len(prompts)} {counts}, rates={rates}, chunk={args.chunk}, deadline={args.deadline}")
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_worker, args=(runs, prompts, cls, args.reps, args.chunk, "prio_sweep_raw.json"))
    p.start(); p.join()
    assert p.exitcode == 0, "worker failed"
    compare("prio_sweep_raw.json")


if __name__ == "__main__":
    main()
