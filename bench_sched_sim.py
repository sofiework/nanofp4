"""
Scheduler simulation: online serving trace + full metric set.

engine under test: mixed batching (decode + chunked prefill co-scheduled per iteration) +
3-tier priority scheduling with wait-time aging.

workload (small-team serving scenario):
  - 500 requests, prompt lengths from 3 buckets (short chat / doc context / long doc)
  - tiers P2 premium 10% | P1 standard 30% | P0 batch 60%, fixed across all runs
  - open-loop Poisson arrivals at a configurable rate; the trace is precomputed and replayed
    identically by every arm, so the offered load never shrinks when the engine falls behind
  - each rate runs two arms on the SAME requests + arrivals:
      baseline : every request priority=0 (pure FCFS by arrival)
      priority : per-tier priority, aging promotes a starving seq one level every `deadline` s

metrics (per tier and overall):
  latency    queueing delay p95/max, TTFT p50/p95/p99, TPOT p50/p95/p99, max-ITL p99, E2E p95/p99
  throughput output token throughput
  fairness   max P0 queueing delay (bounded <=> aging works)
  internals  per-iteration running/waiting depth, batch size (seqs, tokens), KV-block utilization,
             preemption rate  -> shows which cap binds and whether the engine is thrashing

all latencies are measured from the NOMINAL trace arrival (open-loop), not from the moment the
driver managed to inject the request; injection lag is reported separately as a driver-health check.

usage (lock GPU clock first: sudo nvidia-smi -lgc <freq>):
    python3 bench_sched_sim.py --probe                        # closed-loop drain rate -> lambda_sat
    python3 bench_sched_sim.py --rates 3,5,7 --deadline 15    # the sweep
    python3 bench_sched_sim.py --rates 7 --dsweep 5,15,40     # aging-step sweep at one rate
    python3 bench_sched_sim.py --compare-only                 # rebuild tables from the raw json
"""
import os
import json
import time
import argparse
from random import randint, seed, shuffle, Random

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
N_REQ = 500
BUCKETS = [(250, (200, 800)),        # 50%  short chat turns
           (175, (1000, 3000)),      # 35%  doc / code context
           (75, (4000, 6000))]       # 15%  long doc
FRACS = (0.60, 0.30, 0.10)           # P0 batch, P1 standard, P2 premium
TIER_NAME = {0: "P0-batch", 1: "P1-std", 2: "P2-prem"}
MAX_TOKENS = 512                     # ignore_eos -> every request emits exactly this many
MAX_NUM_SEQS = 64
CHUNK = 2048                         # max_num_batched_tokens (chunked-prefill chunk)
MAX_MODEL_LEN = 8192


def gen_prompts(n=N_REQ):
    """Fixed prompt set: token ids are random, lengths follow BUCKETS, order shuffled."""
    seed(0)
    prompts = []
    scale = n / sum(c for c, _ in BUCKETS)
    for count, (lo, hi) in BUCKETS:
        prompts += [[randint(0, 10000) for _ in range(randint(lo, hi))] for _ in range(round(count * scale))]
    shuffle(prompts)
    return prompts[:n]


def gen_class(n, fracs=FRACS, seed_val=2):
    """Assign each request a tier in {0,1,2} by cumulative fracs. Fixed seed -> shared across runs."""
    rng = Random(seed_val)
    c0, c1 = fracs[0], fracs[0] + fracs[1]
    return [0 if (r := rng.random()) < c0 else 1 if r < c1 else 2 for _ in range(n)]


def gen_arrivals(n, rate, seed_val=1):
    """Poisson arrivals: cumulative Exp(rate) inter-arrival offsets from t0. rate <= 0 -> all at 0."""
    if rate <= 0:
        return [0.0] * n
    rng = Random(seed_val)
    t, arr = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        arr.append(t)
    return arr


def _sp_for(mode, cls):
    """baseline -> every request priority 0 (FCFS); priority -> per-tier priority."""
    from nanovllm import SamplingParams
    return [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=MAX_TOKENS,
                           priority=(c if mode == "priority" else 0)) for c in cls]


def _run_once(llm, prompts, sp, arrivals):
    """Replay the arrival trace open-loop, then reduce the engine logs to per-request rows."""
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
        if llm.is_finished():                       # nothing to run: sleep to the next arrival
            if idx < n:
                time.sleep(max(0.0, arrivals[idx] - (time.perf_counter() - t0)))
            continue
        llm.step()
    wall = time.perf_counter() - t0

    ids = sorted(llm.token_times.keys())            # = submission order = prompt index
    assert len(ids) == n, f"{len(ids)} requests recorded, expected {n}"
    reqs = []
    for i, sid in enumerate(ids):
        tt, r = llm.token_times[sid], llm.req_log[sid]
        itls = [b - a for a, b in zip(tt, tt[1:])]
        reqs.append({
            "req_id": i,
            "arrival_ts": arrivals[i],                          # open-loop, from the trace
            "inject_ts": llm.submit_time[sid] - t0,             # when the driver actually enqueued
            "sched_ts": r["first_scheduled_time"] - t0,         # first admission = prefill start
            "first_token_ts": tt[0] - t0,
            "last_token_ts": tt[-1] - t0,
            "input_len": r["num_prompt_tokens"],
            "output_len": r["num_completion_tokens"],
            "max_itl": max(itls) if itls else 0.0,
            "num_preemptions": r["num_preemptions"],
        })
    _check(reqs)
    prefill_tok = sum(it["n_prefill_tokens"] for it in llm.iter_log)
    want = sum(r["input_len"] for r in reqs)   # >= (not ==): a preempted seq re-prefills
    assert prefill_tok >= 0.95 * want, f"prefilled {prefill_tok} of {want} prompt tokens: stale prefix cache?"
    iters = [{**it, "t": it["t"] - t0} for it in llm.iter_log]
    return {"wall": wall, "reqs": reqs, "iters": iters,
            "blocked": llm.scheduler.total_blocked_admissions}


def _check(reqs):
    """Ordering / completeness invariants — a violation means the instrumentation is wrong."""
    for r in reqs:
        assert r["arrival_ts"] <= r["inject_ts"] <= r["sched_ts"], r
        assert r["sched_ts"] <= r["first_token_ts"] <= r["last_token_ts"], r
        assert r["output_len"] == MAX_TOKENS, r


def _build_llm(mixbatch):
    from nanovllm import LLM
    llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=MAX_MODEL_LEN,
              max_num_batched_tokens=CHUNK, max_num_seqs=MAX_NUM_SEQS, enable_mixbatch=mixbatch)
    nb = llm.model_runner.config.num_kvcache_blocks
    print(f"KV cache: {nb} blocks x {llm.scheduler.block_size} = {nb * llm.scheduler.block_size} tokens",
          flush=True)
    return llm


def _worker(runs, prompts, cls, reps, mixbatch, out_path):
    """One process, model loaded once; each run mutates priority_deadline + the SamplingParams list."""
    llm = _build_llm(mixbatch)
    llm.collect_metrics = False
    llm.generate(prompts[:32], _sp_for("baseline", cls[:32]), use_tqdm=False)   # warm-up (discarded)

    results = []
    for run in runs:
        llm.scheduler.priority_deadline = run["deadline"]
        sp = _sp_for(run["mode"], cls)
        arrivals = gen_arrivals(len(prompts), run["rate"])
        reps_data = []
        for r in range(reps):
            llm.scheduler.block_manager.reset_cache()   # cold prefix cache: same prefill work every arm
            rep = _run_once(llm, prompts, sp, arrivals)
            print(f"  [{run['tag']}] rep {r}: wall {rep['wall']:.1f}s  iters {len(rep['iters'])}",
                  flush=True)
            reps_data.append(rep)
        results.append({**run, "reps": reps_data})
        json.dump({"max_tokens": MAX_TOKENS, "cls": cls, "mixbatch": mixbatch, "runs": results},
                  open(out_path, "w"))            # checkpoint after every run
    print(f"saved: {out_path}", flush=True)


def probe(mixbatch):
    """Closed-loop drain rate: all requests at t=0, one generate call -> lambda_sat reference."""
    prompts, cls = gen_prompts(), gen_class(N_REQ)
    llm = _build_llm(mixbatch)
    llm.generate(prompts[:32], _sp_for("baseline", cls[:32]), use_tqdm=False)
    llm.scheduler.block_manager.reset_cache()
    t = time.perf_counter()
    llm.generate(prompts, _sp_for("baseline", cls), use_tqdm=False)
    wall = time.perf_counter() - t
    print(f"PROBE n={len(prompts)} wall={wall:.1f}s  drain_rate={len(prompts)/wall:.2f} req/s  "
          f"out_tok/s={len(prompts)*MAX_TOKENS/wall:.0f}", flush=True)


# ---------------------------------------------------------------- analysis

def _pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _latency(run, cls, want):
    """Per-tier latency metrics in ms, pooled over reps. want=None -> all tiers."""
    q, ttft, tpot, mitl, e2e, lag, npre = [], [], [], [], [], [], []
    for rep in run["reps"]:
        for r in rep["reqs"]:
            if want is not None and cls[r["req_id"]] != want:
                continue
            a = r["arrival_ts"]
            q.append((r["sched_ts"] - a) * 1000)
            ttft.append((r["first_token_ts"] - a) * 1000)
            tpot.append((r["last_token_ts"] - r["first_token_ts"]) / (r["output_len"] - 1) * 1000)
            mitl.append(r["max_itl"] * 1000)
            e2e.append((r["last_token_ts"] - a) * 1000)
            lag.append((r["inject_ts"] - a) * 1000)
            npre.append(r["num_preemptions"])
    return {
        "n": len(ttft),
        "queue_p95": _pct(q, 95), "queue_max": max(q) if q else 0.0,
        "ttft_p50": _pct(ttft, 50), "ttft_p95": _pct(ttft, 95), "ttft_p99": _pct(ttft, 99),
        "tpot_p50": _pct(tpot, 50), "tpot_p95": _pct(tpot, 95), "tpot_p99": _pct(tpot, 99),
        "max_itl_p99": _pct(mitl, 99),
        "e2e_p95": _pct(e2e, 95), "e2e_p99": _pct(e2e, 99),
        "lag_p50": _pct(lag, 50), "lag_p99": _pct(lag, 99),
        "preempted_pct": 100.0 * sum(1 for x in npre if x) / max(len(npre), 1),
    }


def _internals(run):
    """Per-iteration scheduler internals + throughput, pooled over reps."""
    run_, wait, bseq, btok, kv, dts = [], [], [], [], [], []
    npre = ntok = niter = nmix = npf = 0
    wall = 0.0
    slopes = []
    for rep in run["reps"]:
        its = rep["iters"]
        wall += rep["wall"]
        ntok += sum(r["output_len"] for r in rep["reqs"])
        npre += its[-1]["preempt_cum"] if its else 0
        niter += len(its)
        npf += sum(1 for it in its if it["n_prefill_seqs"])
        nmix += sum(1 for it in its if it["n_prefill_seqs"] and it["n_decode_seqs"])
        for it in its:
            run_.append(it["running"])
            wait.append(it["waiting"])
            bseq.append(it["n_prefill_seqs"] + it["n_decode_seqs"])
            btok.append(it["n_prefill_tokens"] + it["n_decode_seqs"])
            kv.append(100.0 * it["kv_used"] / it["kv_total"])
            dts.append(it["dt"] * 1000)
        slopes.append(_queue_slope(its))
    return {
        "throughput": ntok / wall if wall else 0.0,
        "iters": niter,
        "step_ms_p50": _pct(dts, 50), "step_ms_p99": _pct(dts, 99),
        "running_mean": sum(run_) / max(len(run_), 1), "running_p95": _pct(run_, 95),
        "waiting_mean": sum(wait) / max(len(wait), 1), "waiting_p95": _pct(wait, 95),
        "batch_seqs_mean": sum(bseq) / max(len(bseq), 1), "batch_seqs_p95": _pct(bseq, 95),
        # prefill iterations are rare but huge, so the token distribution is long-tailed: p99, not p95
        "batch_tok_mean": sum(btok) / max(len(btok), 1), "batch_tok_p99": _pct(btok, 99),
        # share of prefill iterations that also carried decode work = how often mixbatch actually fires
        "mix_iters_pct": 100.0 * nmix / max(npf, 1),
        "kv_util_mean": sum(kv) / max(len(kv), 1), "kv_util_max": max(kv) if kv else 0.0,
        "preempt_per_1k_iters": 1000.0 * npre / max(niter, 1),
        "blocked": sum(rep.get("blocked", 0) for rep in run["reps"]),
        "queue_slope": sum(slopes) / max(len(slopes), 1),
    }


def _queue_slope(iters):
    """Least-squares slope (waiting seqs per second) over the second half of the run.

    ~0 means the backlog is stationary (the engine keeps up); clearly positive means the offered
    rate exceeds capacity, which is what marks the saturation knee.
    """
    half = iters[len(iters) // 2:]
    if len(half) < 8:
        return 0.0
    n = len(half)
    mx = sum(i["t"] for i in half) / n
    my = sum(i["waiting"] for i in half) / n
    num = sum((i["t"] - mx) * (i["waiting"] - my) for i in half)
    den = sum((i["t"] - mx) ** 2 for i in half)
    return num / den if den else 0.0


def compare(raw_path, out_prefix):
    d = json.load(open(raw_path))
    cls = d["cls"]
    rows, internals = [], []
    for run in d["runs"]:
        internals.append({"rate": run["rate"], "scheduler": run["tag"], "deadline": run["deadline"],
                          **_internals(run)})
        for tier in (2, 1, 0, None):
            rows.append({"rate": run["rate"], "scheduler": run["tag"], "deadline": run["deadline"],
                         "tier": TIER_NAME.get(tier, "ALL"), **_latency(run, cls, tier)})

    with open(f"{out_prefix}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: (round(v, 1) if isinstance(v, float) else v)
                                for k, v in r.items()}) + "\n")

    n_tier = {TIER_NAME[t]: cls.count(t) for t in (2, 1, 0)}
    L = [f"# Scheduler simulation — Qwen3-1.7B (mixbatch={d.get('mixbatch')})", "",
         f"{len(cls)} requests, tiers {n_tier}, output {d['max_tokens']} tok (ignore_eos), "
         f"max_num_seqs={MAX_NUM_SEQS}, chunk={CHUNK}, prompt buckets {BUCKETS}", "",
         "## Latency (ms)", "",
         "| rate | scheduler | D | tier | n | queue p95 | queue max | TTFT p50 | TTFT p95 | TTFT p99 "
         "| TPOT p50 | TPOT p99 | maxITL p99 | E2E p95 | E2E p99 | preempted % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['rate']} | {r['scheduler']} | {r['deadline']} | {r['tier']} | {r['n']} | "
                 f"{r['queue_p95']:.0f} | {r['queue_max']:.0f} | {r['ttft_p50']:.0f} | "
                 f"{r['ttft_p95']:.0f} | {r['ttft_p99']:.0f} | {r['tpot_p50']:.1f} | "
                 f"{r['tpot_p99']:.1f} | {r['max_itl_p99']:.0f} | {r['e2e_p95']:.0f} | "
                 f"{r['e2e_p99']:.0f} | {r['preempted_pct']:.1f} |")

    L += ["", "## Scheduler internals (per iteration)", "",
          "| rate | scheduler | D | out tok/s | iters | prefill it w/ decode % | step p50 ms | running mean/p95 | "
          "waiting mean/p95 | batch seqs mean/p95 | batch tok mean/p99 | KV util mean/max % | "
          "preempt /1k it | blocked | queue slope req/s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in internals:
        L.append(f"| {r['rate']} | {r['scheduler']} | {r['deadline']} | {r['throughput']:.0f} | "
                 f"{r['iters']} | {r['mix_iters_pct']:.1f} | {r['step_ms_p50']:.1f} | "
                 f"{r['running_mean']:.1f}/{r['running_p95']:.0f} | "
                 f"{r['waiting_mean']:.1f}/{r['waiting_p95']:.0f} | "
                 f"{r['batch_seqs_mean']:.1f}/{r['batch_seqs_p95']:.0f} | "
                 f"{r['batch_tok_mean']:.0f}/{r['batch_tok_p99']:.0f} | "
                 f"{r['kv_util_mean']:.0f}/{r['kv_util_max']:.0f} | "
                 f"{r['preempt_per_1k_iters']:.2f} | {r['blocked']} | {r['queue_slope']:+.2f} |")

    # headline: premium gain per rate, batch starvation bound, driver health, saturation knee
    def get(rate, sched, tier, field):
        for r in rows:
            if r["rate"] == rate and r["scheduler"] == sched and r["tier"] == tier:
                return r[field]
        return None

    L += ["", "## Headline", ""]
    rates = sorted({r["rate"] for r in rows})
    for rate in rates:
        b, p = get(rate, "baseline", "P2-prem", "ttft_p99"), get(rate, "priority", "P2-prem", "ttft_p99")
        if b and p:
            L.append(f"- rate {rate}: **P2 premium p99 TTFT {b:.0f} -> {p:.0f} ms "
                     f"({(p/b-1)*100:+.1f}%)** (FCFS -> priority+aging)")
    for rate in rates:
        qa, qn = get(rate, "priority", "P0-batch", "queue_max"), get(rate, "no-aging", "P0-batch", "queue_max")
        if qa is not None and qn is not None:
            L.append(f"- rate {rate}: **P0 batch max queueing delay** aging {qa:.0f} ms vs "
                     f"no-aging {qn:.0f} ms (starvation bound)")
    lag = max((r["lag_p99"] for r in rows if r["tier"] == "ALL"), default=0.0)
    ttft0 = min((r["ttft_p50"] for r in rows if r["tier"] == "ALL"), default=0.0)
    L.append(f"- driver health: injection lag p99 {lag:.0f} ms vs lowest TTFT p50 {ttft0:.0f} ms "
             f"({'OK' if lag < 0.2 * max(ttft0, 1) else 'TOO HIGH — the driver is distorting the trace'})")
    knee = [i for i in internals if i["scheduler"] == "baseline" and i["queue_slope"] < 0.2]
    if knee:
        L.append(f"- saturation knee: backlog stays stationary up to **{max(i['rate'] for i in knee)} "
                 f"req/s** (queue slope < 0.2 req/s on the second half of the run)")

    open(f"{out_prefix}.md", "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nsaved: {out_prefix}.jsonl, {out_prefix}.md")
    _plots(d, cls, out_prefix)


def _plots(d, cls, out_prefix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    top = max(r["rate"] for r in d["runs"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for run in d["runs"]:
        if run["rate"] != top or run["tag"] not in ("baseline", "priority"):
            continue
        for tier in (2, 0):
            xs = sorted((r["first_token_ts"] - r["arrival_ts"]) * 1000
                        for rep in run["reps"] for r in rep["reqs"] if cls[r["req_id"]] == tier)
            ax[0].plot(xs, [i / len(xs) for i in range(len(xs))],
                       label=f"{run['tag']} {TIER_NAME[tier]}")
    ax[0].set_xlabel("TTFT (ms)"), ax[0].set_ylabel("CDF"), ax[0].legend(fontsize=7)
    ax[0].set_title(f"TTFT CDF @ {top} req/s"), ax[0].grid(alpha=.3)

    for run in d["runs"]:
        if run["rate"] == top and run["tag"] == "priority":
            its = run["reps"][0]["iters"]
            ax[1].plot([i["t"] for i in its], [i["waiting"] for i in its], label="waiting")
            ax[1].plot([i["t"] for i in its], [i["running"] for i in its], label="running")
            ax[1].plot([i["t"] for i in its], [100 * i["kv_used"] / i["kv_total"] for i in its],
                       label="KV util %")
    ax[1].set_xlabel("t (s)"), ax[1].legend(fontsize=7), ax[1].grid(alpha=.3)
    ax[1].set_title(f"queue depth + KV utilization @ {top} req/s")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=120)
    print(f"saved: {out_prefix}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=str, default="3,5,7", help="comma-separated arrival rates req/s")
    ap.add_argument("--deadline", type=float, default=15.0, help="aging step D (s); <=0 disables aging")
    ap.add_argument("--dsweep", type=str, default="", help="comma-separated D values to sweep at the top rate")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--n", type=int, default=N_REQ, help="number of requests (smaller = pilot run)")
    ap.add_argument("--mixbatch", type=str, default="on", choices=["on", "off"])
    ap.add_argument("--out", type=str, default="sched_sim")
    ap.add_argument("--probe", action="store_true", help="measure the closed-loop drain rate only")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    if args.compare_only:
        compare(f"{args.out}_raw.json", args.out)
        return

    print("[reminder] lock GPU clock: sudo nvidia-smi -lgc <freq>")
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    mixbatch = args.mixbatch == "on"

    if args.probe:
        p = ctx.Process(target=probe, args=(mixbatch,))
        p.start(); p.join()
        assert p.exitcode == 0, "probe failed"
        return

    prompts, cls = gen_prompts(args.n), gen_class(args.n)
    rates = [float(x) for x in args.rates.split(",")]
    top = max(rates)

    runs = []
    if args.dsweep:
        for dv in [float(x) for x in args.dsweep.split(",")]:
            runs.append(dict(tag=f"prio-D{dv:g}", mode="priority", rate=top, deadline=dv))
    else:
        for rate in rates:
            runs.append(dict(tag="baseline", mode="baseline", rate=rate, deadline=args.deadline))
            runs.append(dict(tag="priority", mode="priority", rate=rate, deadline=args.deadline))
        runs.append(dict(tag="no-aging", mode="priority", rate=top, deadline=0.0))

    n_tier = {TIER_NAME[t]: cls.count(t) for t in (2, 1, 0)}
    avg_len = sum(len(p) for p in prompts) / len(prompts)
    print(f"requests={len(prompts)} {n_tier} avg_prompt={avg_len:.0f} tok, rates={rates}, "
          f"D={args.deadline}, mixbatch={mixbatch}, runs={len(runs)}", flush=True)

    raw = f"{args.out}_raw.json"
    p = ctx.Process(target=_worker, args=(runs, prompts, cls, args.reps, mixbatch, raw))
    p.start(); p.join()
    assert p.exitcode == 0, "worker failed"
    compare(raw, args.out)


if __name__ == "__main__":
    main()
