"""Deadline (aging-step) permutation at a fixed overload rate, to show the starvation<->latency knob.
Reuses bench_priority's worker; varies config.priority_deadline across runs on one loaded model."""
import json
from bench_priority import gen_prompts, gen_class, _worker, _pct, _tier_metrics, _throughput, TIER_NAME, FRACS, BUCKETS

RATE = 24.0   # ~2x saturation -> deep backlog


def build_table(raw_path):
    d = json.load(open(raw_path))
    cls = d["cls"]
    rows = []
    for run in d["runs"]:
        for tier in (2, 1, 0):
            ttft, itl = _tier_metrics(run, cls, tier)
            rows.append({"tag": run["tag"], "deadline": run["deadline"], "tier": TIER_NAME[tier],
                         "ttft_p50": round(_pct(ttft, 50), 1), "ttft_p99": round(_pct(ttft, 99), 1),
                         "ttft_max": round(max(ttft), 1) if ttft else 0.0,
                         "throughput": round(_throughput(run), 1)})
    lines = [f"# Deadline (aging-step) permutation at rate={RATE} req/s (~2x saturation), Qwen3-1.7B",
             "", f"tiers: P2 {FRACS[2]:.0%} / P1 {FRACS[1]:.0%} / P0 {FRACS[0]:.0%}; prompts={sum(c for c,_ in BUCKETS)}",
             "", "| aging D (s) | tier | TTFT p50 | TTFT p99 | TTFT max | thrpt |",
             "|-------------|------|----------|----------|----------|-------|"]
    for r in rows:
        d_lbl = "off (FCFS)" if r["tag"] == "baseline" else ("off" if r["deadline"] <= 0 else f"{r['deadline']:g}")
        lines.append(f"| {d_lbl} | {r['tier']} | {r['ttft_p50']} | {r['ttft_p99']} | {r['ttft_max']} | {r['throughput']} |")
    open("prio_deadline.md", "w").write("\n".join(lines))
    print("\n".join(lines))
    print("\nsaved: prio_deadline.md")


def main():
    prompts = gen_prompts()
    cls = gen_class(len(prompts))
    runs = [
        dict(tag="baseline", mode="baseline", rate=RATE, deadline=4.0),   # FCFS reference
        dict(tag="prio-D4",  mode="priority", rate=RATE, deadline=4.0),    # aging too aggressive
        dict(tag="prio-D15", mode="priority", rate=RATE, deadline=15.0),
        dict(tag="prio-D40", mode="priority", rate=RATE, deadline=40.0),
        dict(tag="no-aging", mode="priority", rate=RATE, deadline=0.0),    # max priority, P0 starves
    ]
    _worker(runs, prompts, cls, reps=3, chunk=2048, out_path="prio_deadline_raw.json")
    build_table("prio_deadline_raw.json")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    p = mp.get_context("spawn").Process(target=main)
    p.start(); p.join()
    assert p.exitcode == 0
