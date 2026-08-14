# Scheduler simulation — mixed batching + 3-tier priority (Qwen3-1.7B, RTX 4090)

## Setup

| | |
|---|---|
| engine | nano-vllm, branch `priority-3tier` (mixbatch merged in) |
| model | Qwen3-1.7B bf16, cuda-graph decode, flash-attn |
| GPU | RTX 4090 24 GB, KV cache **631 blocks x 256 = 161,536 tokens** |
| engine config | `max_num_seqs=64`, `max_num_batched_tokens=2048`, `max_model_len=8192`, `enable_mixbatch=True` |
| workload | 500 requests; prompts 50% 200-800 tok / 35% 1000-3000 / 15% 4000-6000 (avg 1730); output fixed 512 tok (`ignore_eos`) |
| tiers | P2 premium 10% (55) / P1 standard 30% (150) / P0 batch 60% (295) |
| arrivals | open-loop Poisson, precomputed trace replayed identically by every arm |
| saturation | closed-loop drain rate **lambda_sat = 3.97 req/s** (2034 output tok/s) |

Every latency is measured from the **nominal trace arrival**, not from the moment the driver
injected the request. Injection lag (p99 69 ms, about one engine step) is reported separately; it is
only a non-negligible fraction of TTFT at 2.0-4.0 req/s, where TTFT itself is 60-120 ms.

Each rate runs two arms on the identical request set and arrival trace: `baseline` (all requests
priority 0 = pure FCFS) and `priority` (per-tier priority + wait-time aging). The prefix cache is
dropped before every run, so all arms prefill exactly the same token count.

## Premium tier (P2) — FCFS vs priority + aging (D = 15 s)

| rate | x lambda_sat | queue p95 FCFS -> prio | TTFT p99 FCFS -> prio | change |
|---|---|---|---|---|
| 2.0 | 0.5x | 68 -> 65 ms | 211 -> 204 ms | -3% |
| 3.2 | 0.8x | 106 -> 90 ms | 351 -> 214 ms | **-39%** |
| 4.0 | 1.0x | 685 -> 142 ms | 1351 -> 597 ms | **-56%** |
| 4.8 | 1.2x | 13604 -> 449 ms | 14168 -> 965 ms | **-93%** |
| 6.4 | 1.6x | 33830 -> 34195 ms | 37785 -> 39218 ms | +4% (D too small, see below) |

Cost to the batch tier at 1.2x, and throughput:

| rate | tier | queue p95 FCFS -> prio | E2E p99 FCFS -> prio | out tok/s FCFS -> prio |
|---|---|---|---|---|
| 4.8 | P1-std | 13467 -> 10184 ms | 32338 -> 30322 ms | 1994 -> 1993 |
| 4.8 | P0-batch | 14267 -> 18834 ms | 32591 -> 37711 ms | (same run) |

Priority is a **pure reordering**: aggregate output throughput is unchanged (1994 vs 1993 tok/s),
TPOT is unchanged across tiers (35.1 vs 35.9 ms p50), and the gain to P2 is paid for by P0.

## Aging step D at 1.6x saturation (6.4 req/s)

Backlog-induced queueing at this rate is ~40 s, which sets the scale D has to match.

| D (s) | P2 queue p95 | P2 TTFT p99 | P0 queue max | verdict |
|---|---|---|---|---|
| FCFS (no priority) | 33830 | 37785 | 39407 | reference |
| 5 | 35449 | 39705 | 41235 | collapsed to FCFS |
| 15 | 35948 | 39750 | 41633 | collapsed to FCFS |
| **40** | **1007** | **1464** | 45878 | **tiering restored, batch bounded** |
| 80 | 1405 | 1804 | 46068 | ~ aging off |
| 0 (aging off) | 1315 | 2007 | 45772 | max premium gain, batch unbounded under sustained load |

D must be on the order of the backlog drain time. Too small and every request ages to the top tier
within a couple of steps, so the scheduler degenerates to FCFS; too large (or off) and the batch
tier has no bound at all. At D = 40 s, P2 p99 TTFT is **37785 -> 1464 ms (-96%)** while P0 max
queueing delay rises only 39.4 -> 45.9 s (+16%) and stays bounded by `max_priority * D`.

## Scheduler internals (per iteration)

| rate | out tok/s | prefill it w/ decode % | step p50 | running mean/p95 | waiting mean/p95 | batch tok mean/p99 | KV util mean/max | preempt /1k it | queue slope |
|---|---|---|---|---|---|---|---|---|---|
| 2.0 | 1010 | 99.7 | 8.5 ms | 10.1/21 | 0.0/0 | 44/1936 | 13/49% | 0.00 | -0.00 |
| 3.2 | 1591 | 99.7 | 14.0 ms | 27.9/49 | 0.0/0 | 122/2048 | 37/80% | 0.00 | +0.00 |
| 4.0 | 1927 | 99.7 | 21.5 ms | 51.3/75 | 0.4/2 | 217/2048 | 67/100% | 2.51 | +0.01 |
| 4.8 | 1994 | 99.7 | 22.9 ms | 62.9/85 | 19.8/59 | 250/2048 | 81/100% | 16.18 | +0.94 |
| 6.4 | 2034 | 99.7 | 23.1 ms | 66.4/85 | 60.6/147 | 265/2048 | 85/100% | 19.18 | +2.14 |

- **Mixed batching fires on 99.7% of prefill iterations** at every rate: a prefill chunk almost
  never occupies an iteration alone, so decodes are not blocked behind arriving prompts.
- **Saturation knee**: the waiting queue is stationary up to 4.0 req/s and grows from 4.8 req/s
  (slope +0.94 req/s over the arrival window), matching the independently measured
  lambda_sat = 3.97 req/s.
- **Which cap binds**: below the knee the per-iteration seq cap binds (`running` p95 = 75-85 against
  `max_num_seqs=64` per batch) and KV utilization stays under 67%. From 4.0 req/s KV utilization
  pins at 100% and preemption appears (2.5 -> 19.2 per 1000 iterations, 2.4% -> 12% of requests
  preempted), so **KV capacity, not the sequence cap, is the binding constraint under overload**.
- Output throughput saturates at ~2000 tok/s = the measured drain rate; TPOT degrades from 10.4 to
  34.9 ms p50 purely from larger decode batches.

## Reproduce

```
python3 bench_sched_sim.py --probe                            # lambda_sat
python3 bench_sched_sim.py --rates 2,3.2,4,4.8,6.4 --deadline 15
python3 bench_sched_sim.py --rates 6.4 --dsweep 5,15,40,80 --out sched_dsweep
python3 bench_sched_sim.py --compare-only                     # tables + plots from the raw json
```

Outputs: `sched_sim.md` / `sched_sim.jsonl` / `sched_sim_raw.json` / `sched_sim.png`.
