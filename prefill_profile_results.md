# Prefill NVFP4 GEMM profile — dequant-bound, not MMA-bound (4090, Qwen3-32B)

Decides whether mixed precision (W4A8 fp8-tensor-core prefill) can help. It cannot.

## Roofline (bench_nvfp4_gemm.py, M=512 prefill, vs cublas fp16)

| shape | nvfp4 | cublas fp16 | ratio |
|---|---|---|---|
| qkv_proj (5120→10240) | 1.285 ms | 0.403 ms | 3.2× |
| o_proj (8192→5120) | 1.325 ms | 0.326 ms | 4.1× |
| gate_up (5120→51200) | 6.937 ms | 1.871 ms | 3.7× |
| down_proj (25600→5120) | 5.096 ms | 1.007 ms | 5.1× |

cublas fp16 reaches ~135–145 TFLOPS (~85% of the 4090 fp16-tensor peak): the matmul *itself* is
MMA-bound. Our dequant kernel runs 3–5× slower, so the gap is not the matmul.

## ncu (gate_up, M=512, _nvfp4_gemm_kernel)

| metric | value | reading |
|---|---|---|
| Compute (SM) throughput | 26.8% | nothing saturated |
| Memory throughput | 15.8% | not memory-bound |
| DRAM throughput | 8.3% | weights fit in L2 at prefill |
| Highest-utilized pipe | **ALU 27.9%** | integer/logic = 4-bit unpack + E2M1 decode |
| Tensor (HMMA) pipe | not the bottleneck | tensor cores ~73% idle |
| Executed IPC | 0.94 / 4 | latency-bound |
| Active warps / scheduler | **2.0 / 12** | low occupancy |
| Scheduler cycles with no eligible warp | **76.4%** | starved |
| Warp cycles per issued inst | 8.46 | heavy stalling |
| Top stall (40%) | scoreboard dep. on L1TEX load | waiting on weight/scale loads feeding dequant |

## Verdict

The prefill kernel is **dequant / latency / occupancy bound**, not MMA-bound. Tensor cores sit idle
~73% of the time waiting on the dependency chain: global load (packed byte + scale) → integer unpack →
E2M1 decode → scale multiply → `tl.dot`. Only 2 warps/scheduler means there is nothing to hide that
latency behind.

**W4A8 (fp8 tensor cores) would not help prefill, and would likely hurt:**
1. MMA is not the bottleneck — halving MMA time does nothing for a kernel starved on dequant/loads.
2. A real W4A8 path still unpacks fp4 weights (same ALU chain) **and adds** per-token activation fp8
   quant — more ALU work on an already ALU/latency-bound kernel.
3. Amortization is per-tile (fixed by BLOCK_M), so larger prefill batches do not shift it toward MMA.

The lever is the dequant kernel itself, not precision:
- raise occupancy (more warps / more pipeline stages) to hide the load→decode→dot chain;
- improve the packed-weight + scale load pattern (vectorized/coalesced loads, stage through shared mem);
- cheapen the E2M1 decode dependency chain.

Precision is separately a non-issue: W4A16→W4A8 activation quant is ~lossless (99.4% top-1 agreement,
KL 1.3e-3, 10/12 prompts byte-identical greedy) — see precision_a16_vs_a8.md.
