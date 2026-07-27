# Prefill NVFP4 GEMM optimization — W4A16, Qwen3-32B (4090)

Kernel-only speedup (no precision change, no spec). Output is bit-identical to the old kernel.

## What was wrong (from the profile)

Legacy `[N,K/2]` weight layout is uncoalesced on the `BLOCK_N` (stride `K/2`) axis. Production
already relayouts to `[K/2,N]` (N-contiguous) at load — but the GEMM launch config was tuned for the
old 128×128/8-warp tile, giving **1 block/SM = 2 active warps/scheduler**, so the dequant math-pipe
latency couldn't be hidden (68–76% of scheduler cycles had no eligible warp).

## Fix

- Confirmed the relaid `[K/2,N]` layout is the production path (coalesced weight load).
- Retuned the prefill launch config (M>64): **128×128 / 8 warps / 4 stages → 128×64 / 4 warps / 3
  stages**. Smaller `BLOCK_N` + fewer warps → ~2 blocks/SM. `BLOCK_K` unchanged (64) so the fp32
  K-reduction order is identical → **output bit-identical** (verified `max|diff|=0.0`).
- Added `NVFP4_BM/BN/BK/WARPS/STAGES` env overrides for future tuning.

## Results (M=512 prefill, vs cublas fp16)

| GEMM | before (TFLOPS) | after (TFLOPS) | per-GEMM speedup |
|---|---|---|---|
| qkv_proj  | 68  | 116 | 1.70× |
| o_proj    | 54  | 87  | 1.60× |
| gate_up   | 77  | 114 | 1.48× |
| down_proj | 53  | 86  | 1.62× |

ncu (gate_up), before → after tuning:

| metric | before | after |
|---|---|---|
| Compute (SM) throughput | 32.6% | **51.8%** |
| Memory throughput | 19.4% | 43.8% |
| active warps / scheduler | 2.0 | **2.9** |
| scheduler cycles, no eligible warp | 68% | **52%** |
| duration | 3.90 ms | 2.44 ms |

**End-to-end prefill TTFT on the full 32B (N=512): 380.7 ms → 313.6 ms = 1.21×.** (End-to-end < per-GEMM
because TTFT also covers attention, KV store, norms, embeddings.)

## Notes

- ALU bit-op dequant was the right call: ALU is not the bottleneck (~28% util); a LUT gather would add
  L1TEX pressure, which is what was already choking the kernel. Dequant is already fused into the K-loop.
- Remaining opportunity: the kernel loads each packed byte **twice** (once per nibble). A half-load
  rewrite (load `BLOCK_K/2` bytes, decode both nibbles, two `tl.dot`s) would cut weight-load traffic;
  after tuning, compute/memory are balanced (52%/44%) so the expected further gain is moderate.
- W4A8 is still not indicated for prefill — SM is 52% and the kernel is not MMA-bound; fp8 tensor cores
  don't address occupancy/dequant. See prefill_profile_results.md.
