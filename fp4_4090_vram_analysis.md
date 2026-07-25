# Qwen3-32B NVFP4 on RTX 4090 — VRAM breakdown, max-context, and allocation-permutation study

Measured 2026-07-24 on the 4090 (23.52 GiB usable, sm_89, torch 2.9.1/cu126, triton 3.5.1).
Model: `Qwen3-32B_nvfp4` (AngelSlim packed W4A4, weight-only dequant) + fp4 paged KV + optional EAGLE-3.
Weight-scale stored fp8_e4m3 (`weight_scale_fp8=True`). All numbers below are **measured**, not estimated,
unless marked "theory". Scripts: `probe_vram.py`, `bench_ctx.py`, `bench_nvfp4_gemm.py`.

## TL;DR
> **Update (§7, 2026-07-25):** the prefill bottleneck below (item 4) is now FIXED — a tiled paged-read
> kernel + fp8 KV scale cut prefill TTFT **19–100×** (2048-tok prefill 184 s → 2.6 s). Items 1–6 are the
> original round-1 study; §7 has the optimized kernel + updated permutation table.

1. **Weights = 19.26 GB** (target) / **20.83 GB** (with EAGLE-3 head). Embed and lm_head are **NOT tied**
   (two bf16 copies = 2.90 GB); the earlier "1.5 GB tied embed" note was wrong.
2. **no_spec max context ≈ 21k tokens** (util 0.94, 83 KV blocks); safe **16k** at util 0.92.
3. **eagle3 barely fits**: it runs **eager** (no CUDA graph), so higher util = bigger KV = less runtime
   headroom → OOM. Runs only at **util ≤ 0.93** with real prefills; useful context ~**3k tokens**.
4. **Prefill, not VRAM, is the practical limit.** A 2048-token prefill takes **184 s**. The bottleneck is
   the **fp4-KV dequant-in-loop paged-attention read kernel** (O(N²) prefill, O(N)/step decode), NOT the
   weight GEMM (decode GEMM is at fp16 parity; prefill GEMM only 3–5× and is <5 % of prefill wall-time).
5. **EAGLE-3 accept_len = 2.52** (k=2, healthy) but end-to-end **~1.05× speedup only**, because eager
   eagle3 (~210 ms/forward) cancels the acceptance gain vs the CUDA-graphed no_spec baseline (~100 ms).

---

## 1. Per-component VRAM breakdown (measured)

| Component | Measured GB | Theory | Notes |
|---|---|---|---|
| Packed weights (uint8) | **14.53** | 0.5 B/param × 31.2 B = 14.53 | 64 layers of linears, 2 fp4 codes/byte |
| Weight scale (fp8_e4m3) | **1.82** | N·K/16 = 1.95e9 B | per-16-element group |
| Weight scale_2 (fp32) | 0.02 | 4·N per linear | global per-output scale |
| Embed tokens (bf16) | **1.45** | 151936·5120·2 | **not tied** |
| lm_head (bf16) | **1.45** | 151936·5120·2 | **separate copy** (tie=False) |
| Norms (bf16) | 0.001 | — | RMSNorm |
| **Target weights total** | **19.26** | | `current(weights)` from engine |
| EAGLE-3 head (extra) | **+1.57** | fc+midlayer+draft lm_head 0.31 | embed shared w/ target (not double-counted) |
| **Weights w/ EAGLE-3** | **20.83** | | |
| CUDA context (non-torch) | ~0.66 | — | `used − current` |
| fp4 KV per block | **0.0246 (24 MiB)** | 2·64·256·8·(64·1 + 8·4) | 256 tokens/block |
| → fp4 KV per **token** | **96 KB** | 64 KB data + **32 KB fp32 scale** | ⅓ of KV is the fp32 scale |
| Draft KV per block | 0.00024 | 1 layer, head_dim 80 | negligible |
| Warmup activation spike | mnbt256: **0.06** / mnbt2048: **0.46** | grows with mnbt | `peak − current`; bounds runtime activation (chunked prefill) |

**KV budget formula** (verified exactly): `blocks = ⌊(total·util − weights − ctx − spike) / block_bytes⌋`.
Fixed pre-KV overhead: **20.16 GB** (no_spec) / **21.55 GB** (eagle3). At 23.52 GB usable this leaves only
1–3 GB for KV + runtime — the whole problem in one line.

---

## 2. Allocation-permutation sweep

### no_spec (CUDA-graphed; memory reserved at init)
| util | KV blocks | max ctx (tok) | post-init free | runs? |
|---|---|---|---|---|
| 0.90 | 43 | 11,008 | 0.98 GB | ✓ |
| 0.92 | 63 | 16,128 | 0.51 GB | ✓ **(safe pick)** |
| 0.94 | 83 | 21,248 | **0.043 GB** | ✓ (max, fragile) |
| 0.95+ | (95+) | — | <0 | ✗ graph-capture OOM (predicted) |

### eagle3 (forced eager; runtime reservation collides with KV)
| util | KV blocks | ctx (tok) | post-init free | tiny gen | **512 prefill** |
|---|---|---|---|---|---|
| 0.90 | −17 | — | — | ✗ budget<0 (assert) | — |
| 0.92 | 3 | 768 | 1.95 GB | ✓ | ✓ |
| **0.93** | **13** | **3,328** | 1.71 GB | ✓ | ✓ **(best usable)** |
| 0.94 | 23 | 5,888 | 1.48 GB | ✓ | ✗ **OOM** |
| 0.95 | 33–47 | — | ~1.2 GB | ✗ OOM | ✗ OOM |

Key: eagle3's runnable ceiling is set by **runtime free headroom (~1.5–1.7 GB needed for a real prefill)**,
not the KV budget. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` did **not** rescue util 0.95.
The vise: fit needs util > 0.916, but every 0.01 util above that steals ~0.47 GB from the eager runtime.

---

## 3. Max-context + prefill/decode timing

### no_spec (util 0.92)
| prompt len | TTFT (prefill) | prefill tok/s | decode tok/s | ITL p50 |
|---|---|---|---|---|
| 256 (1 chunk, 0 prior ctx) | 1.4 s | 179 | 9.94 | 100 ms |
| 512 | 11.0 s | 46 | 8.88 | 113 ms |
| 1024 | 37.0 s | 28 | 7.29 | 137 ms |
| 2048 | **184 s** | 11 | 5.36 | 187 ms |

Prefill is **super-linear** (~O(N²)); decode ITL climbs with context (100→187 ms). Extrapolated: 8k ≈ 12 min,
15k ≈ 23 min of prefill. **Effective usable context is prefill-bound to ~1–2k**, despite 16–21k of KV capacity.

### eagle3 (util 0.93, plen 512)
| metric | value | vs no_spec@512 |
|---|---|---|
| accept_len (k=2) | **2.52** | — (healthy) |
| decode tok/s | 9.3 | 8.88 → **1.05×** |
| ITL p50 / p99 | 210 / 1348 ms | bimodal (burst then verify gap) |
| prefill tok/s | 37 | 46 (aux extraction overhead) |

### NVFP4 GEMM microbench (isolated)
| M | nvfp4 vs fp16 | eff. BW |
|---|---|---|
| 1 (decode) | **~1.0× (parity)** | 250–270 GB/s |
| 512 (prefill) | 3.2–5.2× slower | 14–23 GB/s |

Weight GEMM is fine for decode and only 3–5× for prefill (~0.5–1 s across 64 layers/chunk) — so it accounts
for <5 % of the 23 s/chunk prefill. **The remaining ~95 % is the fp4-KV paged attention.**

---

## 4. Root-cause analysis

- **fp4-KV dequant-in-loop paged read kernel is the dominant cost** at any real context, in BOTH prefill
  (O(q·context), super-linear as chunks accumulate) and decode (O(context)/step; ITL 100→187 ms). Confirmed
  by: 256-token prefill with *zero* prior KV = 1.4 s (mostly GEMM), but adding prior fp4 KV explodes cost.
- **eager eagle3 kills the speedup.** accept_len 2.52 should give ~2× fewer forwards, but each eagle3 forward
  is eager (~210 ms) vs graphed no_spec (~100 ms), so net ≈ parity. The tuple aux-return blocks graph capture.
- **Memory vise on eagle3** comes from three fixable choices: untied lm_head (1.45 GB), bf16 head (1.57 GB),
  and fp32 KV scale (⅓ of KV). Together they leave no room for both KV and eager runtime.

---

## 5. Reference comparison
- **AdaLLM (NVFP4-on-4090)**: ceiling Gemma3-27B @ 19.83 GB, ~13.6 tok/s decode (short ctx). Our no_spec
  decode at short ctx (9.9 tok/s) scales to ~parity for a larger 32B; no dense-32B proof point exists there.
  AdaLLM's `NVFP4_FP8` (W4A8) uses fp8 tensor cores for prefill — the lever we haven't pulled.
- **vLLM on H100 + fp8**: EAGLE-3 ~1.5× and far higher absolute tok/s; the 4090 gap is (a) ~3.3× HBM
  bandwidth, (b) fp8 doesn't fit 24 GB, (c) no fp8-TC prefill here, (d) eager eagle3.

---

## 6. Ranked squeeze levers → next memory-allocation permutation

| # | Lever | Frees / gains | Cost | Why |
|---|---|---|---|---|
| 1 | **fp4 KV scale fp32 → fp8_e4m3** | KV 96→72 KB/tok (−25%); +blocks; **faster paged read** (less scale gather) | kernel + pool change (`paged_attn.py`, `model_runner.py`) | double win: more context AND attacks the #1 perf bottleneck |
| 2 | **Optimize fp4-KV paged read kernel** | prefill & decode-at-ctx speed | kernel work (apply arithmetic E2M1 decode like the GEMV path; batch scale loads) | this is 95% of prefill and the decode-ctx slope |
| 3 | **Quantize eagle3 head to nvfp4** + **tie/quantize lm_head** | ~1.45 + ~1.0 GB → ~2.5 GB | loader + head build | gives eagle3 real context + graph headroom |
| 4 | **Make eagle3 capturable (CUDA graph)** | ~2× per forward → realizes the accept_len 2.52 as a real speedup | refactor aux tuple return | biggest spec-decode win |
| 5 | **fp8 tensor-core prefill (W4A8)** | prefill 3–5× → ~1× | AdaLLM-style kernel | fixes the weight-GEMM part of prefill |

### Recommended configs to adopt now (knobs only, no code)
- **no_spec**: `util 0.92, mnbt 256, mns 8, enforce_eager=False` → 16k KV capacity, safe. Use 0.94 only if you
  accept a 43 MB margin. Keep prompts ≤ ~2k for tolerable TTFT.
- **eagle3**: `util 0.93, mnbt 256, mns 1` (forced eager) → ~3k ctx, accept_len 2.52, but only ~parity speed.
  **Not worth enabling** until levers #3 + #4 land. Today no_spec is the better single-4090 config.

### Verdict
Qwen3-32B fp4 **works** on one 4090 (10 tok/s decode at short ctx, 19.3 GB weights). But usable context is
prefill-bound to ~1–2k, and EAGLE-3 is currently not worth its memory. The highest-leverage next step is the
**fp8 KV scale + paged-read kernel optimization** (levers #1/#2), which improves both context and the real
bottleneck; freeing head/embed memory (#3/#4) is what would finally make speculation pay off.

---

## 7. Kernel optimization round 2 (2026-07-25) — fp8 KV scale + tiled paged read

**Profiling (nsys) settled the root cause.** In a *single-chunk* prefill the nvfp4 weight GEMM is 96.8%;
in the *chunked* prefill (mnbt=256, the real config) the fp4 paged read kernel is **95.8% (46 s / 1024 tok,
241 ms per layer-launch)**. Not the weight GEMM (decode GEMM is fp16-parity; prefill GEMM 3–5×, <5%).
Three defects vs the weight path: (1) LUT gather not arithmetic decode; (2) grid `(Nq,H)` → each query row
re-loads+re-dequantizes the same KV (O(N²)); (3) no split-KV at decode. (ncu blocked by counter perms.)

### Changes
1. **fp8 KV scale** (`kv_scale_fp8`): fp4 per-16 scale fp32→fp8_e4m3. **KV 25.17→18.87 MB/block (−25%)**,
   **+33% blocks** at a given util. Store/read kernels bitcast via a uint8 view (sm_89). fp32 path unchanged
   (byte-exact regression check passed).
2. **Query-tiled prefill kernel** (`_paged_attn_tiled_kernel`): `BLOCK_M=32` query tile, `BLOCK_N=64`, arithmetic
   E2M1 decode, `tl.dot` — each KV tile loaded+dequantized once, reused across the tile. Correct vs SDPA
   (max 4e-3) and vs the per-row kernel (fp4/fp8: mean 9e-6).
3. **Split-KV decode** (`_paged_attn_splitkv_kernel`, flash-decoding): bit-exact vs per-row, but **not a win at
   bs=1** (64 heads already ~half-fill the SMs; merge + under-used splits cost more). Default off
   (`NANO_DECODE_SPLIT=1`), kept for batched/long-ctx use.

### Prefill TTFT (no_spec, util 0.90, fp8 scale) — before → after
| prompt | baseline TTFT | optimized TTFT | speedup |
|---|---|---|---|
| 512  | 11,040 ms | 589 ms  | **19×** |
| 1024 | 37,035 ms | 1,223 ms | **30×** |
| 2048 | 184,322 ms | 2,611 ms | **71×** |
| 4096 | (~740 s extrap.) | 7,533 ms | **~100×** |

Prefill throughput ~800 tok/s, now ~O(N). Decode (per-row) 7.2 / 5.6 / 3.9 / 2.4 tok/s at 512/1k/2k/4k ctx
(memory-bound, grows with context); arithmetic decode was ~parity-to-slightly-slower than the LUT for the
per-row path (cross-session; decode was never the bottleneck). Coherent on real prompts under CUDA graph.

### Permutation table (no_spec, mnbt=256, graph) — blocks / max context
| util | fp32 scale | fp8 scale |
|---|---|---|
| 0.90 | 43 / 11.0k tok | **57 / 14.6k tok** (measured, graph OK) |
| 0.92 | 63 / 16.1k tok | 84 / 21.5k tok (graph headroom tight) |
| 0.94 | 83 / 21.2k tok | 110 / 28.2k tok (enforce_eager to fit graph) |

fp8 scale gives +33% context at every util. Graph-capture headroom shrinks with util (KV pool grows);
util 0.90 is the safe graphed point; push higher with `enforce_eager=True` (loses the decode graph).

### Recommended no_spec config (optimized)
`weight_quant=nvfp4, weight_scale_fp8=True, kv_quant_target=fp4, kv_scale_fp8=True, util=0.90,
max_num_batched_tokens=256, enforce_eager=False` → 14.6k ctx, ~800 tok/s prefill, coherent.
