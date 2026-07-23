# Running a ~30B model on a single 4090 — fp4 weights + fp8/fp4 KV + EAGLE-3

## Goal
On one RTX 4090 (24 GB, sm_89), run a ~30B-class model at interactive throughput by stacking three
independent levers:
- **fp4 weight quant (NVFP4)** — makes a ~30B model *fit* (and cuts per-token weight bytes).
- **fp8/fp4 KV quant** — more room (context/batch) + decode bandwidth.
- **EAGLE-3 speculative decoding** — throughput multiplier.

Benchmark metrics: **throughput (tok/s), TTFT, ITL, accept_len**.

## Why it works (mental model)
Single-stream decode is **memory-bandwidth bound**: `tok/s ≈ BW / bytes_per_token`, where
`bytes_per_token ≈ weight_bytes (+ KV)`. So:
- Quantization cuts bytes → fits a bigger model *and* speeds decode.
- Spec decoding amortizes each (now-smaller) weight read over `accept_len` tokens.
- Clean split: **quantization = quality ceiling + what fits; spec decoding = how fast you hit it,
  losslessly relative to the quantized target.** They compose.

Ballpark on 4090 (~1 TB/s): dense 32B fp4 ≈ 30–40 tok/s base → ~45–65 with EAGLE-3.
30B-A3B MoE fp4 (only ~3B active/token) ≈ much faster; likely the sweet spot.

## Hardware / kernel facts (researched, vLLM 0.23.1 source + upstream)
- **flash-attn 2.8.3 (FA2) cannot read fp8/fp4 KV.** FA3 fp8 = **Hopper sm_90 only**; Ada falls back to FA2.
  → real quantized KV on Ada needs a **custom Triton dequant-in-loop attention kernel**
  (port of vLLM `kernel_unified_attention` + `triton_reshape_and_cache_flash`).
- **Ada (sm_89) has fp8 tensor cores, no fp4 TC.** vLLM does fp8 *attention MAC* only via Triton (needs fp8 Q);
  the default FA path dequants. fp8 **W8A8 linear GEMM** runs on Ada via CUTLASS (`cutlass_scaled_mm`, cap≥89).
- **For KV cache specifically:** decode is memory-bound → **storage + dequant-in-loop (bf16 matmul)** is the
  right approach; fp8 MAC buys ~nothing for decode. fp8 MAC only helps compute-bound long-context *prefill* (v2).
- **AdaLLM** (`BenChaliah/NVFP4-on-4090-vLLM`): a nano-vllm-shaped engine with **NVFP4 weights + fp8 KV on Ada**
  (~2.4× VRAM, ~20–25% throughput tax). The one capability vLLM lacks on Ada = **NVFP4 weights**. Strong reference.

## Design decisions
- **EAGLE-3 draft weights: bf16.** The head is 1 layer (~1–1.5 GB for a 32B target) — fp4 saves ~nothing but
  hurts accept rate. Matching the target's fp4 does **not** help accept (alignment comes from the *shared target
  hidden states*, not dtype). No KV-pool overhead (weights aren't in the pool).
- **Draft KV: default quantize (uniform with target)** to keep the single KV pool one dtype; keep bf16 only if the
  accept-rate delta justifies the mixed-pool plumbing. ← this is what Step 0 measures.
- **Target: weights fp4 (NVFP4), KV fp4 (or fp8).**
- **Biggest accept-rate risk:** the EAGLE-3 head was trained for the *bf16* target; an fp4 target shifts its hidden
  states → accept rate drops regardless of draft precision. Mitigation: re-distill the head on fp4-target features.
- **DFlash** (diffusion parallel draft): higher accept length but the draft is **unquantizable bf16** and larger →
  it spends the VRAM fp4 freed. Parked for the *headroom* regime, not the VRAM-maxed 4090 goal.

## Benchmark spec
Configs: (1) **eagle3 + uniform fp4 KV**, (2) **eagle3 + bf16 draft KV** (target fp4 KV), (3) **baseline no-spec**.
All with an fp4 ~30B target, on the 4090. Metrics: throughput, TTFT, ITL, accept_len.

## Metric feasibility / hardware split
- **accept_len is numerically exact via fake-quant on any GPU** — it depends only on the quant *error* injected into
  the K/V (and weights) read back, which a quant→dequant round-trip reproduces exactly. Measurable **now** on the
  local 1.7B eagle3 (3060 Ti). The *relative delta* between configs transfers; the absolute differs from a 30B.
- **throughput / TTFT / ITL need the real kernels + 4090 + 30B.** Fake-quant timings are meaningless (it stores
  bf16, saves no memory/bandwidth, adds overhead).

## Build sequence
0. **[NOW, local]** accept-rate fake-quant sweep on 1.7B eagle3: `{no-quant, fp8 uniform, fp4 uniform,
   fp4 target + bf16 draft}`. Decides uniform-vs-bf16-draft-KV and the mixed-pool go/no-go.  ← *in progress*
1. **Real fp8/fp4 KV path:** fp8/fp4 cache alloc + per-token/block scales, fp8/fp4 store kernel, Triton
   dequant-in-loop attention read kernel (port vLLM). Mixed-pool option. Dev locally (bf16 path verifiable),
   validate quant paths on the 4090.
2. **fp4 weight path:** NVFP4 loader + dequant GEMM (or lift AdaLLM's kernels). Fit the 30B.
3. **[4090]** Full benchmark: 3 configs × 4 metrics on the 30B.
- **Optional shortcut:** run the KV-quant + spec comparison on an 8B that fits **bf16 on the 4090** first, to get
  throughput/TTFT/ITL for the KV-precision and spec-on/off variables *before* the fp4-weight build.

## Assets — all accessible (HF, AngelSlim; same source as the working 1.7B head) [CHECKED]
| component | repo |
|---|---|
| target dense | `Qwen/Qwen3-32B` (64 layers, hidden 5120, head_dim 80) |
| target MoE | `Qwen/Qwen3-30B-A3B` (3B active — faster decode, likely better target) |
| eagle3 head (dense) | `AngelSlim/Qwen3-32B_eagle3` — `LlamaForCausalLMEagle3`, 1 midlayer, draft_vocab 32000; accept 1.86–2.01, ~1.85× (vLLM) |
| eagle3 head (MoE) | `AngelSlim/Qwen3-a3B_eagle3` |
| fp4 weights | `AngelSlim/Qwen3-32B_nvfp4` (compressed-tensors **W4A4**, group16, lm_head/embed excluded, kv_cache_scheme null) |
| int4 (weight-only) | `AngelSlim/Qwen3-32B_int4_gptq` (simpler, fit-first) |
| fp8 weights | `AngelSlim/Qwen3-30B-A3B_fp8_static` |

Wiring gaps (implementation, not blockers):
- Eagle3Proposer hardcoded to 1.7B dims → read the head's config.json (32B: hidden 5120, heads 64, kv 8, head_dim 80, interm 25600).
- **aux layers**: vLLM convention `(2, N//2, N-3)`; nano-vllm uses that **−1** (matched 1.7B accept 1.70≈vLLM 1.71) → 32B (64L) = **`(1, 31, 60)`**; verify accept ≈ vLLM ~1.9.
- 32B head `tie_word_embeddings=false` (1.7B was true) — loads own lm_head; confirm it still shares the target embed_tokens.
- fp4-weights build size = checkpoint choice: **W4A4 nvfp4** (real fp4 MAC on Ada TC, complex: compressed-tensors loader + activation quant + fp4 GEMM) vs **weight-only int4/nvfp4** (dequant, ~20–25% tax, simple, fit-first).
- draft KV: uniform with target (Step 0).

## Open logistics
- Is the 4090 accessible now (run-now) or are we developing to run later (dev-blind)?
- Which 30B + is there an NVFP4 checkpoint (dense-30B vs 30B-A3B MoE — changes the kernel)?
- Build order (KV kernel first / fp4 weights first / 8B-bf16 shortcut).

## Step 0 result — accept-rate fake-quant sweep (1.7B eagle3, k=2, 4 prompts) [DONE]
Branch `quant` (off `spec-decode`). Hook: `nanovllm/utils/kv_quant.py` fake_quant_kv (fp8 e4m3 per-token-head,
fp4 NVFP4-style E2M1 per-16-block) + attention.py hook + per-layer mount in model_runner.

| config | target KV | draft KV | accept_len | vs base | agreement |
|---|---|---|---|---|---|
| baseline_eagle3 | bf16 | bf16 | 1.719 | — | 1.00 |
| draft_fp4_only | bf16 | fp4 | 1.723 | +0.2% | 1.00 |
| fp4_bf16draft | fp4 | bf16 | 1.616 | −6.0% | 0.10 |
| uniform_fp4 | fp4 | fp4 | 1.605 | −6.6% | 0.10 |
| fp8_bf16draft | fp8 | bf16 | 1.694 | −1.5% | 0.32 |
| uniform_fp8 | fp8 | fp8 | 1.691 | −1.6% | 0.32 |

**Decisions from Step 0:**
- **Draft-KV quant is essentially free** (draft_fp4_only = baseline, agreement 1.00) → **go uniform (draft KV = target KV); do NOT build the bf16-draft-KV mixed pool** (uniform_fp4 vs fp4_bf16draft = ~0.7%).
- **Accept-rate cost is all on the target KV; fp8 ≪ fp4:** fp8 target ~−1.6%, fp4 target ~−6.6%. Real kernel tradeoff = fp8 (2× mem, ~2% accept) vs fp4 (4× mem, ~6–7% accept).
- Caveats: 1.7B (relative deltas transfer); **KV only, weights not yet quantized** (fp4-weights accept hit + head train/serve mismatch = next fake-quant measurement); agreement<1 = output-quality cost needing a real quality eval.

## Step 0b result — NVFP4 weight fake-quant (1.7B eagle3, weight-only E2M1 group16 + fp8 block scale) [DONE]
| config | weights | KV | accept_len | vs base | agreement |
|---|---|---|---|---|---|
| baseline_eagle3 | bf16 | bf16 | 1.719 | — | 1.00 |
| w_nvfp4 | nvfp4 | bf16 | 1.638 | −4.7% | 0.08 |
| w_nvfp4_kv_fp8 | nvfp4 | fp8 | 1.596 | −7.2% | 0.09 |
| w_nvfp4_kv_fp4 | nvfp4 | fp4 | 1.614 | −6.1% | 0.09 |

**Decisions from Step 0b:**
- NVFP4 weights cost only ~5% accept_len (head degrades gracefully) → **head re-distill is optional, not required**.
- Full fp4 stack (nvfp4 weights + fp4 KV) ≈ accept 1.61 (~−6%) → **EAGLE-3 survives full fp4; the plan holds.**
- Under fp4 weights, **fp8-vs-fp4 KV wash out** (weight-quant error dominates) → **use fp4 KV for max memory** on a full-fp4 target.
- Caveats: 1.7B (30B likely *more* robust → smaller drop); weight-only (real W4A4 slightly worse); agreement ~0.08 = output-quality cost, needs a real accuracy benchmark (GSM8K/MT-bench) on the 4090 — not visible in accept_len.

## Status
- EAGLE-3: done in nano-vllm (`spec-decode` branch), accept_len ~1.7, ~1.4×.
- Step 0 (KV fake-quant): **DONE** — go uniform KV.
- Step 0b (NVFP4 weight fake-quant): **DONE** — full fp4 stack keeps accept ~1.6 (~−6%); head re-distill optional; use fp4 KV.
- Local fake-quant conclusion: **the "30B fp4 + EAGLE-3" plan is validated on accept-rate.** Only open unknown local-side = output *quality* (needs a real accuracy benchmark on the 4090).
- **Triton paged decode kernel: DONE + locally validated** (`nanovllm/layers/paged_attn.py`). bf16 exact vs flash-attn (~2e-3); fp8 dequant + per-token-head scale structure validated via e5m2 (~4e-3). Kernel is dtype-agnostic.
  - **sm_86 finding**: Triton can't compile e4m3 (`fp8e4nv`) on the 3060 — only e5m2/e4b15. On the 4090 (sm_89) allocate the KV cache as `torch.float8_e4m3fn` (production) and the same kernel runs.
- **Decisions locked**: dense **Qwen3-32B** (nano-vllm is dense-only; MoE = separate build), **NVFP4 weights + fp4 KV**.
- **32B head wiring: DONE + validated** — `Eagle3Proposer` now reads dims from the head `config.json` (1.7B repro exact; 32B → hidden 5120/heads 64/head_dim 80), aux layers computed `(1, L//2-1, L-4)` → 32B `(1,31,60)`.
- **Benchmark file: DONE** — `bench_fp4_4090.py` (no_spec vs eagle3; throughput/TTFT/ITL/accept; bf16-draft-KV dropped per Step 0). MODEL points at the packed nvfp4 checkpoint; real kernels wired, so timings are representative on the 4090.
- **Real kernels: DONE + locally validated** (see "Build" below) — weight-only NVFP4 fused GEMM + fp4/fp8 paged KV (decode+varlen). Only residual work is 4090-only (checkpoint download, fp8-read on sm_89, perf tuning, VRAM fit).

### Build — DONE, locally validated on 3060 Ti (sm_86), ready to copy to the 4090
Decision: weight-only NVFP4 dequant (packed weights resident, dequant to fp16 inside the GEMM — no
fp4 tensor cores), fp4/fp8 KV with a real decode+varlen read kernel. Ported/adapted from AdaLLM
(`BenChaliah/NVFP4-on-4090-vLLM`); the fused GEMM is AdaLLM's fp16 kernel mode.

1. **NVFP4 weight GEMM** (`layers/nvfp4_gemm.py`, `layers/quant.py`, `layers/linear.py`, `utils/loader.py`):
   fused fp16 dequant GEMM; kernel-mode `LinearBase` holds uint8 `weight` [out,in/2] + fp16 `weight_scale`
   [out,in/16] (global `weight_scale_2` folded into the per-group scale at load, so merged qkv/gate_up stay
   correct); loader reads AngelSlim keys (`.weight` uint8 / `.weight_scale` / `.weight_scale_2`, `.input_scale`
   skipped) AND packs a bf16 checkpoint on the fly (local test path). QuantConfig toggled around the target
   build so the eagle3 head + lm_head/embed stay bf16 (matches the checkpoint's exclude list).
   Validated: fused GEMM == dequant-matmul to ~3e-4; 1.7B end-to-end coherent.
2. **KV quant** (`layers/paged_attn.py`, `layers/attention.py`, `model_runner.allocate_kv_cache`):
   torch quant-store (matches `fake_quant_kv` exactly -> accept == Step-0) + one dequant-in-loop
   paged-attention read kernel covering decode AND verify/prefill/prefix (per-query-row seq+causal-len map);
   single uniform pool dtype + fp32 scale pool, mounted per layer incl. the draft layer.
   Validated: kernel bf16-exact (2e-7) vs flash-attn, fp4 ~5e-4 (decode+varlen); full stack (nvfp4 weights +
   fp4 KV + eagle3) coherent end-to-end; non-quant path byte-exact (no regression).

### VRAM fit on the 4090 (measured 2026-07-23) + fp8-scale fix
Diagnostic (util 0.9, batched=512): `total=23.52G weights=21.09G peak=21.20G free=1.95G`. Weights alone
(packed 4-bit ~15.6G + **fp16 per-group scales ~3.9G** + tied embed/lm_head ~1.5G) nearly fill the 90%
line → `num_kvcache_blocks` computes ≤0 (assert fails); the eagle3 head (~1.3G) does not fit at all.
- **Root cause of the spike-OOM**: `max_num_batched_tokens` default 16384 → warmup runs a 16384-token
  prefill. Fix: `max_num_batched_tokens=2048` (bench now sets it) shrinks the warmup activation ~8×.
- **Weight-memory fix (implemented)**: store the per-group block scale as **fp8_e4m3** (the ckpt's native
  dtype; we were upcasting to fp16) via `weight_scale_fp8=True`. Frees ~2G (3.9→2.0), lossless, matches the
  Step-0b fake-quant. Global `scale_2` kept as a per-output-column vector applied in the GEMM (handles
  merged qkv/gate_up). → weights ~19.1G, eagle3 head fits. AdaLLM confirms this is the native format (it
  wastefully upcasts to fp16); AdaLLM has **no** dense-32B proof point (its ceiling = Gemma3-27B @ 19.83G).
- Implemented + validated on sm_86: `config.weight_scale_fp8`, `quant.QuantConfig.scale_fp8`,
  `linear.py` (fp8 scale param + `weight_scale_2` [out] vector), `nvfp4_gemm.py` (S2_ptr, `acc*=g`),
  `loader.py` (route native fp8 scale + expand scalar scale_2 per shard). GEMM scale_2 path matches
  reference (rel<5e-4); fp16 default path byte-exact (no regression); fp8 storage/loader ops OK; full
  eagle3 stack coherent. **4090-only residual**: the Triton fp8 e4m3 *load* in the GEMM (sm_89).

### Residual — 4090-only (cannot be done/validated on sm_86)
- **Download** `AngelSlim/Qwen3-32B_nvfp4` (packed weights) + `AngelSlim/Qwen3-32B_eagle3` (head); point
  `bench_fp4_4090.py:MODEL` at the nvfp4 dir (already set).
- **fp8 KV read kernel**: implemented but only compiles on sm_89 (Triton `fp8e4nv`); fp4 KV is the bench
  path and is fully validated locally. fp8 store + alloc are plain torch (arch-independent).
- **Perf tuning**: GEMM tile configs + the paged read kernel (BLOCK=block_size=256 tiling) are correct but
  untuned for Ada; expect iteration on the 4090. Timings are only meaningful there.
- **VRAM fit**: ~18 GB packed weights + ~3 GB embed/lm_head + eagle3 head is tight on 24 GB; may need to
  lower `gpu_memory_utilization`/`max_model_len`, or run `no_spec` (no head) first. This is a runtime knob,
  not a code gap.
- **Output-quality eval** (GSM8K/MT-bench) under full fp4 — the open accept-rate-invisible cost from Step-0b.
