# Qwen3-32B NVFP4 on a single RTX 4090 — bring-up & debugging log

Chronicle of getting Qwen3-32B running on one 24 GB RTX 4090 (Ada, sm_89) via **NVFP4 4-bit weights +
fp4 KV cache + EAGLE-3**, from a from-scratch ~1200-line engine (nano-vllm `quant` branch). Each bug is
listed as symptom → root cause → fix → commit, in the order hit — because most were walls reached only
after clearing the previous one.

## Result (bs=1, Qwen3-32B, 4090)
| config | tok/s | ITL p50 | TTFT (steady) | accept_len | notes |
|---|---|---|---|---|---|
| no_spec (nvfp4 + fp4 KV + CUDA graph) | **10.3** | 101 ms | ~215–270 ms | 1.0 | 6× the initial 1.7 |
| eagle3 (+ draft head) | see run | — | — | ~1.5–1.6 | eager (tuple return blocks graph) |

**Context**: decode at bs=1 is memory-bandwidth-bound. Reference: AdaLLM (the NVFP4-on-Ada engine)
Gemma3-27B = 13.6 tok/s → ~11–12 scaled to 32B, so we're at **~parity with the reference**. A datacenter
H100+fp8 is ~5–7× faster, but that's ~3.3× the memory bandwidth at ~15× the cost, and fp8 (~33 GB) can't
fit a 4090 at all — 4-bit is *required* to run 32B on this card.

## Why it fits & runs
- **NVFP4 weight-only dequant GEMM** (`layers/nvfp4_gemm.py`): packed uint8 weights resident, dequantized
  to fp16 inside a Triton GEMM (no fp4 tensor cores). ~9.5 GB packed + fp8 scales.
- **fp4 KV cache** (`layers/paged_attn.py`): quant store + dequant-in-loop paged-attention read.
- **EAGLE-3** draft head (`utils/eagle3.py`): 1 midlayer, shares the target embed.
- Enabled purely by `LLM(...)` kwargs: `weight_quant="nvfp4"`, `weight_scale_fp8=True`,
  `kv_quant_target="fp4"`, `kv_quant_draft="fp4"`, `enable_spec`, `spec_method="eagle3"`, `draft_model`.

---

## Bug → fix log

### 1. `ModuleNotFoundError: nanovllm`
The repo folder name is irrelevant; `nanovllm` is the package inside it. **Fix**: `pip install -e .` in the
repo root (or run from the root). Not a code bug.

### 2. Warmup OOM (16384-token prefill)
**Symptom**: OOM during `warmup_model`, "tried to allocate 1.56 GiB". **Cause**: `max_num_batched_tokens`
defaults to **16384**, so warmup runs a 16384-token prefill → huge activation spike on top of ~21 GB
weights. Lowering `max_model_len` alone did NOT help (warmup total = `num_seqs × seq_len` stays 16384).
**Fix**: lower `max_num_batched_tokens` (bench → 256). That knob, not `max_model_len`, caps the spike.

### 3. `num_kvcache_blocks <= 0` — weights fill the card
**Symptom**: assert fails; no room for KV. Diagnostic showed **weights = 21.09 GB** (packed 15.6 + **fp16
per-group scales 3.9** + tied embed 1.5), nearly the whole 90%-util line. **Cause**: we upcast the
checkpoint's native **fp8_e4m3** block scale to fp16 on load, wasting ~2 GB. **Fix** (`0c61b34`): store the
per-group scale as **fp8_e4m3** (`weight_scale_fp8`), keep the global `scale_2` as a per-output-column
vector applied in the GEMM (handles merged qkv/gate_up). Lossless, frees ~2 GB → weights ~19 GB. AdaLLM
confirmed fp16-scale is its own (wasteful) choice; it has no dense-32B proof point either.
- **3a** (`35b721e`): the big `.weight` and its `.weight_scale_2` can land in **different safetensors
  shards** → derive output width from the co-located `.weight_scale`, and pre-collect all `scale_2` across
  shards into a map.

### 4. Decode painfully slow — 1.7 tok/s, ITL 589 ms
Three compounding causes, fixed in sequence (all in `layers/nvfp4_gemm.py`):
- **4a. Occupancy** (`634cadf`): at decode M=1 the GEMM grid is only `N/BLOCK_N` programs → most SMs idle
  → ~3% bandwidth. **Fix**: split-K GEMV kernel for small M (env-tunable `NVFP4_SPLIT_K`, default 8;
  `NVFP4_GEMV_MAX_M`). 4090 measured: decode BW 32 → 120 GB/s.
- **4b. fp8 scale load through a jit helper** (`a6045ca`): `S_ptr.to(tl.pointer_type(tl.uint8))` reinterpret
  works *inline* (`7cd8bf9`) but **not on a pointer passed into a `@triton.jit` helper** →
  `cannot cast int32 to fp8e4nv`. **Fix**: view the fp8 scale as `uint8` **on the host**, pass that in, and
  bitcast the bytes in-kernel — no in-kernel pointer reinterpret.
- **4c. Per-element LUT gather** (`a8dbd87`): even at 120 GB/s the kernel was dequant-bound (reads 4× less
  than fp16 yet 2–4× slower). **Fix**: decode E2M1 **arithmetically** (bit ops + one `tl.where`, no memory
  gather); bit-identical to the LUT. → decode ratio ~1.0× vs fp16 (parity — the GEMM is solved), 5.4 tok/s.
- **4d. CUDA graphs** (`cf78a99`): remaining ITL was ~64% eager overhead (64 layers × kernels + Python glue
  every token). **Fix**: enable graph capture for nvfp4 + fp4-KV decode. Required a **capturable fp4 KV
  store kernel** (`_store_kv_fp4_kernel`, fixed grid, skips `-1` slots in-kernel) — the old torch store used
  `torch.any()` (host sync) + boolean-mask indexing (data-dependent shape), both illegal under capture.
  Bug caught: the store kernel assumes contiguous head_dim, but decode k/v are qkv-split views → add
  `.contiguous()`. **→ 10.3 tok/s (1.73× local, 6× total)**. fp8-KV stays eager (non-capturable store).

### 5. Bench crash: "initialize the default process group twice"
**Symptom**: second config (eagle3) crashes building its `LLM`. **Cause**: two `LLM`s in one process each
call `dist.init_process_group`; also the first 20 GB model's VRAM lingers. **Fix** (`9750941`): run each
config in its own **spawned subprocess** (fresh CUDA context + process group, clean VRAM, result via Queue).

### 6. eagle3 `num_kvcache_blocks <= 0` — the draft head doesn't fit
**Cause**: the eagle3 config adds the ~1.7 GB draft head + an extra KV layer on top of 19 GB weights.
**Fix** (`5535c43`→`0584feb`→`74a6754`): **per-config `gpu_memory_utilization`** (no_spec 0.90 — needs
graph-pool headroom; eagle3 0.95 — eager, needs KV room) + `max_num_batched_tokens=256`. Added an
**instrumentation print** in `allocate_kv_cache` (total/used/peak/current/budget/block/blocks) so a
shortfall shows exactly how many bytes short. Result: KV budget **+1.13 GB, 47 blocks** — fits.

### 7. eagle3 draft KV `view(-1,8,40)` invalid
**Symptom**: draft-layer store crashes. **Cause**: the KV pool was **one uniform pool sized with the
target head_dim (128)** with the draft mounted as an extra layer — but Qwen3-32B target uses head_dim
**128** while the AngelSlim eagle3 draft head uses head_dim **80** (`eagle3.py:98`). head_dim-80 K/V can't
store into a 128-wide slot. Worked on 1.7B only because there both are 128; not fp4-specific. **Fix**
(`0363dc2`): `allocate_kv_cache` builds a **separate 1-layer draft pool** sized with the proposer's own
head_dim/num_kv_heads (`make_pool`), sharing the same block count `nb`. Kernels read head_dim from the
tensors, so no kernel change.

### 8. eagle3 `arange's range must be a power of 2`
**Symptom**: fp4 store/read kernels fail to compile for the draft. **Cause**: draft head_dim 80 →
`NB = 80/16 = 5` and `D = 80` are not powers of 2; Triton `tl.arange` requires pow2 (target's 128 → 8/128,
both pow2, hid this). **Fix** (`521ebf7`): pad the head_dim aranges to the next power of 2
(`NB_POW2` in the store kernel, `D_POW2` in the paged read) and **mask** the extra lanes — they load 0,
contribute 0 to the attention dot, aren't stored. No-op when head_dim is already pow2 → no regression.
Validated: head_dim 80 & 128 store+read vs torch reference rel ~2.5e-3.

### 9. eagle3 quant-kernel first-launch OOM at util 0.95
**Symptom**: launch itself OOMs after the KV pool is sized. **Cause**: a Triton kernel's **first launch
reserves persistent driver memory** (module + per-thread local scaled by SM count). The target paged-attn
read (first fired in `run_verify`) and the split-K GEMV reserve it *after* `allocate_kv_cache` already
consumed free memory up to util 0.95. **Fix**: `_warmup_quant_kernels` — force JIT-compile + first-launch
of every quant kernel (paged read per pool spec, fp4 store, split-K GEMV) during `warmup_model`, *before*
`allocate_kv_cache`, so `mem_get_info`'s free-memory budget accounts for the reservation. Dummies are freed
but the driver reservation stays visible.

---

## Key knobs (bench_fp4_4090.py)
- `weight_scale_fp8=True` — fp8 scale storage (required to fit 32B).
- `NVFP4_SPLIT_K` (env, default 8) — decode GEMV occupancy; `NVFP4_GEMV_MAX_M` (default 64), `=0` disables.
- per-config `gpu_memory_utilization` (no_spec 0.90 / eagle3 0.95), `max_num_batched_tokens=256`,
  `max_num_seqs=8`, `enforce_eager=False` (no_spec graphs; eagle3 forced eager internally).

## Cross-cutting lessons
- The **target-128 / draft-80 head_dim split** is unique to Qwen3-32B + AngelSlim eagle3 and caused bugs
  7 & 8; 1.7B never exposed them (dims matched).
- **fp8 e4m3** anything (weight scale, KV) is sm_89-only in Triton — untestable on the local 3060Ti; those
  paths were validated on the 4090. Everything else (fp4, arithmetic decode, split-K, graphs, pow2 pad,
  two-pool) was validated locally on sm_86 first.
- CUDA graph capture forbids host syncs and data-dependent shapes → every hot kernel must be **fixed-grid**
  and skip work in-kernel (the fp4 store rewrite, bug 4d).

## Remaining
- **Prefill/TTFT** still ~3–5× vs fp16 (compute-bound `tl.dot` + dequant vs cuBLAS) — lever is fp8
  tensor-cores for prefill (AdaLLM's `NVFP4_FP8` W4A8). Decode (headline) is done.
- Output-quality eval (GSM8K/MT-bench) under full fp4 — the accept-rate-invisible cost from Step 0b.
