"""Quantized paged-KV attention read kernels + quant store helpers.

flash-attn (FA2) cannot read a quantized (fp8/fp4) paged KV cache, so when KV quant is on we replace
BOTH read paths with one Triton kernel that dequantizes in-loop:
  - decode      : one query token per seq
  - verify/prefill (with block_tables): several query tokens per seq (chunked / spec-verify / prefix)
Both reduce to the same shape — a set of query rows, each attending to its causal history [0, ctx) in
the paged cache — so a single kernel handles them: program = (query_row, query_head); per row we look up
its seq (QSEQ) and causal length (QCTX), walk that seq's block table, dequant each token, online softmax.

Store is done in plain torch (small, not bandwidth-critical): it writes the quantized cache + scales,
matching utils/kv_quant.fake_quant_kv exactly so accept-rate == the fake-quant Step-0 measurement.

Cache layouts ([block, token, kv_head, ...] like nano-vllm's bf16 pool):
  fp8 : cache float8_e4m3fn [.., D]      scale fp32 [..]            (per-(token,head),  amax/448)
  fp4 : cache uint8         [.., D//2]   scale fp32 [.., D//16]     (per-(token,head,16-block), amax/6)

fp8 note: sm_89+ (4090) required — Triton can't compile e4m3 (`fp8e4nv`) on sm_86. fp4 is int-packed
so it compiles/runs on any arch (locally validated on sm_86).
"""
import torch
import triton
import triton.language as tl

# E2M1 code -> signed value (bit3=sign, bits[2:1]=exp, bit0=mantissa); index == 4-bit code.
_E2M1_GRID = [0., .5, 1., 1.5, 2., 3., 4., 6.]
_LUT_F32 = torch.tensor([(-1. if c & 8 else 1.) * _E2M1_GRID[c & 7] for c in range(16)], dtype=torch.float32)
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0
_lut_cache: dict = {}


def _lut(device):
    if device not in _lut_cache:
        _lut_cache[device] = _LUT_F32.to(device)
    return _lut_cache[device]


@triton.jit
def _e2m1_decode(nib):
    # 4-bit E2M1 code -> value, arithmetically (no LUT gather). bit3=sign, bits[2:1]=exp, bit0=mantissa;
    # magnitudes {0,.5,1,1.5,2,3,4,6}. Matches paged_attn._LUT_F32 exactly (all values exact in fp16).
    sign = tl.where(nib >= 8, -1.0, 1.0)
    e = (nib >> 1) & 3
    m = (nib & 1).to(tl.float32)
    powv = (1 << e).to(tl.float32) * 0.5                       # 2^(e-1): e=0->.5,1->1,2->2,3->4
    mag = tl.where(e == 0, 0.5 * m, powv * (1.0 + 0.5 * m))     # e==0 subnormal (0 or .5)
    return sign * mag


@triton.jit
def _paged_attn_kernel(
    Q, KC, VC, KS, VS, BT, QSEQ, QCTX, LUT, O,
    scale, num_kv_groups,
    stride_qn, stride_qh, stride_on, stride_oh, stride_bt,
    KVH: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, QUANT: tl.constexpr,
    D_POW2: tl.constexpr, SCALE_FP8: tl.constexpr,
):
    qr = tl.program_id(0)
    h = tl.program_id(1)
    seq = tl.load(QSEQ + qr)
    ctx = tl.load(QCTX + qr)
    kvh = h // num_kv_groups
    # arange must be a power of 2; pad head_dim to D_POW2 and mask d<D (no-op when D is a power of 2, e.g.
    # 128; needed for the eagle3 draft head_dim 80). Padded dims load/contribute 0 and are not stored.
    d = tl.arange(0, D_POW2)
    dm = d < D
    q = tl.load(Q + qr * stride_qn + h * stride_qh + d, mask=dm, other=0.).to(tl.float32) * scale

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D_POW2], dtype=tl.float32)
    offs = tl.arange(0, BLOCK)
    Dp: tl.constexpr = D // 2
    Dg: tl.constexpr = D // 16
    n_blk = tl.cdiv(ctx, BLOCK)
    for blk in range(0, n_blk):
        phys = tl.load(BT + seq * stride_bt + blk)
        tok = blk * BLOCK + offs
        mask = tok < ctx
        kvmask = mask[:, None] & dm[None, :]             # token-in-ctx AND head_dim-in-D (pow2 pad)
        if QUANT == 0:                                   # bf16 / fp16
            ptr = phys * (BLOCK * KVH * D) + kvh * D + offs[:, None] * (KVH * D) + d[None, :]
            k = tl.load(KC + ptr, mask=kvmask, other=0.0).to(tl.float32)
            v = tl.load(VC + ptr, mask=kvmask, other=0.0).to(tl.float32)
        elif QUANT == 1:                                 # fp8 e4m3, per-(token,head) scale
            ptr = phys * (BLOCK * KVH * D) + kvh * D + offs[:, None] * (KVH * D) + d[None, :]
            k = tl.load(KC + ptr, mask=kvmask, other=0.0).to(tl.float32)
            v = tl.load(VC + ptr, mask=kvmask, other=0.0).to(tl.float32)
            sptr = phys * (BLOCK * KVH) + kvh + offs * KVH
            k = k * tl.load(KS + sptr, mask=mask, other=0.0)[:, None]
            v = v * tl.load(VS + sptr, mask=mask, other=0.0)[:, None]
        else:                                            # fp4 E2M1, per-(token,head,16-block) scale
            pidx = d >> 1
            bptr = phys * (BLOCK * KVH * Dp) + kvh * Dp + offs[:, None] * (KVH * Dp) + pidx[None, :]
            kb = tl.load(KC + bptr, mask=kvmask, other=0)
            vb = tl.load(VC + bptr, mask=kvmask, other=0)
            is_high = (d & 1)[None, :]
            k = _e2m1_decode(tl.where(is_high == 1, kb >> 4, kb & 0x0F))   # arithmetic, no LUT gather
            v = _e2m1_decode(tl.where(is_high == 1, vb >> 4, vb & 0x0F))
            gidx = d // 16
            sptr = phys * (BLOCK * KVH * Dg) + kvh * Dg + offs[:, None] * (KVH * Dg) + gidx[None, :]
            ks = tl.load(KS + sptr, mask=kvmask, other=0)
            vs = tl.load(VS + sptr, mask=kvmask, other=0)
            if SCALE_FP8:                                # KS/VS are uint8 views of an fp8_e4m3 pool
                ks = ks.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                vs = vs.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            k = k * ks
            v = v * vs

        s = tl.where(mask, tl.sum(q[None, :] * k, axis=1), -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(O + qr * stride_on + h * stride_oh + d, (acc / l_i).to(O.dtype.element_ty), mask=dm)


@triton.jit
def _paged_attn_splitkv_kernel(
    Q, KC, VC, KS, VS, BT, QSEQ, QCTX, OP, MP, LP,
    scale, num_kv_groups,
    stride_qn, stride_qh, stride_bt, stride_opq, stride_oph, stride_ops,
    stride_mpq, stride_mph,
    KVH: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, QUANT: tl.constexpr,
    D_POW2: tl.constexpr, SCALE_FP8: tl.constexpr, SPLIT: tl.constexpr,
):
    # Flash-decoding pass 1: program (query, head, split). Each split owns a contiguous slice of the
    # seq's KV blocks -> SPLIT× more programs than the per-row kernel (fills the GPU at bs=1) and each
    # walks 1/SPLIT of the context. Writes per-split partials (m, l, unnormalized acc); pass 2 merges.
    qr = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    seq = tl.load(QSEQ + qr)
    ctx = tl.load(QCTX + qr)
    kvh = h // num_kv_groups
    d = tl.arange(0, D_POW2)
    dm = d < D
    q = tl.load(Q + qr * stride_qn + h * stride_qh + d, mask=dm, other=0.).to(tl.float32) * scale
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D_POW2], dtype=tl.float32)
    offs = tl.arange(0, BLOCK)
    Dp: tl.constexpr = D // 2
    Dg: tl.constexpr = D // 16
    n_blk = tl.cdiv(ctx, BLOCK)
    blk_per = tl.cdiv(n_blk, SPLIT)
    blk0 = sp * blk_per
    blk1 = tl.minimum(blk0 + blk_per, n_blk)
    for blk in range(blk0, blk1):
        phys = tl.load(BT + seq * stride_bt + blk)
        tok = blk * BLOCK + offs
        mask = tok < ctx
        kvmask = mask[:, None] & dm[None, :]
        if QUANT == 0:
            ptr = phys * (BLOCK * KVH * D) + kvh * D + offs[:, None] * (KVH * D) + d[None, :]
            k = tl.load(KC + ptr, mask=kvmask, other=0.0).to(tl.float32)
            v = tl.load(VC + ptr, mask=kvmask, other=0.0).to(tl.float32)
        elif QUANT == 1:
            ptr = phys * (BLOCK * KVH * D) + kvh * D + offs[:, None] * (KVH * D) + d[None, :]
            k = tl.load(KC + ptr, mask=kvmask, other=0.0).to(tl.float32)
            v = tl.load(VC + ptr, mask=kvmask, other=0.0).to(tl.float32)
            sptr = phys * (BLOCK * KVH) + kvh + offs * KVH
            k = k * tl.load(KS + sptr, mask=mask, other=0.0)[:, None]
            v = v * tl.load(VS + sptr, mask=mask, other=0.0)[:, None]
        else:
            pidx = d >> 1
            bptr = phys * (BLOCK * KVH * Dp) + kvh * Dp + offs[:, None] * (KVH * Dp) + pidx[None, :]
            kb = tl.load(KC + bptr, mask=kvmask, other=0)
            vb = tl.load(VC + bptr, mask=kvmask, other=0)
            is_high = (d & 1)[None, :]
            k = _e2m1_decode(tl.where(is_high == 1, kb >> 4, kb & 0x0F))
            v = _e2m1_decode(tl.where(is_high == 1, vb >> 4, vb & 0x0F))
            gidx = d // 16
            sptr = phys * (BLOCK * KVH * Dg) + kvh * Dg + offs[:, None] * (KVH * Dg) + gidx[None, :]
            ks = tl.load(KS + sptr, mask=kvmask, other=0)
            vs = tl.load(VS + sptr, mask=kvmask, other=0)
            if SCALE_FP8:
                ks = ks.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                vs = vs.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            k = k * ks
            v = v * vs
        s = tl.where(mask, tl.sum(q[None, :] * k, axis=1), -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    tl.store(OP + qr * stride_opq + h * stride_oph + sp * stride_ops + d, acc, mask=dm)
    tl.store(MP + qr * stride_mpq + h * stride_mph + sp, m_i)
    tl.store(LP + qr * stride_mpq + h * stride_mph + sp, l_i)


@triton.jit
def _splitkv_merge_kernel(
    OP, MP, LP, O, stride_opq, stride_oph, stride_ops, stride_mpq, stride_mph,
    stride_on, stride_oh, D: tl.constexpr, D_POW2: tl.constexpr, SPLIT: tl.constexpr,
):
    # Flash-decoding pass 2: merge the SPLIT partials for (query, head) via log-sum-exp.
    qr = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D_POW2)
    dm = d < D
    m = -float("inf")
    l = 0.0
    acc = tl.zeros([D_POW2], dtype=tl.float32)
    for sp in range(0, SPLIT):
        m_s = tl.load(MP + qr * stride_mpq + h * stride_mph + sp)
        l_s = tl.load(LP + qr * stride_mpq + h * stride_mph + sp)
        acc_s = tl.load(OP + qr * stride_opq + h * stride_oph + sp * stride_ops + d, mask=dm, other=0.)
        m_new = tl.maximum(m, m_s)
        alpha = tl.exp(m - m_new)
        beta = tl.exp(m_s - m_new)
        l = l * alpha + l_s * beta
        acc = acc * alpha + acc_s * beta
        m = m_new
    tl.store(O + qr * stride_on + h * stride_oh + d, (acc / l).to(O.dtype.element_ty), mask=dm)


@triton.jit
def _paged_attn_tiled_kernel(
    Q, KC, VC, KS, VS, BT, MB_ROW0, MB_SEQ, MB_REND, QCTX, O,
    scale, num_kv_groups, Nq,
    stride_qn, stride_qh, stride_on, stride_oh, stride_bt,
    KVH: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, QUANT: tl.constexpr,
    D_POW2: tl.constexpr, SCALE_FP8: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Query-tiled paged attention: one program per (m-block, head). A BLOCK_M tile of query rows from ONE
    # seq attends to that seq's paged KV in BLOCK_N-token steps; each K/V tile is loaded + dequantized ONCE
    # and reused across all BLOCK_M queries via tl.dot (vs the per-row kernel re-dequantizing per query row).
    # BLOCK_N divides BLOCK (block_size), so each n-tile lives inside one physical block (one BT lookup).
    mb = tl.program_id(0)
    h = tl.program_id(1)
    row0 = tl.load(MB_ROW0 + mb)
    seq = tl.load(MB_SEQ + mb)
    rend = tl.load(MB_REND + mb)
    kvh = h // num_kv_groups

    rows = row0 + tl.arange(0, BLOCK_M)          # [M] global query-row indices
    rmask = rows < rend                          # valid rows in this seq's m-block
    ctx = tl.load(QCTX + rows, mask=rmask, other=0)   # [M] per-row causal length
    d = tl.arange(0, D_POW2)
    dm = d < D
    q = (tl.load(Q + rows[:, None] * stride_qn + h * stride_qh + d[None, :],
                 mask=rmask[:, None] & dm[None, :], other=0.).to(tl.float32) * scale).to(tl.float16)  # [M,D]

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_POW2], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    Dp: tl.constexpr = D // 2
    Dg: tl.constexpr = D // 16
    max_ctx = tl.max(ctx, axis=0)
    for n in range(0, max_ctx, BLOCK_N):
        tokpos = n + offs_n                          # [N] global token position
        blk = n // BLOCK                              # phys block index (BLOCK_N | BLOCK -> tile in one block)
        wtok = (n % BLOCK) + offs_n                   # token index within the physical block
        phys = tl.load(BT + seq * stride_bt + blk)
        nmask = tokpos < max_ctx
        kvmask = nmask[:, None] & dm[None, :]         # [N, D_POW2]
        if QUANT == 0:                                # bf16 / fp16
            ptr = phys * (BLOCK * KVH * D) + kvh * D + wtok[:, None] * (KVH * D) + d[None, :]
            k = tl.load(KC + ptr, mask=kvmask, other=0.0).to(tl.float32)
            v = tl.load(VC + ptr, mask=kvmask, other=0.0).to(tl.float32)
        elif QUANT == 1:                              # fp8 e4m3, per-(token,head) scale
            ptr = phys * (BLOCK * KVH * D) + kvh * D + wtok[:, None] * (KVH * D) + d[None, :]
            k = tl.load(KC + ptr, mask=kvmask, other=0.0).to(tl.float32)
            v = tl.load(VC + ptr, mask=kvmask, other=0.0).to(tl.float32)
            sptr = phys * (BLOCK * KVH) + kvh + wtok * KVH
            k = k * tl.load(KS + sptr, mask=nmask, other=0.0)[:, None]
            v = v * tl.load(VS + sptr, mask=nmask, other=0.0)[:, None]
        else:                                         # fp4 E2M1, per-(token,head,16-block) scale
            pidx = d >> 1
            bptr = phys * (BLOCK * KVH * Dp) + kvh * Dp + wtok[:, None] * (KVH * Dp) + pidx[None, :]
            kb = tl.load(KC + bptr, mask=kvmask, other=0)
            vb = tl.load(VC + bptr, mask=kvmask, other=0)
            is_high = (d & 1)[None, :]
            k = _e2m1_decode(tl.where(is_high == 1, kb >> 4, kb & 0x0F))
            v = _e2m1_decode(tl.where(is_high == 1, vb >> 4, vb & 0x0F))
            gidx = d // 16
            sptr = phys * (BLOCK * KVH * Dg) + kvh * Dg + wtok[:, None] * (KVH * Dg) + gidx[None, :]
            ks = tl.load(KS + sptr, mask=kvmask, other=0)
            vs = tl.load(VS + sptr, mask=kvmask, other=0)
            if SCALE_FP8:
                ks = ks.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                vs = vs.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            k = k * ks
            v = v * vs

        # qk = q @ k^T -> [M, N]; K/V tile loaded/dequantized once, reused across the M queries via tl.dot.
        qk = tl.dot(q, tl.trans(k.to(tl.float16)))    # [M, N]
        causal = (tokpos[None, :] < ctx[:, None]) & rmask[:, None] & nmask[None, :]
        qk = tl.where(causal, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])               # [M, N]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))   # [M, D_POW2]
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + rows[:, None] * stride_on + h * stride_oh + d[None, :],
             acc.to(O.dtype.element_ty), mask=rmask[:, None] & dm[None, :])


_QUANT_CODE = {"": 0, "fp8": 1, "fp4": 2}


def _build_mblocks(q_seq, block_m, device):
    """Group the (varlen, seq-contiguous) query rows into BLOCK_M tiles that never straddle a seq.
    Returns (mb_row0, mb_seq, mb_rend) int32 tensors, one entry per m-block."""
    seqs, counts = torch.unique_consecutive(q_seq, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    row0, mseq, rend = [], [], []
    starts_l = starts.tolist(); counts_l = counts.tolist(); seqs_l = seqs.tolist()
    for s, st, cnt in zip(seqs_l, starts_l, counts_l):
        for j in range(0, cnt, block_m):
            row0.append(st + j); mseq.append(s); rend.append(st + cnt)
    t = lambda xs: torch.tensor(xs, dtype=torch.int32, device=device)
    return t(row0), t(mseq), t(rend)


_TILED_BLOCK_M = 32     # query-tile height for the prefill/verify path
_TILED_BLOCK_N = 64     # KV-tile width (must divide block_size 256); keeps smem within the 4090 limit
import os
# flash-decoding splits for the decode path. Default 1 (per-row): split-KV is bit-exact but NOT a win at
# bs=1 (64 heads already fill ~half the SMs; the extra merge kernel + under-used splits cost more than the
# occupancy gain). Kept available via NANO_DECODE_SPLIT for batched / very-long-context decode experiments.
_DECODE_SPLIT = int(os.getenv("NANO_DECODE_SPLIT", "1"))


def paged_attn(q, k_cache, v_cache, block_tables, q_seq, q_ctx, scale,
               fmt="", k_scale=None, v_scale=None, prefill=False):
    """q: [Nq, H, D]. block_tables: [B, max_blocks]. q_seq/q_ctx: [Nq] (int32) — each query row's seq
    index and causal length. Returns o: [Nq, H, D]. fmt in {'', 'fp8', 'fp4'}.
    prefill=True (multiple query rows per seq) uses the query-tiled kernel (tl.dot, dequant-once);
    decode (one query row per seq) uses the per-row kernel."""
    Nq, H, D = q.shape
    BLOCK, KVH = k_cache.shape[1], k_cache.shape[2]
    o = torch.empty_like(q)
    # fp8_e4m3 scale pool (fp4 + kv_scale_fp8): pass a uint8 view + bitcast in-kernel (Triton fp8-ptr limit).
    scale_fp8 = k_scale is not None and k_scale.dtype == torch.float8_e4m3fn
    ks_arg = k_scale.view(torch.uint8) if scale_fp8 else k_scale
    vs_arg = v_scale.view(torch.uint8) if scale_fp8 else v_scale
    common = dict(KVH=KVH, D=D, BLOCK=BLOCK, QUANT=_QUANT_CODE[fmt],
                  D_POW2=triton.next_power_of_2(D), SCALE_FP8=scale_fp8)
    if prefill and Nq > 1:
        mb_row0, mb_seq, mb_rend = _build_mblocks(q_seq, _TILED_BLOCK_M, q.device)
        _paged_attn_tiled_kernel[(mb_row0.numel(), H)](
            q, k_cache, v_cache, ks_arg, vs_arg, block_tables, mb_row0, mb_seq, mb_rend, q_ctx, o,
            scale, H // KVH, Nq,
            q.stride(0), q.stride(1), o.stride(0), o.stride(1), block_tables.stride(0),
            BLOCK_M=_TILED_BLOCK_M, BLOCK_N=_TILED_BLOCK_N, num_warps=4, num_stages=2, **common,
        )
        return o
    # decode: split-KV (flash-decoding) fills the GPU at small batch and shortens each program's context
    # walk. Fixed SPLIT (no host sync / dynamic shape) so the decode path stays CUDA-graph-capturable;
    # splits past the context do no work and write neutral partials (-inf, 0) that the merge ignores.
    split = _DECODE_SPLIT
    if split > 1:
        op = torch.empty(Nq, H, split, triton.next_power_of_2(D), dtype=torch.float32, device=q.device)
        mp = torch.empty(Nq, H, split, dtype=torch.float32, device=q.device)
        lp = torch.empty(Nq, H, split, dtype=torch.float32, device=q.device)
        _paged_attn_splitkv_kernel[(Nq, H, split)](
            q, k_cache, v_cache, ks_arg, vs_arg, block_tables, q_seq, q_ctx, op, mp, lp,
            scale, H // KVH,
            q.stride(0), q.stride(1), block_tables.stride(0), op.stride(0), op.stride(1), op.stride(2),
            mp.stride(0), mp.stride(1), SPLIT=split, **common,
        )
        _splitkv_merge_kernel[(Nq, H)](
            op, mp, lp, o, op.stride(0), op.stride(1), op.stride(2), mp.stride(0), mp.stride(1),
            o.stride(0), o.stride(1), D=D, D_POW2=triton.next_power_of_2(D), SPLIT=split,
        )
        return o
    _paged_attn_kernel[(Nq, H)](
        q, k_cache, v_cache, ks_arg, vs_arg, block_tables, q_seq, q_ctx, _lut(q.device), o,
        scale, H // KVH,
        q.stride(0), q.stride(1), o.stride(0), o.stride(1), block_tables.stride(0),
        num_warps=int(os.getenv("NANO_PAGED_WARPS", "8")), **common,
    )
    return o


@torch.no_grad()
def _quant_fp4_kv(x: torch.Tensor):
    """x: [n, KVH, D] bf16 -> packed uint8 [n, KVH, D//2], scale fp32 [n, KVH, D//16].
    Matches utils/kv_quant.fake_quant_kv('fp4') (per-16-block amax/6, plain fp32 scale)."""
    n, KVH, D = x.shape
    xb = x.float().reshape(n, KVH, D // 16, 16)
    scale = (xb.abs().amax(-1, keepdim=True) / _E2M1_MAX).clamp_min(1e-12)
    xs = xb / scale
    grid = _paged_grid(x.device)
    mids = (grid[1:] + grid[:-1]) / 2
    mag = torch.bucketize(xs.abs(), mids)
    code = (mag + (xs < 0).to(mag.dtype) * 8).reshape(n, KVH, D).to(torch.uint8)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()
    return packed, scale.reshape(n, KVH, D // 16).contiguous()


_grid_cache: dict = {}


def _paged_grid(device):
    if device not in _grid_cache:
        _grid_cache[device] = torch.tensor(_E2M1_GRID, device=device)
    return _grid_cache[device]


@triton.jit
def _fp4_encode(xs):
    # normalized value -> E2M1 code 0..15 (magnitude via nearest-grid thresholds + sign bit). Matches
    # torch.bucketize on the grid midpoints used by _quant_fp4_kv / fake_quant_kv.
    a = tl.abs(xs)
    idx = ((a > 0.25).to(tl.int32) + (a > 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
           + (a > 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a > 3.5).to(tl.int32) + (a > 5.0).to(tl.int32))
    return idx + tl.where(xs < 0, 8, 0).to(tl.int32)


@triton.jit
def _fp4_pack(e, o):
    # e,o: [NB,8] even/odd elements of each 16-wide block. per-block amax/6 scale, encode, pack 2 codes/byte.
    amax = tl.max(tl.maximum(tl.abs(e), tl.abs(o)), axis=1)     # [NB]
    scale = tl.maximum(amax / 6.0, 1e-12)                       # _E2M1_MAX; literal for triton jit
    ce = _fp4_encode(e / scale[:, None])
    co = _fp4_encode(o / scale[:, None])
    return (ce | (co << 4)).to(tl.uint8), scale                # packed [NB,8], scale [NB]


@triton.jit
def _store_kv_fp4_kernel(K, V, KC, VC, KS, VS, SLOT,
                         s_n, s_h, sc_s, sc_h, ss_s, ss_h, NB: tl.constexpr, NB_POW2: tl.constexpr,
                         SCALE_FP8: tl.constexpr):
    # one program per (token, kv_head). -1 slot -> skip (in-kernel branch, no host sync) so the grid is
    # fixed [n, KVH] and the launch is CUDA-graph-capturable (unlike the boolean-mask torch path).
    t = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(SLOT + t)
    if slot == -1:
        return
    # NB = head_dim/16 blocks. arange must be a power of 2, so pad to NB_POW2 and mask nb<NB
    # (no-op when NB is already a power of 2, e.g. head_dim 128 -> 8; needed for head_dim 80 -> 5).
    nb = tl.arange(0, NB_POW2)[:, None]     # block index within head_dim
    p = tl.arange(0, 8)[None, :]            # pair index within a 16-block
    m = nb < NB
    even = nb * 16 + p * 2
    odd = even + 1
    off_in = t * s_n + h * s_h
    pk = nb * 8 + p                          # packed byte position (D//2)
    ks_idx = tl.arange(0, NB_POW2)
    ksm = ks_idx < NB
    kp, ks = _fp4_pack(tl.load(K + off_in + even, mask=m, other=0.).to(tl.float32),
                       tl.load(K + off_in + odd, mask=m, other=0.).to(tl.float32))
    vp, vs = _fp4_pack(tl.load(V + off_in + even, mask=m, other=0.).to(tl.float32),
                       tl.load(V + off_in + odd, mask=m, other=0.).to(tl.float32))
    if SCALE_FP8:                                    # KS/VS are uint8 views of an fp8_e4m3 pool
        ks = ks.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        vs = vs.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(KC + slot * sc_s + h * sc_h + pk, kp, mask=m)
    tl.store(KS + slot * ss_s + h * ss_h + ks_idx, ks, mask=ksm)
    tl.store(VC + slot * sc_s + h * sc_h + pk, vp, mask=m)
    tl.store(VS + slot * ss_s + h * ss_h + ks_idx, vs, mask=ksm)


@torch.no_grad()
def store_kv_quant(k, v, k_cache, v_cache, k_scale, v_scale, slot_mapping, fmt):
    """Write quantized k,v (+ scales) into the paged cache at slot_mapping (-1 = skip).
    fp4: a fixed-grid Triton kernel (CUDA-graph-capturable). fp8: torch (kept eager)."""
    n, KVH, D = k.shape
    if fmt == "fp4":
        # the kernel indexes head_dim as contiguous; k/v arrive as qkv-split views (head_dim IS unit-stride,
        # but make sure) — cheap since k,v are tiny ([n, KVH, D]).
        k, v = k.contiguous(), v.contiguous()
        kc, vc = k_cache.view(-1, KVH, D // 2), v_cache.view(-1, KVH, D // 2)
        # fp8_e4m3 scale pool: Triton can't take an fp8 pointer, so pass a uint8 view + bitcast in-kernel.
        scale_fp8 = k_scale.dtype == torch.float8_e4m3fn
        ks_src = k_scale.view(torch.uint8) if scale_fp8 else k_scale
        vs_src = v_scale.view(torch.uint8) if scale_fp8 else v_scale
        ks_, vs_ = ks_src.view(-1, KVH, D // 16), vs_src.view(-1, KVH, D // 16)
        _store_kv_fp4_kernel[(n, KVH)](
            k, v, kc, vc, ks_, vs_, slot_mapping,
            k.stride(0), k.stride(1), kc.stride(0), kc.stride(1), ks_.stride(0), ks_.stride(1),
            NB=D // 16, NB_POW2=triton.next_power_of_2(D // 16), SCALE_FP8=scale_fp8,
        )
        return
    # fp8: torch path (boolean mask, not graph-capturable -> fp8-KV runs eager).
    valid = slot_mapping != -1
    if not torch.any(valid):
        return
    sl = slot_mapping[valid].long()
    k, v = k[valid], v[valid]
    KVH, D = k.shape[1], k.shape[2]
    # fp8 (e4m3): per-(token,head) amax/448 scale
    sk = (k.abs().amax(-1) / _E4M3_MAX).clamp_min(1e-12)
    sv = (v.abs().amax(-1) / _E4M3_MAX).clamp_min(1e-12)
    k_cache.view(-1, KVH, D)[sl] = (k / sk.unsqueeze(-1)).to(k_cache.dtype)
    v_cache.view(-1, KVH, D)[sl] = (v / sv.unsqueeze(-1)).to(v_cache.dtype)
    k_scale.view(-1, KVH)[sl] = sk
    v_scale.view(-1, KVH)[sl] = sv
