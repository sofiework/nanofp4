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
def _paged_attn_kernel(
    Q, KC, VC, KS, VS, BT, QSEQ, QCTX, LUT, O,
    scale, num_kv_groups,
    stride_qn, stride_qh, stride_on, stride_oh, stride_bt,
    KVH: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, QUANT: tl.constexpr,
):
    qr = tl.program_id(0)
    h = tl.program_id(1)
    seq = tl.load(QSEQ + qr)
    ctx = tl.load(QCTX + qr)
    kvh = h // num_kv_groups
    d = tl.arange(0, D)
    q = tl.load(Q + qr * stride_qn + h * stride_qh + d).to(tl.float32) * scale

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    offs = tl.arange(0, BLOCK)
    Dp: tl.constexpr = D // 2
    Dg: tl.constexpr = D // 16
    n_blk = tl.cdiv(ctx, BLOCK)
    for blk in range(0, n_blk):
        phys = tl.load(BT + seq * stride_bt + blk)
        tok = blk * BLOCK + offs
        mask = tok < ctx
        if QUANT == 0:                                   # bf16 / fp16
            ptr = phys * (BLOCK * KVH * D) + kvh * D + offs[:, None] * (KVH * D) + d[None, :]
            k = tl.load(KC + ptr, mask=mask[:, None], other=0.0).to(tl.float32)
            v = tl.load(VC + ptr, mask=mask[:, None], other=0.0).to(tl.float32)
        elif QUANT == 1:                                 # fp8 e4m3, per-(token,head) scale
            ptr = phys * (BLOCK * KVH * D) + kvh * D + offs[:, None] * (KVH * D) + d[None, :]
            k = tl.load(KC + ptr, mask=mask[:, None], other=0.0).to(tl.float32)
            v = tl.load(VC + ptr, mask=mask[:, None], other=0.0).to(tl.float32)
            sptr = phys * (BLOCK * KVH) + kvh + offs * KVH
            k = k * tl.load(KS + sptr, mask=mask, other=0.0)[:, None]
            v = v * tl.load(VS + sptr, mask=mask, other=0.0)[:, None]
        else:                                            # fp4 E2M1, per-(token,head,16-block) scale
            pidx = d >> 1
            bptr = phys * (BLOCK * KVH * Dp) + kvh * Dp + offs[:, None] * (KVH * Dp) + pidx[None, :]
            kb = tl.load(KC + bptr, mask=mask[:, None], other=0)
            vb = tl.load(VC + bptr, mask=mask[:, None], other=0)
            is_high = (d & 1)[None, :]
            k = tl.load(LUT + tl.where(is_high == 1, kb >> 4, kb & 0x0F))
            v = tl.load(LUT + tl.where(is_high == 1, vb >> 4, vb & 0x0F))
            gidx = d // 16
            sptr = phys * (BLOCK * KVH * Dg) + kvh * Dg + offs[:, None] * (KVH * Dg) + gidx[None, :]
            k = k * tl.load(KS + sptr, mask=mask[:, None], other=0.0)
            v = v * tl.load(VS + sptr, mask=mask[:, None], other=0.0)

        s = tl.where(mask, tl.sum(q[None, :] * k, axis=1), -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(O + qr * stride_on + h * stride_oh + d, (acc / l_i).to(O.dtype.element_ty))


_QUANT_CODE = {"": 0, "fp8": 1, "fp4": 2}


def paged_attn(q, k_cache, v_cache, block_tables, q_seq, q_ctx, scale,
               fmt="", k_scale=None, v_scale=None):
    """q: [Nq, H, D]. block_tables: [B, max_blocks]. q_seq/q_ctx: [Nq] (int32) — each query row's seq
    index and causal length. Returns o: [Nq, H, D]. fmt in {'', 'fp8', 'fp4'}."""
    Nq, H, D = q.shape
    BLOCK, KVH = k_cache.shape[1], k_cache.shape[2]
    o = torch.empty_like(q)
    _paged_attn_kernel[(Nq, H)](
        q, k_cache, v_cache, k_scale, v_scale, block_tables, q_seq, q_ctx, _lut(q.device), o,
        scale, H // KVH,
        q.stride(0), q.stride(1), o.stride(0), o.stride(1), block_tables.stride(0),
        KVH=KVH, D=D, BLOCK=BLOCK, QUANT=_QUANT_CODE[fmt],
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
                         s_n, s_h, sc_s, sc_h, ss_s, ss_h, NB: tl.constexpr):
    # one program per (token, kv_head). -1 slot -> skip (in-kernel branch, no host sync) so the grid is
    # fixed [n, KVH] and the launch is CUDA-graph-capturable (unlike the boolean-mask torch path).
    t = tl.program_id(0)
    h = tl.program_id(1)
    slot = tl.load(SLOT + t)
    if slot == -1:
        return
    nb = tl.arange(0, NB)[:, None]          # block index within head_dim
    p = tl.arange(0, 8)[None, :]            # pair index within a 16-block
    even = nb * 16 + p * 2
    odd = even + 1
    off_in = t * s_n + h * s_h
    pk = nb * 8 + p                          # packed byte position (D//2)
    ks_idx = tl.arange(0, NB)
    kp, ks = _fp4_pack(tl.load(K + off_in + even).to(tl.float32), tl.load(K + off_in + odd).to(tl.float32))
    tl.store(KC + slot * sc_s + h * sc_h + pk, kp)
    tl.store(KS + slot * ss_s + h * ss_h + ks_idx, ks)
    vp, vs = _fp4_pack(tl.load(V + off_in + even).to(tl.float32), tl.load(V + off_in + odd).to(tl.float32))
    tl.store(VC + slot * sc_s + h * sc_h + pk, vp)
    tl.store(VS + slot * ss_s + h * ss_h + ks_idx, vs)


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
        ks_, vs_ = k_scale.view(-1, KVH, D // 16), v_scale.view(-1, KVH, D // 16)
        _store_kv_fp4_kernel[(n, KVH)](
            k, v, kc, vc, ks_, vs_, slot_mapping,
            k.stride(0), k.stride(1), kc.stride(0), kc.stride(1), ks_.stride(0), ks_.stride(1),
            NB=D // 16,
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
