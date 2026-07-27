import torch

# NVFP4-style E2M1 magnitude grid (sign handled separately) and e4m3 max.
_E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
_E4M3_MAX = 448.0
_FP4_BLOCK = 16   # NVFP4 micro-block size along head_dim


@torch.no_grad()
def fake_quant_kv(x: torch.Tensor, fmt: str, scale_fp8: bool = False) -> torch.Tensor:
    """Quant->dequant round-trip of a KV tensor x = [N, H, D] (N=token, H=kv_heads, D=head_dim).

    Injects exactly the error a real low-bit KV store would read back, so ACCEPT RATE is
    numerically equivalent to real quantized storage. Note: this is a *measurement* hook — it
    keeps the physical dtype (bf16) so it saves no memory/bandwidth and its timings are NOT
    representative of a real fp8/fp4 kernel.

    fmt: "" -> no-op | "fp8" (e4m3, per-token-head scale) | "fp4" (NVFP4-style E2M1, per-16-block scale)
    """
    if not fmt:
        return x
    xf = x.float()
    if fmt == "fp8":
        # per-(token, head) dynamic scale into e4m3 range; the native cast does the rounding
        scale = (xf.abs().amax(dim=-1, keepdim=True) / _E4M3_MAX).clamp_min(1e-12)
        q = (xf / scale).to(torch.float8_e4m3fn).float()
        return (q * scale).to(x.dtype)
    if fmt == "fp4":
        N, H, D = xf.shape
        assert D % _FP4_BLOCK == 0, f"head_dim {D} not divisible by fp4 block {_FP4_BLOCK}"
        xb = xf.reshape(N, H, D // _FP4_BLOCK, _FP4_BLOCK)
        scale = (xb.abs().amax(dim=-1, keepdim=True) / _E2M1[-1]).clamp_min(1e-12)
        if scale_fp8:                                    # match the fp8_e4m3 scale pool rounding
            scale = scale.to(torch.float8_e4m3fn).float().clamp_min(1e-12)
        xs = xb / scale
        grid = _E2M1.to(x.device)
        mid = (grid[1:] + grid[:-1]) / 2                 # round-to-nearest thresholds
        idx = torch.bucketize(xs.abs(), mid)
        deq = (xs.sign() * grid[idx] * scale).reshape(N, H, D)
        return deq.to(x.dtype)
    raise ValueError(f"unknown kv quant fmt: {fmt!r}")


@torch.no_grad()
def fake_quant_act_fp8(x: torch.Tensor) -> torch.Tensor:
    """Per-token (per last-dim row) fp8_e4m3 quant->dequant of an activation tensor x = [..., K].

    Injects exactly the rounding error a real W4A8 fp8-activation path would read, so logits/accuracy
    are numerically equivalent to real fp8 activations. MEASUREMENT hook only: keeps the physical dtype
    (fp16/bf16) and the GEMM stays fp16, so it saves no memory and is NOT a speed signal. Dynamic
    per-token scale = amax(|x|)/448 matches the standard W8A8/A8 activation quantizer.
    """
    xf = x.float()
    scale = (xf.abs().amax(dim=-1, keepdim=True) / _E4M3_MAX).clamp_min(1e-12)
    q = (xf / scale).to(torch.float8_e4m3fn).float()
    return (q * scale).to(x.dtype)


@torch.no_grad()
def fake_quant_nvfp4_weight(w: torch.Tensor, block: int = 16) -> torch.Tensor:
    """NVFP4 weight fake-quant of a linear weight w = [out, in]: E2M1 4-bit elements, per-`block`
    (16) group along the in-dim, with an fp8 (e4m3) block scale — matches AngelSlim / compressed-tensors
    nvfp4. Returns a lossy tensor of the same dtype. Weight-only: the real checkpoint is W4A4, so this
    *underestimates* the full error (activations not quantized here)."""
    if w.dim() != 2 or w.shape[1] % block:
        return w   # skip non-2D / odd shapes -> full precision (rare)
    out, cin = w.shape
    wf = w.float().reshape(out, cin // block, block)
    scale = (wf.abs().amax(dim=-1, keepdim=True) / _E2M1[-1]).clamp_min(1e-12)
    scale = scale.to(torch.float8_e4m3fn).float().clamp_min(1e-12)     # fp8 e4m3 block scale
    xs = wf / scale
    grid = _E2M1.to(w.device)
    mid = (grid[1:] + grid[:-1]) / 2
    idx = torch.bucketize(xs.abs(), mid)
    deq = (xs.sign() * grid[idx] * scale).reshape(out, cin)
    return deq.to(w.dtype)
