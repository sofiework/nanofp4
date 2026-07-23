"""NVFP4 weight-only fused dequant GEMM (kernel mode) — keeps weights packed 4-bit resident so a
32B model fits on a 24 GB 4090, and dequantizes each 16-wide micro-block to fp16 *inside* the GEMM
(no fp4 tensor cores — plain fp16 tl.dot). Adapted from BenChaliah/NVFP4-on-4090-vLLM (AdaLLM),
simplified for nano-vllm: single fp16 path, and the two-level (per-group fp8 * global fp32) scale is
folded into one per-group fp16 scale at load time so nano-vllm's merged qkv_proj / gate_up_proj
(which concatenate q/k/v and gate/up along the output dim, each with its own global scale) stay correct.

Layout (matches modelopt / compressed-tensors NVFP4, e.g. AngelSlim/Qwen3-32B_nvfp4):
    weight        : uint8  [out, in//2]        two E2M1 codes per byte (low nibble = even in-index)
    weight_scale  : fp16   [out, in//group]    per-group scale (fp8 block-scale * global, folded)
E2M1 code bits: bit3=sign, bits[2:1]=exp, bit0=mantissa -> magnitudes {0,.5,1,1.5,2,3,4,6}.
"""
import torch
import triton
import triton.language as tl

# E2M1 magnitude grid (index == low 3 bits of the code); code = mag_idx | (sign<<3).
_GRID = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
_E2M1_MAX = 6.0


def _make_lut() -> torch.Tensor:
    lut = []
    for code in range(16):
        sign = -1.0 if (code & 0x8) else 1.0
        lut.append(sign * _GRID[code & 0x7].item())
    return torch.tensor(lut, dtype=torch.float16)


_LUT = _make_lut()
_lut_cache: dict = {}


def _lut_on(device) -> torch.Tensor:
    if device not in _lut_cache:
        _lut_cache[device] = _LUT.to(device)
    return _lut_cache[device]


@triton.jit
def _nvfp4_gemm_kernel(
    A_ptr, B_ptr, S_ptr, LUT_ptr, C_ptr, M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk, stride_sn, stride_sk, stride_cm, stride_cn,
    GROUP_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=mask_m[:, None] & mask_k[None, :], other=0).to(tl.float16)
        packed_idx = offs_k >> 1
        b_packed = tl.load(B_ptr + packed_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                           mask=mask_k[:, None] & mask_n[None, :], other=0)
        low = b_packed & 0x0F
        high = b_packed >> 4
        b_nibble = tl.where((offs_k & 1)[:, None] == 1, high, low)
        base = tl.load(LUT_ptr + b_nibble)
        scale_idx = offs_k // GROUP_SIZE
        scale = tl.load(S_ptr + scale_idx[:, None] * stride_sk + offs_n[None, :] * stride_sn,
                        mask=mask_k[:, None] & mask_n[None, :], other=0).to(tl.float16)
        b = base * scale
        acc += tl.dot(a, b)
        k += BLOCK_K
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def _select_config(m, k):
    if m <= 16:
        return dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3)
    if m <= 64:
        return dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=4)
    return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=4)


def nvfp4_linear(x: torch.Tensor, w_packed: torch.Tensor, w_scale: torch.Tensor,
                 bias: torch.Tensor | None, group_size: int) -> torch.Tensor:
    """x: [..., K] -> [..., N]. w_packed: uint8 [N, K//2]; w_scale: fp16 [N, K//group_size]."""
    out_dtype = x.dtype
    xh = x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.float16)
    x2d = xh.reshape(-1, xh.shape[-1])
    M, K = x2d.shape
    N = w_packed.shape[0]
    assert w_packed.shape[1] * 2 == K, f"packed {w_packed.shape} vs K={K}"
    c = torch.empty((M, N), device=x.device, dtype=torch.float16)
    cfg = _select_config(M, K)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    _nvfp4_gemm_kernel[grid](
        x2d, w_packed, w_scale, _lut_on(x.device), c, M, N, K,
        x2d.stride(0), x2d.stride(1), w_packed.stride(0), w_packed.stride(1),
        w_scale.stride(0), w_scale.stride(1), c.stride(0), c.stride(1),
        GROUP_SIZE=group_size, BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    if bias is not None:
        c = c + bias.to(c.dtype)
    return c.to(out_dtype).view(*x.shape[:-1], N)


@torch.no_grad()
def quantize_nvfp4(w: torch.Tensor, group_size: int = 16, fp8_scale: bool = True):
    """bf16 weight [out, in] -> (packed uint8 [out, in//2], scale fp16 [out, in//group]).
    Used both to fabricate a local nvfp4 checkpoint from a bf16 model, and as the reference encoder.
    fp8_scale mirrors real NVFP4 (block scale stored fp8_e4m3) so the injected error matches the
    fake-quant Step-0b measurement; set False if fp8 cast is unavailable."""
    out, inf = w.shape
    assert inf % group_size == 0
    ng = inf // group_size
    wf = w.float().reshape(out, ng, group_size)
    scale = (wf.abs().amax(-1, keepdim=True) / _E2M1_MAX).clamp_min(1e-12)
    if fp8_scale:
        scale = scale.to(torch.float8_e4m3fn).float().clamp_min(1e-12)
    xs = wf / scale
    grid = _GRID.to(w.device)
    mids = (grid[1:] + grid[:-1]) / 2
    mag = torch.bucketize(xs.abs(), mids)                       # 0..7
    code = (mag + (xs < 0).to(mag.dtype) * 8).reshape(out, inf).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, scale.reshape(out, ng).to(torch.float16).contiguous()


@torch.no_grad()
def dequant_nvfp4(w_packed: torch.Tensor, w_scale: torch.Tensor, group_size: int = 16,
                  dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Reference dequant (for testing / a fallback non-kernel path)."""
    lut = _LUT.to(w_packed.device, torch.float32)
    out, inh = w_packed.shape
    inf = inh * 2
    deq = torch.empty(out, inf, device=w_packed.device, dtype=torch.float32)
    deq[:, 0::2] = lut[(w_packed & 0x0F).long()]
    deq[:, 1::2] = lut[(w_packed >> 4).long()]
    ng = inf // group_size
    deq = deq.reshape(out, ng, group_size) * w_scale.float().reshape(out, ng, 1)
    return deq.reshape(out, inf).to(dtype)
