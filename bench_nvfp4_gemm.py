"""Isolated NVFP4 GEMM microbench — decode (M=1) and prefill (M=512) at Qwen3-32B linear shapes,
comparing the packed-4bit dequant kernel against a torch.matmul fp16 upper bound, and reporting the
effective weight-read bandwidth (the decode ceiling is bandwidth, not FLOPs).

Use it to (a) confirm the GEMM is the bottleneck and (b) tune NVFP4_SPLIT_K:
    for s in 4 8 16 32; do NVFP4_SPLIT_K=$s python bench_nvfp4_gemm.py; done

Run on the 4090 (fp8 scale needs sm_89). On sm_86 set NVFP4_SPLIT_K and it still runs with fp16 scale
(pass --fp16-scale) for a logic/latency sanity check (bandwidth numbers won't transfer).
"""
import os, sys, time
import torch
from nanovllm.layers.nvfp4_gemm import nvfp4_linear, quantize_nvfp4, _GEMV_SPLIT_K, _GEMV_MAX_M

FP16_SCALE = "--fp16-scale" in sys.argv
G = 16
# (name, K=in, N=out) for Qwen3-32B: hidden 5120, 64 q-heads / 8 kv-heads / head_dim 128, interm 25600.
SHAPES = [
    ("qkv_proj",  5120, (64 + 2 * 8) * 128),   # 5120 -> 10240
    ("o_proj",    64 * 128, 5120),             # 8192 -> 5120
    ("gate_up",   5120, 2 * 25600),            # 5120 -> 51200
    ("down_proj", 25600, 5120),
]


def make(K, N):
    w = (torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.1)
    packed, scale = quantize_nvfp4(w, G)
    packed, scale = packed.cuda(), scale.cuda()
    if not FP16_SCALE:
        scale = scale.to(torch.float8_e4m3fn)     # exercise the real fp8 load path (sm_89)
    s2 = torch.ones(N, device="cuda", dtype=torch.float32)
    return packed, scale, s2, w


def timed(fn, iters=50):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3   # ms


def main():
    print(f"SPLIT_K={_GEMV_SPLIT_K} GEMV_MAX_M={_GEMV_MAX_M} scale={'fp16' if FP16_SCALE else 'fp8_e4m3'} "
          f"dev={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    for M in (1, 512):
        print(f"\n--- M={M} ({'decode' if M == 1 else 'prefill'}) ---")
        for name, K, N in SHAPES:
            try:
                packed, scale, s2, w = make(K, N)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {name:10s} K={K:5d} N={N:5d}  OOM building weight (too big for this GPU) — skip")
                continue
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)
            t_q = timed(lambda: nvfp4_linear(x, packed, scale, s2, None, G))
            t_f = timed(lambda: torch.matmul(x, w.t()))
            wbytes = packed.numel() + scale.numel() * scale.element_size()   # 4-bit weight + scale bytes
            gbs = wbytes / (t_q * 1e-3) / 1e9
            print(f"  {name:10s} K={K:5d} N={N:5d}  nvfp4={t_q:7.3f}ms  fp16={t_f:7.3f}ms  "
                  f"ratio={t_q/t_f:5.1f}x  BW={gbs:6.1f} GB/s")
            del packed, scale, s2, w, x
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
