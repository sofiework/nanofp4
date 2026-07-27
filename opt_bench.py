"""Prefill NVFP4 GEMM optimization microbench (no model load). Times each Qwen3-32B linear shape at
M=512, both weight layouts (legacy [N,K/2] vs relaid [K/2,N]), checks correctness vs cublas fp16, and
reports ms / TFLOPS / ratio. Rerun after each kernel edit.
    python opt_bench.py            # default M=512
    python opt_bench.py 256 1024   # extra M values
"""
import sys, torch
from nanovllm.layers.nvfp4_gemm import nvfp4_linear, quantize_nvfp4

G = 16
SHAPES = [
    ("qkv_proj",  5120, (64 + 2 * 8) * 128),
    ("o_proj",    64 * 128, 5120),
    ("gate_up",   5120, 2 * 25600),
    ("down_proj", 25600, 5120),
]
Ms = [int(a) for a in sys.argv[1:]] or [512]


def cuda_time(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters      # ms


def main():
    dev = torch.cuda.get_device_name(0)
    print(f"dev={dev} cap={torch.cuda.get_device_capability(0)}")
    for M in Ms:
        print(f"\n=== M={M} ===")
        print(f"{'shape':10s} {'layout':7s} {'ms':>8s} {'TFLOPS':>8s} {'cublas':>8s} {'ratio':>6s} {'max_err':>9s}")
        for name, K, N in SHAPES:
            w = (torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.08)
            packed, scale = quantize_nvfp4(w, G)                  # [N,K/2], [N,K/G]
            packed, scale = packed.cuda(), scale.cuda().to(torch.float8_e4m3fn)
            s2 = torch.ones(N, device="cuda", dtype=torch.float32)
            x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
            ref = (x @ w.t()).float()
            flops = 2 * M * N * K
            t_cub = cuda_time(lambda: torch.matmul(x, w.t()))
            for lname, pk, sc in (("legacy", packed, scale),
                                  ("relaid", packed.t().contiguous(), scale.t().contiguous())):
                out = nvfp4_linear(x, pk, sc, s2, None, G)
                err = (out.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
                t = cuda_time(lambda: nvfp4_linear(x, pk, sc, s2, None, G))
                print(f"{name:10s} {lname:7s} {t:8.3f} {flops/t/1e9:8.1f} {t_cub*1e3:8.1f}us "
                      f"{t/t_cub:6.2f} {err:9.2e}")
            del w, packed, scale, x, ref
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
