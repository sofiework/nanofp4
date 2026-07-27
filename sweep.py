"""Config sweep for the relaid [K/2,N] prefill GEMM. Reads NVFP4_BM/BN/BK/WARPS/STAGES from env
(so one config per process), times gate_up + down_proj at M=512, checks correctness. Driven by a
shell loop that sets the env vars."""
import os, torch
from nanovllm.layers.nvfp4_gemm import nvfp4_linear, quantize_nvfp4

G = 16
SHAPES = [("gate_up", 5120, 2 * 25600), ("down_proj", 25600, 5120)]
M = 512
tag = f"BM{os.environ.get('NVFP4_BM','-')} BN{os.environ.get('NVFP4_BN','-')} " \
      f"BK{os.environ.get('NVFP4_BK','-')} W{os.environ.get('NVFP4_WARPS','-')} " \
      f"S{os.environ.get('NVFP4_STAGES','-')}"


def cuda_time(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


out = []
for name, K, N in SHAPES:
    w = (torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.08)
    packed, scale = quantize_nvfp4(w, G)
    pk = packed.cuda().t().contiguous()
    sc = scale.cuda().to(torch.float8_e4m3fn).t().contiguous()
    s2 = torch.ones(N, device="cuda", dtype=torch.float32)
    x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
    ref = (x @ w.t()).float()
    try:
        o = nvfp4_linear(x, pk, sc, s2, None, G)
        err = (o.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        t = cuda_time(lambda: nvfp4_linear(x, pk, sc, s2, None, G))
        out.append(f"{name}={t:.3f}ms({2*M*N*K/t/1e9:.0f}TF,e{err:.1e})")
    except Exception as ex:
        out.append(f"{name}=FAIL({type(ex).__name__})")
    del w, packed, scale, pk, sc, x, ref
    torch.cuda.empty_cache()
print(f"[{tag}]  " + "  ".join(out))
