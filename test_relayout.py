"""Validate nvfp4_linear on both weight layouts: legacy [N,K/2] vs relaid [K/2,N] vs dequant-matmul ref."""
import torch
from nanovllm.layers.nvfp4_gemm import nvfp4_linear, quantize_nvfp4, dequant_nvfp4
torch.manual_seed(0)
dev = "cuda"


def rel(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)

for (K, N) in [(5120, 10240), (8192, 5120), (25600, 5120), (5120, 51200)]:
    w = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05)
    packed, scale = quantize_nvfp4(w, 16, fp8_scale=False)   # [N,K/2] uint8, [N,K/16] fp16
    s2 = torch.ones(N, device=dev, dtype=torch.float32)
    deq = dequant_nvfp4(packed, scale, None, 16, torch.float32)   # [N,K]
    for M in [1, 8, 256]:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1
        ref = (x.float() @ deq.t())                              # [M,N]
        legacy = nvfp4_linear(x, packed, scale, s2, None, 16).float()
        relaid = nvfp4_linear(x, packed.t().contiguous(), scale.t().contiguous(), s2, None, 16).float()
        print(f"K={K:5d} N={N:5d} M={M:3d} | legacy-vs-ref {rel(legacy, ref):.3e} "
              f"relaid-vs-ref {rel(relaid, ref):.3e} relaid-vs-legacy {rel(relaid, legacy):.3e}")
