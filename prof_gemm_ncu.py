"""Run ONE nvfp4 prefill GEMM (M=512, Qwen3-32B gate_up shape) for ncu to profile the kernel and
reveal whether it is Tensor(MMA)-bound or FMA/ALU(dequant)-bound. Warms up first so ncu --launch-skip
lands on a steady-state launch (fixed config, no JIT).
"""
import torch
from nanovllm.layers.nvfp4_gemm import nvfp4_linear, quantize_nvfp4

G = 16
M, K, N = 512, 5120, 2 * 25600            # gate_up: the biggest prefill GEMM
w = (torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.1)
packed, scale = quantize_nvfp4(w, G)
packed = packed.cuda(); scale = scale.cuda().to(torch.float8_e4m3fn)
# relaid [K/2, N] layout — this is what the real model runs (relayout_nvfp4_weights at load).
packed = packed.t().contiguous(); scale = scale.t().contiguous()
s2 = torch.ones(N, device="cuda", dtype=torch.float32)
x = torch.randn(M, K, device="cuda", dtype=torch.float16)

for _ in range(12):                        # warmup: JIT + steady state
    nvfp4_linear(x, packed, scale, s2, None, G)
torch.cuda.synchronize()
nvfp4_linear(x, packed, scale, s2, None, G)   # <- the profiled launch
torch.cuda.synchronize()
print("done")
