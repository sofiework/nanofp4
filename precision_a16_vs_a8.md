# W4A16 vs W4A8 activation-quant precision — Qwen3-32B NVFP4 (4090)

Same fp4 (E2M1) weights both runs; bf16 KV both runs. Only difference: activation dtype at every
nvfp4 GEMM input — fp16 (A16) vs per-token fp8_e4m3 (A8). Delta below is purely activation-quant error.
12 prompts, 32 greedy tokens each, 357 comparable positions.

| metric | all positions | prefill only |
|---|---|---|
| top-1 token agreement | 0.9944 | 1.0000 |
| top-5 overlap | 0.9625 | 0.9333 |
| logit cosine sim | 0.99730 | 0.99246 |
| logit relative-L2 | 0.0668 | 0.1203 |
| KL(p_A16 || p_A8) | 1.3090e-03 | 0.0000e+00 |
| max |logit diff| | 0.9594 | — |

Greedy divergence: mean token-match 0.932 over 32 tokens; 10/12 prompts never diverged.
