# Lab2 Quantization — Notes (COMPLETE Q1–Q10)

MIT 6.5940 Lab2, self-done. K-Means (Q1-3) + Linear (Q4-10). macOS/no-CUDA. Answers recorded here; lab2.md left unedited.

## Concepts
- **K-means quant:** cluster weights into `2^bw` centroids; store `labels`(n-bit idx, 1D `[numel]`) + `centroids`(fp32 codebook, 1D `[2^bw]`). Reconstruct `centroids[labels].view_as(w)`. Indexing `A[B]` → output shape = B's shape.
- **QAT:** fake-quant fwd, keep fp32 master weights, STE backprop updates fp32 (not centroids/params). Labels fixed; centroids re-derived (mean of cluster). ONE loss (quant = constraint). Train what you deploy; memory saving is inference-time.
- **Modern LLM quant:** all PTQ (≠QAT), linear-based — GPTQ (Hessian err comp), AWQ (activation-aware weight scale), SmoothQuant (migrate activation outliers→weights). Weight quant offline; activation quant online w/ offline-calibrated scales (outlier channels stable across inputs).

## Linear formulas
- `S=(r_max−r_min)/(q_max−q_min)`; `Z=int(round(q_min−r_min/S))` clamped (int → represents 0 exactly, enables int arithmetic).
- Quant `q=round(r/S)+Z` then clamp; dequant `r=S(q−Z)`.
- Weight: per-channel, symmetric (Z=0). Activation: per-tensor, asymmetric. Bias: `S_bias=S_in·S_w`, Z=0, int32; `Q_bias=q_bias−Linear[Z_in,q_w]` (inference uses Q_bias).

## Code answers
- **Q1** `n_clusters = 1 << bitwidth`
- **Q3** `codebook.centroids[k] = fp32_tensor[codebook.labels == k].mean()` (bool mask + mean)
- **Q5** `scale=(fp_max-fp_min)/(q_max-q_min)`; `zero_point=q_min - fp_min/scale`
- **Q6** `bias_scale = input_scale * weight_scale`
- **Q7** `quantized_linear`: `output = output.float()*(input_scale*weight_scale.view(1,-1)/output_scale)` then `+ output_zero_point`. `view(1,-1)`→`[1,oc]` broadcasts vs output `[batch,oc]`. int8×int8 accum int32.
- **Q8** `quantized_conv2d`: same but `weight_scale.view(1,-1,1,1)` (`[1,oc,1,1]`) vs 4D output. `-1`=auto-infer (numel/other dims=oc); weight_scale stored `[oc,1,1,1]`, values still just oc numbers.

## Q9 — full int8 PTQ pipeline
(1) fuse Conv-BN (equiv, folds BN into prev conv); (2) fwd hooks on calibration batch record activation min/max → activation S/Z (weight range from weights, no data needed); (3) replace by layer type: Conv→QuantizedConv2d, Linear→QuantizedLinear, pools→wrappers. Quantized classes = nn.Module overriding forward; store int weight/bias/SZ in `register_buffer` (int can't be Parameter); pool wrappers `.float()`→pool→int8 (no SZ). Data flows int8 throughout. Conv output S/Z taken **after ReLU**.
- **Q9.1** `return (x*255 - 128).clamp(-128,127).to(torch.int8)` — linear map (0,1)→(−128,127): a=255,b=−128.
- **Q9.2** No ReLU: output S/Z calibrated post-ReLU → neg region outside quantized range → final `.clamp` reproduces `max(0,x)`. One clamp does both int8-overflow-guard + ReLU (free).

## Q10 — k-means vs linear
- **K-means (non-uniform):** codebook fits weight distribution (non-uniform from clustering, not drift). ✅ better accuracy@low-bit, high compression. ❌ lookup/dequant→fp32 before MAC, no int arithmetic, weak HW, saves memory not latency; can't quantize activations.
- **Linear (uniform):** ✅ int-only MAC, HW-native (TensorCore/NPU/VNNI), saves latency+energy+memory. ❌ outliers stretch r_max → slightly worse accuracy.
- **Modern LLM:** Linear = standard; outlier weakness fixed by per-channel/group-wise + SmoothQuant/AWQ/GPTQ. LLM memory-bound → bitwidth↓ helps both, but linear enables W8A8 int pipeline (k-means can't). K-means idea survives as NF4(QLoRA)/codebook-quant(AQLM) niches.
- **Summary:** k-means = accuracy/compression but HW-unfriendly; linear = int-accelerable, full ecosystem → default.
