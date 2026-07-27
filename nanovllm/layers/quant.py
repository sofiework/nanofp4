from dataclasses import dataclass


@dataclass
class QuantConfig:
    """Global NVFP4 weight-quant switch, read by LinearBase.__init__ to decide whether a layer
    holds packed uint8 weights (kernel mode) or a plain bf16 weight. Toggled by ModelRunner so the
    target model is quantized while the eagle3 draft head (built afterwards) stays bf16.

    scale_fp8: store the per-group block scale as fp8_e4m3 (the checkpoint's native dtype) instead of
    fp16, keeping the global scale_2 as a separate per-output-column vector applied in the GEMM. Halves
    scale memory (~2 GB on a 32B) and is lossless vs the checkpoint. fp8 scale LOAD in the Triton GEMM
    needs sm_89 (Ada/4090) — leave False on sm_86."""
    enabled: bool = False
    group_size: int = 16
    scale_fp8: bool = False
    # activation dtype at the nvfp4 GEMM input. "" -> fp16 (W4A16, default). "fp8_fake" -> per-token
    # fp8_e4m3 quant->dequant fake-quant, injecting exactly the W4A8 activation rounding error while the
    # matmul math stays fp16 (a MEASUREMENT hook for the precision study; no speedup). "fp8" -> real
    # fp8-tensor-core W4A8 path (Phase 2). Flipped at runtime to toggle A16 vs A8 on one loaded model.
    act_mode: str = ""


_QUANT = QuantConfig()


def get_quant_config() -> QuantConfig:
    return _QUANT


def set_weight_quant(enabled: bool, group_size: int = 16, scale_fp8: bool = False) -> None:
    _QUANT.enabled = enabled
    _QUANT.group_size = group_size
    _QUANT.scale_fp8 = scale_fp8


def set_act_mode(mode: str) -> None:
    """"" (W4A16) | "fp8_fake" (fake-quant precision hook) | "fp8" (real W4A8 tensor-core path)."""
    assert mode in ("", "fp8_fake", "fp8")
    _QUANT.act_mode = mode
