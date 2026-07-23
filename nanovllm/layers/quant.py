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


_QUANT = QuantConfig()


def get_quant_config() -> QuantConfig:
    return _QUANT


def set_weight_quant(enabled: bool, group_size: int = 16, scale_fp8: bool = False) -> None:
    _QUANT.enabled = enabled
    _QUANT.group_size = group_size
    _QUANT.scale_fp8 = scale_fp8
