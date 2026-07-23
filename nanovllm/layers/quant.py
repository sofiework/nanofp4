from dataclasses import dataclass


@dataclass
class QuantConfig:
    """Global NVFP4 weight-quant switch, read by LinearBase.__init__ to decide whether a layer
    holds packed uint8 weights (kernel mode) or a plain bf16 weight. Toggled by ModelRunner so the
    target model is quantized while the eagle3 draft head (built afterwards) stays bf16."""
    enabled: bool = False
    group_size: int = 16


_QUANT = QuantConfig()


def get_quant_config() -> QuantConfig:
    return _QUANT


def set_weight_quant(enabled: bool, group_size: int = 16) -> None:
    _QUANT.enabled = enabled
    _QUANT.group_size = group_size
