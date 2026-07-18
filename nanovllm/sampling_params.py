from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    priority: int = 0 # normal/fast mode

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
