import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # speculative decoding
    enable_spec: bool = False
    num_spec_tokens: int = 0       # k draft tokens per step (0 when disabled)
    spec_method: str = "ngram"     # "ngram" | "eagle3"
    draft_model: str = ""          # eagle3 draft-head weights path

    # KV-cache quantization (fake-quant measurement hook: "" | "fp8" | "fp4")
    kv_quant_target: str = ""      # target (main model) KV
    kv_quant_draft: str = ""       # draft (eagle3) KV; "" = bf16

    # weight quantization (fake-quant measurement: "" | "nvfp4"); target linears only, head stays bf16
    weight_quant: str = ""

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.spec_method == "eagle3":
            assert self.enable_spec, "eagle3 requires enable_spec=True"
            assert os.path.isdir(self.draft_model), "eagle3 requires draft_model path"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
