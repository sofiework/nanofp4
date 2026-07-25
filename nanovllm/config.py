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
    # store the fp4 KV per-16-block scale as fp8_e4m3 instead of fp32 — cuts KV 96->72 KB/token (-25%)
    # and halves the scale gather in the paged read kernel. fp8 load/store needs sm_89 (4090); keep False
    # on sm_86 (fp32 scale stays byte-exact for local regression). No effect unless kv_quant_target=="fp4".
    kv_scale_fp8: bool = False

    # weight quantization (fake-quant measurement: "" | "nvfp4"); target linears only, head stays bf16
    weight_quant: str = ""
    # store the NVFP4 per-group block scale as fp8_e4m3 (native ckpt dtype) instead of fp16 — halves
    # scale VRAM (~2 GB on a 32B), lossless. fp8 scale LOAD in the GEMM needs sm_89 (4090); keep False
    # on sm_86 (3060 Ti) where the fp16 path stays byte-exact for regression tests.
    weight_scale_fp8: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.spec_method == "eagle3":
            assert self.enable_spec, "eagle3 requires enable_spec=True"
            assert os.path.isdir(self.draft_model), "eagle3 requires draft_model path"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
