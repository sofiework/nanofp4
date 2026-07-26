import os
from dataclasses import dataclass
from transformers import AutoConfig

from nanovllm.utils.spec_decoding.tree import build_spec_tree, SpecTree


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
    num_spec_tokens: int = 0       # k draft tokens per step (0 when disabled); chain length when no tree given
    spec_method: str = "ngram"     # "ngram" | "eagle3"
    draft_model: str = ""          # eagle3 draft-head weights path
    speculative_token_tree: str = ""   # vLLM-style path-list literal; "" -> chain of num_spec_tokens
    spec_tree: SpecTree | None = None  # built in __post_init__ (eagle3 only); ngram stays a dynamic chain

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.spec_method == "eagle3":
            assert self.enable_spec, "eagle3 requires enable_spec=True"
            assert os.path.isdir(self.draft_model), "eagle3 requires draft_model path"
            self.spec_tree = build_spec_tree(self.speculative_token_tree, self.num_spec_tokens)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
