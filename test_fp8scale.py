import os, sys, torch
from nanovllm import LLM, SamplingParams
M = os.path.expanduser("~/huggingface/Qwen3-32B_nvfp4/")
KVFP8 = sys.argv[1] == "fp8" if len(sys.argv) > 1 else True
llm = LLM(M, enforce_eager=True, max_model_len=2048, max_num_batched_tokens=256, max_num_seqs=1,
          gpu_memory_utilization=0.90, weight_quant="nvfp4", weight_scale_fp8=True,
          kv_quant_target="fp4", kv_scale_fp8=KVFP8)
mr = llm.model_runner
print(f"KV_SCALE_FP8={KVFP8} | kv_scale dtype={mr.kv_scale.dtype} shape={tuple(mr.kv_scale.shape)} "
      f"| kv_cache dtype={mr.kv_cache.dtype} | blocks={mr.config.num_kvcache_blocks}")
block_MB = (mr.kv_cache[:, :, :1].numel() * mr.kv_cache.element_size()
            + mr.kv_scale[:, :, :1].numel() * mr.kv_scale.element_size()) / 1e6
print(f"KV bytes/block = {block_MB:.2f} MB")
tok = llm.tokenizer
base = tok("The quick brown fox jumps over the lazy dog. " * 30, add_special_tokens=False)["input_ids"]
ids = (base * 3)[:600]   # 600 tokens -> chunked at mnbt=256 -> exercises the fp4 paged read w/ fp8 scale
torch.manual_seed(0)
o = llm.generate([ids], SamplingParams(temperature=0.6, max_tokens=48), use_tqdm=False)
print("SAMPLE:", repr(o[0]["text"][:220]))
