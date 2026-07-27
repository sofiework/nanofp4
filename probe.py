import torch, os, traceback
from nanovllm import LLM
M = os.path.expanduser("~/huggingface/Qwen3-32B_nvfp4/")
try:
    LLM(M, enforce_eager=True, max_model_len=2048, max_num_batched_tokens=512,
        gpu_memory_utilization=0.9, weight_quant="nvfp4", kv_quant_target="fp4")
except AssertionError:
    traceback.print_exc()
free, total = torch.cuda.mem_get_info()
print(f"total={total/2**30:.2f}G free={free/2**30:.2f}G "
      f"weights(current)={torch.cuda.memory_allocated()/2**30:.2f}G "
      f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}G")