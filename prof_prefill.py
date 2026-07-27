"""Profile a single prefill of N tokens (nvfp4 + fp4 KV, eager). Region bounded by cudaProfilerStart/Stop
so nsys --capture-range=cudaProfilerApi captures ONLY the prefill, not model load.
Usage: python prof_prefill.py <N_tokens>
"""
import os, sys, torch
from nanovllm import LLM, SamplingParams

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
M = os.path.expanduser("~/huggingface/Qwen3-32B_nvfp4/")
llm = LLM(M, enforce_eager=True, max_model_len=2048, max_num_batched_tokens=256, max_num_seqs=1,
          gpu_memory_utilization=0.90, weight_quant="nvfp4", weight_scale_fp8=True, kv_quant_target="fp4")
tok = llm.tokenizer
base = tok("The quick brown fox jumps over the lazy dog. " * 20, add_special_tokens=False)["input_ids"]
ids = (base * (N // len(base) + 1))[:N]

# warm once (short) so JIT/autotune isn't in the profiled region
llm.add_request(ids[:64], SamplingParams(temperature=0.6, max_tokens=1))
while not llm.is_finished():
    llm.step()
torch.cuda.synchronize()

llm.add_request(ids, SamplingParams(temperature=0.6, max_tokens=1))
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
while not llm.is_finished():
    llm.step()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print(f"profiled prefill N={N} done")
