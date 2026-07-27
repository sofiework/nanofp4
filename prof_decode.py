"""Profile the DECODE steps (nvfp4 + fp4 KV). Prefill CTX tokens, then run STEPS decode steps inside a
cudaProfilerStart/Stop region so nsys captures only decode. Also prints measured ITL.
Usage: python prof_decode.py <ctx> <eager:0|1>
"""
import os, sys, time, torch
from nanovllm import LLM, SamplingParams

CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 512
EAGER = (len(sys.argv) > 2 and sys.argv[2] == "1")
STEPS = 40
M = os.path.expanduser("~/huggingface/Qwen3-32B_nvfp4/")
llm = LLM(M, enforce_eager=EAGER, max_model_len=4096, max_num_batched_tokens=256, max_num_seqs=8,
          gpu_memory_utilization=0.90, weight_quant="nvfp4", weight_scale_fp8=True,
          kv_quant_target="fp4", kv_scale_fp8=True)
tok = llm.tokenizer
base = tok("The quick brown fox jumps over the lazy dog. " * 30, add_special_tokens=False)["input_ids"]
ids = (base * (CTX // len(base) + 1))[:CTX]
torch.manual_seed(0)
llm.add_request(ids, SamplingParams(temperature=0.6, max_tokens=STEPS + 4))
seq = llm.scheduler.waiting[-1]

# run prefill (all chunks) until first decode token
while seq.num_completion_tokens < 1:
    llm.step()
torch.cuda.synchronize()

# time + profile the decode steps
t = []
torch.cuda.cudart().cudaProfilerStart()
last = seq.num_completion_tokens
for _ in range(STEPS):
    t0 = time.perf_counter()
    llm.step()
    torch.cuda.synchronize()
    t.append((time.perf_counter() - t0) * 1000)
torch.cuda.cudart().cudaProfilerStop()
t = sorted(t)
print(f"DECODE ctx={CTX} eager={EAGER} steps={len(t)} ITL_p50={t[len(t)//2]:.1f}ms "
      f"min={t[0]:.1f} tok/s={1000/t[len(t)//2]:.2f}")
