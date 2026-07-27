"""End-to-end prefill TTFT A/B on Qwen3-32B NVFP4 (4090), one model load. Toggles the GEMM config via
env between the OLD default (128x128/8w/4s) and the NEW tuned default (128x64/4w/3s), prefilling an
N-token prompt and timing prefill->first-token. No spec. Confirms the microbench win translates."""
import os, time, torch
from nanovllm import LLM, SamplingParams
import nanovllm.layers.nvfp4_gemm as G

MODEL = os.path.expanduser("~/huggingface/Qwen3-32B_nvfp4/")
N = 512
llm = LLM(MODEL, enforce_eager=True, weight_quant="nvfp4", weight_scale_fp8=True, kv_quant_target="fp4",
          max_model_len=1024, max_num_batched_tokens=1024, max_num_seqs=1, gpu_memory_utilization=0.90)
tok = llm.tokenizer
base = tok("The quick brown fox jumps over the lazy dog. " * 20, add_special_tokens=False)["input_ids"]
ids = (base * (N // len(base) + 1))[:N]


def prefill_once():
    llm.add_request(list(ids), SamplingParams(temperature=1.0, max_tokens=1, ignore_eos=True))
    torch.cuda.synchronize(); t = time.perf_counter()
    while not llm.is_finished():
        llm.step()
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 1e3


def measure(label, env):
    os.environ.update({k: str(v) for k, v in env.items()})
    for _ in range(2):
        prefill_once()                          # warmup (JIT the config)
    ts = sorted(prefill_once() for _ in range(8))
    print(f"{label:28s} prefill TTFT (N={N}): median={ts[len(ts)//2]:.1f}ms  min={ts[0]:.1f}ms")
    return ts[len(ts) // 2]


old = measure("OLD 128x128/8w/4s", dict(NVFP4_BM=128, NVFP4_BN=128, NVFP4_BK=64, NVFP4_WARPS=8, NVFP4_STAGES=4))
# clear overrides -> use the new baked-in default
for k in ("NVFP4_BM", "NVFP4_BN", "NVFP4_BK", "NVFP4_WARPS", "NVFP4_STAGES"):
    os.environ.pop(k, None)
new = measure("NEW 128x64/4w/3s (default)", {})
print(f"\nprefill speedup: {old/new:.2f}x  ({old:.1f} -> {new:.1f} ms)")
