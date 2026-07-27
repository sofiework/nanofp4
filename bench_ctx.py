"""Context-length + prefill/decode timing bench (one engine load, multiple prompt lengths).
Usage: python bench_ctx.py <no_spec|eagle3> <util> <mnbt> <mns> <max_model_len> <plen1,plen2,...> <gen>
Prints CTX_JSON {...} per prompt length.
"""
import os, sys, json, time, statistics
import torch
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-32B_nvfp4/")
DRAFT = os.path.expanduser("~/huggingface/Qwen3-32B_eagle3/")

name = sys.argv[1]; util = float(sys.argv[2]); mnbt = int(sys.argv[3]); mns = int(sys.argv[4])
mml = int(sys.argv[5]); plens = [int(x) for x in sys.argv[6].split(",")]; GEN = int(sys.argv[7])
RESULTS = f"ctx_results_{name}.jsonl"

def emit(d):
    line = "CTX_JSON " + json.dumps(d)
    print(line, flush=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(d) + "\n"); f.flush()

KVFP8 = os.getenv("KV_SCALE_FP8", "0") == "1"
CONFIGS = {
    "no_spec": dict(weight_quant="nvfp4", weight_scale_fp8=True, kv_quant_target="fp4", kv_scale_fp8=KVFP8),
    "eagle3":  dict(weight_quant="nvfp4", weight_scale_fp8=True, kv_quant_target="fp4", kv_scale_fp8=KVFP8,
                    kv_quant_draft="fp4", enable_spec=True, num_spec_tokens=2,
                    spec_method="eagle3", draft_model=DRAFT),
}
cfg = CONFIGS[name]
ee = False if name == "no_spec" else True

llm = LLM(MODEL, enforce_eager=ee, max_model_len=mml, max_num_batched_tokens=mnbt,
          max_num_seqs=mns, gpu_memory_utilization=util, **cfg)
tok = llm.tokenizer

# base text to tile into token ids of a target length
base = tok("The quick brown fox jumps over the lazy dog. In a distant land, engineers "
           "optimized inference kernels to squeeze every last byte of GPU memory. ",
           add_special_tokens=False)["input_ids"]

def pct(xs, q):
    return statistics.quantiles(xs, n=100)[q-1] if len(xs) > 1 else (xs[0] if xs else 0.0)

def make_ids(n):
    ids = (base * (n // len(base) + 1))[:n]
    return ids

# warmup: compile the tiled paged kernel + graphs so the first TIMED prefill isn't JIT-inflated
llm.add_request(make_ids(400), SamplingParams(temperature=0.6, max_tokens=2))
while not llm.is_finished():
    llm.step()
torch.cuda.synchronize()

for plen in plens:
    print(f"START plen={plen} @ {round(time.perf_counter(),1)}s", flush=True)
    ids = make_ids(plen)
    torch.manual_seed(0)
    try:
        llm.add_request(ids, SamplingParams(temperature=0.6, max_tokens=GEN))
    except Exception as e:
        emit(dict(name=name, util=util, plen=plen, error=str(e)[:150]))
        continue
    seq = llm.scheduler.waiting[-1]
    t0 = time.perf_counter(); ttft = None; stamps = []; target_fwd = 0; last_n = 0
    prefill_steps = 0
    try:
        while not llm.is_finished():
            _, num_tokens = llm.step()
            if num_tokens > 0:
                prefill_steps += 1
            if num_tokens < 0:
                target_fwd += 1
            n = seq.num_completion_tokens
            if n > last_n:
                now = time.perf_counter()
                if ttft is None:
                    ttft = (now - t0) * 1000
                stamps.append(now); last_n = n
    except Exception as e:
        emit(dict(name=name, util=util, plen=plen, run_error=str(e)[:150]))
        # drain scheduler so next prompt starts clean
        try:
            while not llm.is_finished():
                llm.step()
        except Exception:
            pass
        continue
    gen = seq.num_completion_tokens
    itl = [(stamps[i]-stamps[i-1])*1000 for i in range(1, len(stamps))]
    decode_s = (stamps[-1]-stamps[0]) if len(stamps) > 1 else 1e-9
    out = dict(
        name=name, util=util, kvfp8=KVFP8, mnbt=mnbt, mns=mns, plen=plen, gen=gen,
        prefill_steps=prefill_steps,
        ttft_ms=round(ttft or 0, 1),
        prefill_tok_per_s=round(plen / ((ttft or 1e-9)/1000), 1),
        decode_tok_s=round((gen-1)/decode_s, 2),
        itl_p50_ms=round(pct(itl, 50), 2), itl_p99_ms=round(pct(itl, 99), 2),
        accept_len=round((gen-1)/target_fwd, 3) if target_fwd else 1.0,
        total_s=round(stamps[-1]-t0, 1) if stamps else None,
    )
    emit(out)
