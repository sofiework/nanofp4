"""Per-component VRAM probe for Qwen3-32B NVFP4 on 4090.
Usage: python probe_vram.py <no_spec|eagle3> <util> <mnbt> <max_num_seqs> <mode:probe|gen>
Prints the engine's own [allocate_kv_cache] line, then one line: PROBE_JSON {...}
"""
import os, sys, json, time
import torch
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-32B_nvfp4/")
DRAFT = os.path.expanduser("~/huggingface/Qwen3-32B_eagle3/")

name = sys.argv[1]
util = float(sys.argv[2])
mnbt = int(sys.argv[3])
mns  = int(sys.argv[4])
mode = sys.argv[5] if len(sys.argv) > 5 else "probe"
mml  = int(sys.argv[6]) if len(sys.argv) > 6 else 2048

CONFIGS = {
    "no_spec": dict(weight_quant="nvfp4", weight_scale_fp8=True, kv_quant_target="fp4"),
    "eagle3":  dict(weight_quant="nvfp4", weight_scale_fp8=True, kv_quant_target="fp4",
                    kv_quant_draft="fp4", enable_spec=True, num_spec_tokens=2,
                    spec_method="eagle3", draft_model=DRAFT),
}
cfg = CONFIGS[name]
ee = False if name == "no_spec" else True  # eagle3 is forced eager internally anyway

G = 1 << 30
t_load = time.perf_counter()
try:
    llm = LLM(MODEL, enforce_eager=ee, max_model_len=mml, max_num_batched_tokens=mnbt,
              max_num_seqs=mns, gpu_memory_utilization=util, **cfg)
except Exception as e:
    print("PROBE_JSON " + json.dumps(dict(
        name=name, util=util, mnbt=mnbt, mns=mns, mml=mml, fit=False,
        error=f"{type(e).__name__}: {str(e)[:200]}")))
    sys.exit(0)
load_s = time.perf_counter() - t_load

mr = llm.model_runner
model = mr.model
prop = getattr(mr, "proposer", None)

def group_bytes(named):
    g = {}
    for n, p in named:
        b = p.numel() * p.element_size()
        if p.dtype == torch.uint8 and n.endswith(".weight"):
            k = "packed_weight_u8"
        elif "weight_scale_2" in n:
            k = "weight_scale_2"
        elif "weight_scale" in n:
            k = "weight_scale"
        elif "embed_tokens" in n:
            k = "embed_tokens"
        elif "lm_head" in n:
            k = "lm_head"
        elif n.endswith("norm.weight") or "layernorm" in n:
            k = "norms"
        else:
            k = "other_bf16"
        g[k] = g.get(k, 0) + b
    return g

wb = group_bytes(model.named_parameters())
# tie check
et = dict(model.named_parameters()).get("embed_tokens.weight")
try:
    lm = model.lm_head.weight
    tied = (lm.data_ptr() == model.model.embed_tokens.weight.data_ptr())
except Exception:
    tied = None

pb = {}
if prop is not None:
    pb = group_bytes(prop.named_parameters())
    # d2t buffer
    for n, buf in prop.named_buffers():
        pb["buffers"] = pb.get("buffers", 0) + buf.numel() * buf.element_size()

# KV pools
def nb(x): return 0 if x is None else x.numel() * x.element_size()
kv = nb(mr.kv_cache) + nb(mr.kv_scale)
dkv = nb(getattr(mr, "draft_kv_cache", None)) + nb(getattr(mr, "draft_kv_scale", None))
blocks = mr.config.num_kvcache_blocks

free_after, total = torch.cuda.mem_get_info()
out = dict(
    name=name, util=util, mnbt=mnbt, mns=mns, mml=mml, fit=True, load_s=round(load_s, 1),
    tied_embed=tied, num_blocks=int(blocks),
    total_G=round(total/G, 3),
    weights_G={k: round(v/G, 4) for k, v in wb.items()},
    weights_sum_G=round(sum(wb.values())/G, 3),
    eagle3_head_G={k: round(v/G, 4) for k, v in pb.items()} if pb else {},
    eagle3_head_sum_G=round(sum(pb.values())/G, 3) if pb else 0.0,
    kv_pool_G=round(kv/G, 3), draft_kv_pool_G=round(dkv/G, 4),
    kv_bytes_per_block_MB=round((kv+dkv)/max(blocks,1)/1e6, 3),
    post_init_current_G=round(torch.cuda.memory_allocated()/G, 3),
    post_init_reserved_G=round(torch.cuda.memory_reserved()/G, 3),
    post_init_peak_G=round(torch.cuda.max_memory_allocated()/G, 3),
    post_init_free_G=round(free_after/G, 3),
)

if mode == "gen":
    tok = llm.tokenizer
    prompt = tok.apply_chat_template([{"role": "user", "content": "Explain attention in one sentence."}],
                                     tokenize=False, add_generation_prompt=True)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    o = llm.generate([prompt], SamplingParams(temperature=0.6, max_tokens=32), use_tqdm=False)
    out["gen_ok"] = True
    out["gen_s"] = round(time.perf_counter() - t0, 2)
    out["sample"] = o[0]["text"][:120] if isinstance(o[0], dict) else str(o[0])[:120]

print("PROBE_JSON " + json.dumps(out))
