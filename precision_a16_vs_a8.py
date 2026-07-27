"""Precision: W4A16 vs W4A8 on Qwen3-32B NVFP4 (4090). Phase 1 — fake-fp8 activation quant.

Same fp4 (E2M1) weights in both runs; the ONLY difference is the activation dtype at every nvfp4
GEMM input: fp16 (A16) vs per-token fp8_e4m3 fake-quant (A8, via layers.quant act_mode="fp8_fake").
KV is bf16 in both runs so the measured delta is purely the activation-quant error (W4A16 -> W4A8).

Method (one model load, act_mode flipped at runtime -> requires enforce_eager so graphs don't bake it):
  * Sampler patched to argmax (deterministic greedy); compute_logits wrapped to capture per-step logits.
  * For each prompt, greedy-decode N tokens under A16 then A8. Positions t=0..d (d = first token that
    differs) are on IDENTICAL context in both runs, so their logits are directly comparable.
  * Per comparable position: top-1 match, top-5 overlap, cosine, relative-L2, KL(p16||p8), max|dlogit|.
  * Report cross-prompt aggregates + prefill-only (t=0) aggregates + greedy divergence stats.

Run:  python precision_a16_vs_a8.py
"""
import os, json, types, statistics
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.quant import set_act_mode

MODEL = os.path.expanduser("~/huggingface/Qwen3-32B_nvfp4/")
N = 32                      # greedy tokens per prompt
PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list.",
    "Explain how a transformer attention mechanism works, step by step.",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h? Show your work.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "What are the main causes of the French Revolution?",
    "Implement binary search in C and explain its time complexity.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Describe the process of photosynthesis at the molecular level.",
    "What is the difference between TCP and UDP? Give an example use case for each.",
    "Prove that the square root of 2 is irrational.",
    "Translate 'The weather is beautiful today' into French, German, and Japanese.",
    "List five strategies for reducing carbon emissions in a large city.",
]


def greedy(self, logits, temperatures):
    return logits.argmax(dim=-1)


def top5(v):
    return set(torch.topk(v, 5).indices.tolist())


def compare(a, b):
    """a,b: 1-D logit vectors (float, cpu). Returns per-position metrics dict."""
    a32, b32 = a.double(), b.double()
    pa = torch.softmax(a32, -1)
    pb = torch.softmax(b32, -1)
    kl = (pa * (pa.clamp_min(1e-12).log() - pb.clamp_min(1e-12).log())).sum().item()
    cos = torch.nn.functional.cosine_similarity(a32, b32, dim=0).item()
    rel_l2 = ((b32 - a32).norm() / a32.norm().clamp_min(1e-12)).item()
    return dict(
        top1=int(a.argmax().item() == b.argmax().item()),
        top5=len(top5(a) & top5(b)) / 5.0,
        cos=cos, rel_l2=rel_l2, kl=kl,
        max_abs=(b32 - a32).abs().max().item(),
    )


def run_mode(llm, prompt_ids, mode):
    """Greedy-decode N tokens; return (token_list, per_step_logits[list of 1D cpu float])."""
    set_act_mode(mode)
    cap = []
    model = llm.model_runner.model
    orig = model.compute_logits

    def cap_fn(h):
        out = orig(h)                       # gpu logits -> returned to sampler unchanged
        cap.append(out.detach().float().cpu())
        return out
    model.compute_logits = cap_fn
    llm.add_request(list(prompt_ids), SamplingParams(temperature=1.0, max_tokens=N, ignore_eos=True))
    outs = []
    while not llm.is_finished():
        out, _ = llm.step()
        outs.extend(out)
    model.compute_logits = orig
    # outs is [(seq_id, token_ids)]; single request
    toks = outs[0][1] if outs else []
    logits = [c[0] for c in cap]        # bs=1 -> each capture is [1, vocab]
    m = min(len(toks), len(logits))
    return toks[:m], logits[:m]


def main():
    Sampler.forward = greedy            # deterministic argmax for both runs
    llm = LLM(MODEL, enforce_eager=True, weight_quant="nvfp4", weight_scale_fp8=True,
              kv_quant_target="", max_model_len=512, max_num_batched_tokens=512, max_num_seqs=1,
              gpu_memory_utilization=0.93)
    tok = llm.tokenizer

    all_pos, prefill_pos, per_prompt = [], [], []
    for i, text in enumerate(PROMPTS):
        pid = tok.apply_chat_template([{"role": "user", "content": text}],
                                      tokenize=False, add_generation_prompt=True)
        ids = tok.encode(pid)
        t16, l16 = run_mode(llm, ids, "")
        t8, l8 = run_mode(llm, ids, "fp8_fake")
        # first divergence d; positions 0..d share identical context in both runs
        L = min(len(t16), len(t8))
        d = next((j for j in range(L) if t16[j] != t8[j]), L - 1)
        metrics = [compare(l16[j], l8[j]) for j in range(d + 1)]
        all_pos.extend(metrics)
        prefill_pos.append(metrics[0])
        diverged = any(t16[j] != t8[j] for j in range(L))
        per_prompt.append(dict(prompt=text[:48], gen_len=L, first_divergence=(d if diverged else None),
                               tokens_match=sum(t16[j] == t8[j] for j in range(L)) / max(L, 1)))
        print(f"[{i}] div@{d if diverged else '-':>3}  match={per_prompt[-1]['tokens_match']:.2f}  "
              f"prefill: top1={metrics[0]['top1']} cos={metrics[0]['cos']:.4f} "
              f"rel_l2={metrics[0]['rel_l2']:.4f} kl={metrics[0]['kl']:.4e}")

    def agg(rows, k):
        return round(statistics.mean(r[k] for r in rows), 6)

    summary = dict(
        model="Qwen3-32B_nvfp4", n_prompts=len(PROMPTS), tokens_per_prompt=N,
        comparable_positions=len(all_pos),
        all_positions=dict(top1_agree=agg(all_pos, "top1"), top5_overlap=agg(all_pos, "top5"),
                           cos=agg(all_pos, "cos"), rel_l2=agg(all_pos, "rel_l2"),
                           kl=agg(all_pos, "kl"), max_abs=agg(all_pos, "max_abs")),
        prefill_only=dict(top1_agree=agg(prefill_pos, "top1"), top5_overlap=agg(prefill_pos, "top5"),
                          cos=agg(prefill_pos, "cos"), rel_l2=agg(prefill_pos, "rel_l2"),
                          kl=agg(prefill_pos, "kl")),
        mean_tokens_match=agg(per_prompt, "tokens_match"),
        n_prompts_no_divergence=sum(1 for p in per_prompt if p["first_divergence"] is None),
    )
    print("\n=== SUMMARY (W4A16 vs W4A8, activation-quant delta) ===")
    print(json.dumps(summary, indent=2))
    json.dump(dict(summary=summary, per_prompt=per_prompt), open("precision_a16_vs_a8.jsonl", "w"), indent=2)

    a = summary["all_positions"]
    md = f"""# W4A16 vs W4A8 activation-quant precision — Qwen3-32B NVFP4 (4090)

Same fp4 (E2M1) weights both runs; bf16 KV both runs. Only difference: activation dtype at every
nvfp4 GEMM input — fp16 (A16) vs per-token fp8_e4m3 (A8). Delta below is purely activation-quant error.
{len(PROMPTS)} prompts, {N} greedy tokens each, {summary['comparable_positions']} comparable positions.

| metric | all positions | prefill only |
|---|---|---|
| top-1 token agreement | {a['top1_agree']:.4f} | {summary['prefill_only']['top1_agree']:.4f} |
| top-5 overlap | {a['top5_overlap']:.4f} | {summary['prefill_only']['top5_overlap']:.4f} |
| logit cosine sim | {a['cos']:.5f} | {summary['prefill_only']['cos']:.5f} |
| logit relative-L2 | {a['rel_l2']:.4f} | {summary['prefill_only']['rel_l2']:.4f} |
| KL(p_A16 || p_A8) | {a['kl']:.4e} | {summary['prefill_only']['kl']:.4e} |
| max |logit diff| | {a['max_abs']:.4f} | — |

Greedy divergence: mean token-match {summary['mean_tokens_match']:.3f} over {N} tokens; \
{summary['n_prompts_no_divergence']}/{len(PROMPTS)} prompts never diverged.
"""
    open("precision_a16_vs_a8.md", "w").write(md)
    print("\nsaved precision_a16_vs_a8.md + .jsonl")


if __name__ == "__main__":
    main()
