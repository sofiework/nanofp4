import os, torch
from nanovllm import LLM, SamplingParams
M = os.path.expanduser("~/huggingface/Qwen3-32B_nvfp4/")
llm = LLM(M, enforce_eager=False, max_model_len=4096, max_num_batched_tokens=256, max_num_seqs=8,
          gpu_memory_utilization=0.90, weight_quant="nvfp4", weight_scale_fp8=True,
          kv_quant_target="fp4", kv_scale_fp8=True)
tok = llm.tokenizer
for q in ["Explain how a transformer attention mechanism works, step by step.",
          "Write a Python function to check if a string is a palindrome, with a test."]:
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    torch.manual_seed(0)
    o = llm.generate([p], SamplingParams(temperature=0.6, max_tokens=100), use_tqdm=False)
    print("\n### Q:", q, "\n>>>", o[0]["text"][:400].replace("\n", " "))
