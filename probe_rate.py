"""Calibration probe: offline drain rate (req/s) for the bench prompt set on this GPU."""
import time
from bench_priority import gen_prompts, MODEL_PATH, MAX_TOKENS


def main():
    from nanovllm import LLM, SamplingParams
    prompts = gen_prompts()
    sp = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=MAX_TOKENS) for _ in prompts]
    llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=4096,
              max_num_batched_tokens=2048, max_num_seqs=64)
    # warm-up
    llm.generate(prompts[:16], sp[:16], use_tqdm=False)
    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)   # all arrive at once -> max drain
    wall = time.perf_counter() - t0
    n = len(prompts)
    print(f"prompts={n} wall={wall:.1f}s  drain_rate={n/wall:.2f} req/s  "
          f"tok/s={n*MAX_TOKENS/wall:.0f}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.get_context("spawn").Process(target=main).start()
