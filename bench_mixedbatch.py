"""
Benchmark: mixed batching ON vs OFF.  一条命令跑完:

    python3 bench_mixedbatch.py            # 默认 reps=5
    python3 bench_mixedbatch.py --reps 3

A=off, B=on, 只差 enable_mixbatch 开关, 其它全同 (同模型/卡/参数/输入)。
两个 mode 各自在独立子进程跑 (同进程不能 dist init 两次 + 显存各抢 0.9),
跑完自动出指标 + 图 (itl_cdf.png / ttft_cdf.png / throughput.png)。

建议先锁 GPU 频率降噪:  sudo nvidia-smi -lgc <freq>
"""
import os
import json
import time
import argparse
from random import randint, seed, shuffle, Random

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
MAX_TOKENS = 128          # 固定输出长度 -> 每条 decode 工作量一样, 差异只来自调度 (ITL 降噪关键)
# 丰富 prompt 长度分布: 短/中/长三档, 模拟真实流量
BUCKETS = [(200, (40, 120)),     # 短
           (200, (200, 700)),    # 中
           (100, (1200, 2000))]  # 长


def gen_prompts():
    seed(0)
    prompts = []
    for count, (lo, hi) in BUCKETS:
        prompts += [[randint(0, 10000) for _ in range(randint(lo, hi))] for _ in range(count)]
    shuffle(prompts)      # 长短交错 -> 触发 mixed batching
    return prompts


def gen_arrivals(n: int, rate: float, seed_val: int = 1):
    """Poisson 到达: 返回每条请求相对 t0 的到达偏移(秒). rate<=0 -> 全部 t=0 (一次性灌入)."""
    if rate <= 0:
        return [0.0] * n
    rng = Random(seed_val)
    t, arr = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)    # inter-arrival ~ Exp(rate)
        arr.append(t)
    return arr


def _run_once(llm, prompts, sp, arrivals):
    """按到达时间表在线提交 + step, 返回一遍的计时数据. arrivals 全 0 = 一次性灌入."""
    llm.collect_metrics = True
    llm.reset_metrics()
    n = len(prompts)
    idx = 0
    t0 = time.perf_counter()
    while idx < n or not llm.is_finished():
        now = time.perf_counter() - t0
        while idx < n and arrivals[idx] <= now:       # 到点的请求入队
            llm.add_request(prompts[idx], sp[idx])
            idx += 1
        if llm.is_finished():                          # 当前没活, 睡到下一个到达
            if idx < n:
                time.sleep(max(0.0, arrivals[idx] - (time.perf_counter() - t0)))
            continue
        llm.step()
    wall = time.perf_counter() - t0
    ids = sorted(llm.token_times.keys())               # = 提交顺序, A/B 逐条对齐
    return {
        "wall": wall,
        "submit": [llm.submit_time[i] for i in ids],
        "token_times": [llm.token_times[i] for i in ids],
    }


def _worker(mixbatch: bool, prompts, reps: int, out_path: str, chunk: int, arrivals):
    """在独立进程里跑一个 mode, 结果写 out_path"""
    from nanovllm import LLM, SamplingParams
    sp = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=MAX_TOKENS) for _ in prompts]
    llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=4096,
              enable_mixbatch=mixbatch, max_num_batched_tokens=chunk)

    llm.collect_metrics = False
    llm.generate(prompts, sp, use_tqdm=False)          # 热身一遍丢弃

    reps_data = []
    for r in range(reps):
        rep = _run_once(llm, prompts, sp, arrivals)
        reps_data.append(rep)
        print(f"  [mixbatch={mixbatch}] rep {r}: wall {rep['wall']:.2f}s")
    json.dump({"mixbatch": mixbatch, "max_tokens": MAX_TOKENS, "reps": reps_data}, open(out_path, "w"))


def _rep_metrics(rep):
    # per_req_p99 = 每条请求自己的 p99 ITL (它最坏的那几个 token 间隔), 用于配对比较尾部改善
    ttfts, all_itls, per_req_p99 = [], [], []
    for sub, tt in zip(rep["submit"], rep["token_times"]):
        if not tt:
            continue
        ttfts.append(tt[0] - sub)
        itls = [b - a for a, b in zip(tt, tt[1:])]
        all_itls.extend(itls)
        per_req_p99.append(_pct(itls, 99) if itls else 0.0)
    return ttfts, all_itls, per_req_p99


def _pct(xs, q):
    """numpy-style linear-interpolation percentile, pure python"""
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if not xs:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def compare(file_off: str, file_on: str):
    data = {}
    for tag, fn in [("baseline", file_off), ("mixed-batch", file_on)]:
        d = json.load(open(fn))
        ttfts, itls, walls, per_req_per_rep = [], [], [], []
        for rep in d["reps"]:
            t, i, p = _rep_metrics(rep)
            ttfts += t; itls += i; per_req_per_rep.append(p)
            walls.append(rep["wall"])
        n_req = len(per_req_per_rep[0])
        per_req_p99 = [_mean([rep[j] for rep in per_req_per_rep]) * 1000 for j in range(n_req)]  # 每条请求 p99 ITL, 跨 rep 平均
        total_out = len(d["reps"][0]["submit"]) * d["max_tokens"]
        thr = [total_out / w for w in walls]
        data[tag] = dict(ttft=[x * 1000 for x in ttfts], itl=[x * 1000 for x in itls],
                         per_req_p99=per_req_p99, thr=thr)

    A, B = data["baseline"], data["mixed-batch"]

    print("\n=== ITL (ms) ===")
    print(f"  p50: {_pct(A['itl'],50):.3f} -> {_pct(B['itl'],50):.3f}")
    print(f"  p99: {_pct(A['itl'],99):.3f} -> {_pct(B['itl'],99):.3f}  ({(_pct(B['itl'],99)/_pct(A['itl'],99)-1)*100:+.1f}%)")
    print("=== TTFT (ms) ===")
    print(f"  p50: {_pct(A['ttft'],50):.3f} -> {_pct(B['ttft'],50):.3f}")
    print(f"  p99: {_pct(A['ttft'],99):.3f} -> {_pct(B['ttft'],99):.3f}  ({(_pct(B['ttft'],99)/_pct(A['ttft'],99)-1)*100:+.1f}%)")
    print("=== Throughput (tok/s) ===")
    print(f"  {_mean(A['thr']):.1f} +/- {_std(A['thr']):.1f} -> {_mean(B['thr']):.1f} +/- {_std(B['thr']):.1f}")

    # 配对比较: 同一条请求, 它自己的 p99 ITL, baseline - mixbatch. >0 = mixbatch 尾部更好
    diffs = [a - b for a, b in zip(A["per_req_p99"], B["per_req_p99"])]
    print("=== paired per-request p99 ITL improvement (ms) ===")
    print(f"  mean {_mean(diffs):+.3f}   90% interval [{_pct(diffs,5):.3f}, {_pct(diffs,95):.3f}]")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -> skip plots; metrics above are complete)")
        return

    # ITL CDF: 极端尾巴会把有效区压扁 -> 局部放大到 p99 附近, y 轴聚焦 tail
    itl_cut = 1.2 * max(_pct(A["itl"], 99), _pct(B["itl"], 99))
    plt.figure()
    for tag in ("baseline", "mixed-batch"):
        x = sorted(data[tag]["itl"])
        y = [i / len(x) for i in range(len(x))]
        plt.plot(x, y, label=tag)
    plt.xlim(0, itl_cut); plt.ylim(0.8, 1.0)         # 局部: 只看 tail 区
    plt.xlabel("ITL (ms)"); plt.ylabel("CDF"); plt.legend(); plt.title("ITL CDF (tail zoom)")
    plt.grid(alpha=0.3); plt.savefig("itl_cdf.png", dpi=120); plt.close()

    # TTFT CDF: 全量即可
    plt.figure()
    for tag in ("baseline", "mixed-batch"):
        x = sorted(data[tag]["ttft"])
        y = [i / len(x) for i in range(len(x))]
        plt.plot(x, y, label=tag)
    plt.xlabel("TTFT (ms)"); plt.ylabel("CDF"); plt.legend(); plt.title("TTFT CDF")
    plt.grid(alpha=0.3); plt.savefig("ttft_cdf.png", dpi=120); plt.close()

    # Throughput: y 轴从 0 看不出差 -> 局部放大到数值附近
    tags = ["baseline", "mixed-batch"]
    means = [_mean(data[t]["thr"]) for t in tags]
    errs = [_std(data[t]["thr"]) for t in tags]
    plt.figure()
    bars = plt.bar(tags, means, yerr=errs, capsize=6)
    lo = min(means) - max(errs) * 3 - (max(means) - min(means)) - 5
    hi = max(means) + max(errs) * 3 + (max(means) - min(means)) + 5
    plt.ylim(lo, hi)                                  # 局部: 聚焦数值差
    for b, m in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2, m, f"{m:.1f}", ha="center", va="bottom")
    plt.ylabel("Throughput (tok/s)"); plt.title("Throughput (zoom)")
    plt.savefig("throughput.png", dpi=120); plt.close()
    print("\nsaved: itl_cdf.png, ttft_cdf.png, throughput.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=16384, help="max_num_batched_tokens (prefill chunk); 调小压 ITL p99")
    ap.add_argument("--rate", type=float, default=0.0, help="到达速率 req/s (Poisson). 0=一次性全灌入(offline)")
    ap.add_argument("--compare-only", action="store_true", help="reuse off.json/on.json, skip running")
    args = ap.parse_args()

    if args.compare_only:
        compare("off.json", "on.json")
        return

    print("[reminder] lock GPU clock for stable numbers: sudo nvidia-smi -lgc <freq>")
    import torch.multiprocessing as mp
    prompts = gen_prompts()
    arrivals = gen_arrivals(len(prompts), args.rate)      # A/B 共用同一到达时间表
    ctx = mp.get_context("spawn")
    print(f"max_num_batched_tokens (chunk) = {args.chunk}, rate = {args.rate} req/s "
          f"({'offline 一次性' if args.rate <= 0 else 'staggered arrival'})")
    for mixbatch, out in [(False, "off.json"), (True, "on.json")]:
        p = ctx.Process(target=_worker, args=(mixbatch, prompts, args.reps, out, args.chunk, arrivals))
        p.start(); p.join()
        assert p.exitcode == 0, f"worker (mixbatch={mixbatch}) failed"
    compare("off.json", "on.json")


if __name__ == "__main__":
    main()
