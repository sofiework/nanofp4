# CLAUDE.md

## Communication style
- 回答简洁，不说废话 (concise, no useless talk)
- 中英混合 (mix Chinese and English)
- terminology 只用英文 (technical terms in English only)

## Working rules
- 只做用户明确告诉的决策，不要自己决策 how to implement
- 默认不要帮用户改代码，让用户自己改；除非用户明确要求我改
- 用户会持续往后做，我负责记住进度并自动更新下面的 Progress

## Roadmap
1. mixbatch — done
2. priority scheduling — in progress
3. kv cache quantization — in progress (第一步: draft-dtype fake-quant + vLLM 研究)
4. speculative decoding — todo
(单卡 serving; KV offload to CPU 评估过不值得做 0.6B, 跳过)

## mixbatch design decisions (user-decided)
- 维持 prefill/decode kernel 分开 (PD kernel separate, 不融合)
- 利用 cuda graph
- 用 iteration-level scheduling
- run 策略: 两次独立 forward (decode 一次走 cuda graph, prefill 一次走 varlen), 结果 concat。非真 mixbatch, 只是 iteration-level 同时调度

## Progress
- mixbatch: iteration-level 调度, decode/prefill 两次独立 forward (decode 走 cuda graph, prefill 走 varlen), PD kernel 分开 — 完成
- priority scheduling: 基本实现完, 在调 benchmark
  - SamplingParams.priority (normal=0/fast); Sequence 加 priority + arrival_time
  - Scheduler._pick_waiting 按 (started, priority, -arrival_time) 选 seq
  - _priority(): 超 priority_deadline 的 normal 提升到 fast (防 starvation)
  - bench_priority.py + prio.json/prio_ttft_*.png 在跑
- speculative decoding — n-gram 里程碑完成 (2026-07-02), EAGLE-3 进行中:
  - 通用机制全部跑通: verify forward (prepare_verify, varlen 路径) / greedy accept-reject
    (postprocess_spec) / KV rollback (hash_blocks + num_cached_tokens 回退) / scheduler 1+k 调度
  - n-gram 结果 (Qwen3-1.7B, k=4, greedy, 3060Ti): 1.14× speedup, accept_len 1.17, lossless pass
    详见 nanovllm_ngram_benchmark.md + nanovllm_baseline.jsonl
  - 对标 baseline: real vLLM EAGLE3 (AngelSlim/Qwen3-1.7B_eagle3, k=2): 1.52×, accept_len 1.72
    详见 vllm_vanilla_eagle3_benchmark.md + vllm_baseline.jsonl
  - EAGLE-3: 进行中, 换 proposer (ngram -> eagle3 head), 机制复用
    writeup: eagle3_writeup.md (算法/决策+tradeoff); 实现教程: eagle3_impl_guide.md
    决策已全定: k=2 / paged KV +1 层 / 显式改 forward 返回 aux / 独立 Eagle3Proposer (utils/eagle3.py)
    impl 进度 (按 eagle3_impl_guide.md): Step 1 qwen3.py aux 抽取 — done (用户写);
    Step 2 utils/eagle3.py Eagle3Midlayer+Eagle3Proposer — done (Claude 代写, import 过, 未联调);
    Step 1-4 全部完成, 联调 PASS (2026-07-08):
    - 联调修了 2 个 bug: llm_engine propose 块在错误分支 (else 而非 spec 分支); eagle3 下
      capture_cudagraph 崩 (model 返回 tuple) -> ModelRunner 强制 enforce_eager
    - smoke: accept_len 1.65-1.79 (含 chunked prefill 路径); lossless: 与 ngram 同 seed 逐 token
      对比 400 token 仅 1 处分叉, 验证为 bf16 近平局 (top-2 gap=0.25=1 ulp @ 45.5), 非 bug
    Step 5 bench done (2026-07-08): 1.34× (31.6->42.4 tok/s), accept_len 1.70, lossless pass
    accept_len 与 real-vLLM EAGLE3 打平 (1.70 vs 1.71, 同 k=2), speedup 差距 (1.34 vs 1.52)
    来自 engine 层 (draft loop 每步 CPU sync / 无 batching / 全 eager)
    详见 nanovllm_eagle3_benchmark.md + nanovllm_eagle3.jsonl — EAGLE-3 里程碑完成
- kv cache quantization + NVFP4 weights — real kernels DONE, 本地验证过 (2026-07-22):
  - fake-quant 阶段 (Step 0/0b) 已完成: draft KV quant 免费 (accept 不变), 只 target KV 掉 accept
    (fp8 -1.6%, fp4 -6.6%); NVFP4 weights -5%; 全 fp4 stack accept~1.6. 详见 fp4_spec_4090_plan.md
  - 目标转向: 单卡 4090 跑 ~30B (Qwen3-32B) = nvfp4 weights + fp4/fp8 KV + EAGLE-3。参考 AdaLLM
    (BenChaliah/NVFP4-on-4090-vLLM), port 其 fp16 kernel-mode dequant GEMM
  - 决策 (user): weight-only NVFP4 dequant (packed 常驻 + GEMM 内 dequant 到 fp16, 无 fp4 TC) /
    KV 读路径 full (decode + varlen 都用真 quant kernel, 不 dequant-to-temp)
  - 实现 (Claude 代写, 全部本地验证 on 3060Ti):
    - 权重: layers/nvfp4_gemm.py (fused GEMM + quantize/dequant) + quant.py + linear.py kernel-mode
      (uint8 weight + fp16 per-group scale, global scale_2 折进 per-group 让 merged qkv/gate_up 正确)
      + loader.py (读 AngelSlim key .weight/.weight_scale/.weight_scale_2; 也支持 bf16 ckpt 现场打包)
    - KV: layers/paged_attn.py (torch quant-store 对齐 fake_quant_kv + 统一 dequant-in-loop paged
      read kernel 覆盖 decode/verify/prefill) + attention.py 路由 + model_runner 分配 quant pool+scale
    - 验证: GEMM vs dequant-matmul 3e-4; paged kernel bf16 exact 2e-7 / fp4 5e-4; 1.7B 全栈
      (nvfp4+fp4KV+eagle3) coherent; non-quant byte-exact 无 regression
  - 4090-only 残留 (sm_86 做不了): 下载 packed nvfp4 ckpt / fp8-read kernel 只 sm_89 编译得过 /
    perf tuning / output-quality eval。详见 fp4_spec_4090_plan.md "Residual". bench_fp4_4090.py 已指向
    nvfp4 ckpt, 拷到 4090 即可跑
  - VRAM fit (2026-07-23, 4090 实测): 32B weights 21.09G (packed 15.6 + fp16 scale 3.9 + tied embed 1.5)
    几乎占满 24G 90% 线 -> num_kvcache_blocks≤0 assert 挂; eagle3 head 装不下。两处修复 (Claude 代写,
    sm_86 验过, byte-exact 无 regression):
    (1) max_num_batched_tokens 默认 16384 -> warmup 16384-token prefill OOM; bench 改 2048
    (2) fp8_e4m3 weight scale (weight_scale_fp8=True): per-group scale fp16->fp8 (ckpt 原生 dtype, 无损),
        省 ~2G, global scale_2 拆成 per-output 向量在 GEMM 里乘 (处理 merged qkv/gate_up)。weights->19.1G,
        eagle3 装得下。AdaLLM 证实这是原生格式 (它反而 upcast 到 fp16 浪费), 且 AdaLLM 无 dense-32B 例证
        (天花板 Gemma3-27B@19.83G)。4090-only 残留: Triton fp8 e4m3 GEMM load (sm_89)
  - fp8 GEMM scale load (4090, user 写 7cd8bf9): Triton 不能直接 .to(fp16) 加载 fp8_e4m3 tensor ->
    需 S_ptr.to(pointer_type(uint8)) + s.to(float8e4nv, bitcast=True).to(fp16) (sm_89 only)。Claude 后续
    抽成 _load_scale(SCALE_FP8 constexpr): fp8 走 bitcast, fp16 直接 load (恢复 sm_86 本地验证 + weight_scale_fp8=False)
  - 32B 首跑 (4090): fp8 stack 跑通但 **very slow** no_spec 1.7 tok/s / itl 589ms / ttft ~990ms。诊断:
    decode 是 dequant GEMV, M=1 时 plain kernel grid 只 N/BLOCK_N 个 program (SM 大量闲置) + LUT gather +
    tl.dot 在 M=1 浪费。3060Ti microbench 显示 nvfp4 读的数据比 fp16 少 4x 却慢 7x -> **compute/latency bound
    非纯 BW bound**。Claude 加 split-K GEMV kernel (M<=64, atomic-add fp32, env NVFP4_SPLIT_K/GEMV_MAX_M 可调,
    SPLIT_K=1 或 GEMV_MAX_M=0 可关) + bench_nvfp4_gemm.py。correctness sm_86 验过 (rel<5e-4, 全栈 coherent);
    实际加速需 user 在 4090 measure + sweep SPLIT_K。可能还需 GEMV-shaped kernel (去掉 tl.dot 浪费) 或 fp8-TC prefill
  - decode perf 迭代 (4090 measured):
    (1) split-K GEMV: decode BW 32->120 GB/s, SPLIT_K=8 最优
    (2) arithmetic E2M1 decode (去 LUT gather): decode ratio 降到 ~1.0x vs fp16 (parity, GEMM 解决);
        no_spec 1.7->5.4 tok/s。但 ITL 183ms 里只 ~66ms 是 GEMM, ~117ms 是 eager overhead
    (3) CUDA graph for nvfp4+fp4-KV decode (Claude 代写): 需 (a) 把 fp4 KV store 从 boolean-mask torch
        改成 fixed-grid triton kernel (skip -1 in-kernel, 无 host sync -> capturable);
        (b) model_runner enforce_eager 放开 (nvfp4+fp4KV 可 capture; 只 eagle3 tuple + fp8KV 仍 eager);
        (c) bench enforce_eager=False + max_num_seqs=8。坑: store kernel 假设 head_dim contiguous, 但
        decode k/v 是 qkv-split view -> 必须 .contiguous()。1.7B 本地验证: graph 19->32.9 tok/s = 1.73x,
        coherent, 全 correctness test pass。fp8-KV store 仍 boolean-mask torch (不 capturable, fp8 走 eager)

## priority scheduling design (user-decided)
- scenario: 交互式 fast vs 批处理 normal, 用户可选 fast/normal mode (tiered)
- 防止 normal starvation: deadline 保底 (priority_deadline, config, 超时 normal->fast)

## speculative decoding design (user-decided)
- 独立 draft model 方案对 0.6B 无效: Qwen3 最小就是 0.6B, 更小 distill 没有可用的;
  draft 必须比 target 便宜一个数量级才划算, 0.5B vs 0.6B 净亏 -> 跳过
- 选 n-gram (prompt-lookup) + EAGLE 都做, 两者 draft 成本趋近 0
- 通用机制共享 (verify forward / rejection sampling / KV rollback / scheduler),
  只有 proposer 不同 -> 先做 n-gram 建好机制, EAGLE 再换 proposer
- verify forward 复用现有 varlen prefill 路径 (attention.py block_tables 分支, q_len=γ+1)
- 坑: (1) verify 变长先走 eager, 别塞 cuda graph; (2) reject 的 draft token 要 KV rollback
- rejection sampling 保证输出分布 = target 单独 decode (无损)
- 代码量估计: n-gram 专属 ~30 LOC, 跑通总共 ~200-250 LOC; EAGLE 增量 ~50 LOC + head 权重

## mixbatch benchmark 结论 (3060Ti + Qwen3-0.6B, bench_mixedbatch.py)
- online (staggered arrival) 才看得出 mixbatch 优势; offline 一次性灌入两者打平
- 较好配置: chunk(max_num_batched_tokens)=2048 + rate=20 → ITL p99 -53% (227→106ms), TTFT/throughput 不亏
- chunk=1024 / rate=50 略差 (rate 太高饱和退化, 趋近 offline)

