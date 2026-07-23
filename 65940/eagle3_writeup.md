# EAGLE-3 on nano-vllm — Writeup

> Status: **in progress** (started 2026-07-07)。此文档随 code/决策更新维护, project 做完时是完整 writeup。

## 0. 目标

在 nano-vllm (~1200 行引擎) 里 from-scratch 实现 EAGLE-3 speculative decoding。
spec decoding 通用机制 (verify forward / accept-reject / KV rollback / 1+k 调度) 已由
n-gram 里程碑跑通 ([nanovllm_ngram_benchmark.md](nanovllm_ngram_benchmark.md), 1.14×, accept_len 1.17),
EAGLE-3 只换 proposer。

对标 baseline (同卡同 target, real vLLM + AngelSlim head, k=2):
**1.52× speedup, accept_len 1.72** ([vllm_vanilla_eagle3_benchmark.md](vllm_vanilla_eagle3_benchmark.md))。

- target: Qwen3-1.7B bf16, 3060Ti 8GB, bs=1, greedy
- head: `AngelSlim/Qwen3-1.7B_eagle3`, 本地 `~/huggingface/Qwen3-1.7B_eagle3/` (已确认)

## 1. Head 结构 (state_dict 已核实)

| 组件 | 权重 shape | 说明 |
|---|---|---|
| `fc` | (2048, 6144) | target 3 层 aux hidden concat (3×2048) → 2048, **只在每轮 draft 第一步用** |
| `midlayer` | 1 层 llama-style decoder | attention 输入 4096 = concat(embed(token), feature); q(2048,4096) k/v(1024,4096) o(2048,2048); GQA 16/8, head_dim 128; **无 q_norm/k_norm** (与 Qwen3Attention 不同) |
| `midlayer.input_layernorm` / `hidden_norm` | (2048,) ×2 | 分别 norm embed 流和 feature 流, 再 concat |
| `norm` + `lm_head` | (2048,), (32000, 2048) | draft vocab 32000 |
| `d2t` | (32000,) int64 | draft id → target id 的 offset (target vocab 151936) |
| `t2d` | (151936,) bool | target id 是否在 draft vocab (非 greedy 时用) |

head 无 embed_tokens, 共享 target 的 embedding。权重 fp16 (target 是 bf16, load 时注意 dtype)。

## 2. 算法 (对照 vLLM `llm_base_proposer.py` + `llama_eagle3.py` 查证)

### 每轮 propose = 1 次 window pass + (k-1) 次 single-token pass

关键点: draft 每轮先把 target 这轮 forward 过的 **整个 window** (prefill 轮 = 整个
prompt; decode 轮 = 1+m 个 verify token) 过一遍 midlayer —— 这一步建立/刷新 draft
自己的 KV cache, 顺便出第 1 个 draft token; 之后才是 k-1 次 single-token 自回归。
没有这个 window pass, draft attention 就没有 prefix 的 KV 可看。

**Window pass (每轮 1 次, n = 这轮 target 处理的 token 数):**
1. target forward 抽 3 层 aux hidden `[n, 2048]×3` → concat `[n, 6144]` → `fc` → feature `[n, 2048]`
   (fc 作用于整个 window 每个位置, 不是只切最后一个)
2. draft input token = target token **左移一位**: 位置 i 的输入 pair =
   (embed(token[i+1]), feature[i]); window 最后一个位置放 target 刚 accept/sample 出的 next token。
   即 draft 学的是 "看 (下一个 token, 当前 feature) 预测下下个 token"
3. positions 直接用 target 的 position 原值 (不从 0 重排)
4. 整个 window 过 midlayer (写 draft KV) → 取最后 accepted 位置的输出 → norm → lm_head → **draft_token_1**

**Single-token pass (循环 k-1 次):**

5. input pair = (embed(上一个 draft token), 上一步 midlayer 输出的 **pre-norm** hidden `[2048]`), position+1
6. midlayer forward (attend 到全部 draft KV) → hidden → norm → lm_head → 下一个 draft token

feature 每步都在更新 (双流 (token, feature) 自回归); fc 只处理 target 来的 6144,
draft 自己的 hidden 本来就是 2048 不过 fc。

**Draft KV rollback 是隐式的**: 被 reject 位置的 draft KV, 下一轮 window pass 会用
target-derived feature 重写同一批 slot, 不需要显式 rollback (seq_len 记账要对上)。

### midlayer 内部 (AngelSlim 权重走的 default 路径)

- embed 流过 `input_layernorm`, feature 流过 `hidden_norm`; **residual = pre-norm feature**
- concat 顺序 **embed 在前**: `cat([embed_normed, feature_normed])` → `[*, 4096]`
  (qkv 权重布局依赖这个顺序)
- qkv_proj: q→2048 (16 heads), k/v→各 1024 (8 kv heads, GQA), head_dim 128;
  rope theta 1e6, **无 qk-norm**
- attn → o_proj `[2048]` → +residual → post_attention_layernorm → mlp (silu, 6144) → +residual
  → hidden `[2048]` (pre-norm, 即下一步的 feature)
- `norm(hidden)` → lm_head → logits `[32000]` → argmax = draft_id →
  **target_id = draft_id + d2t[draft_id]**; embed 用 target vocab id (embedding 共享 target 的)
- vLLM 里的 norm_before_fc / fc_norm / norm_output / norm_before_residual 变体对
  AngelSlim 这份权重全是 off, 不用管

### 与现有机制的衔接 (每个 decode step)

```
verify forward (target, 1+k tokens, varlen 路径)   ← 现有, 需加: 返回 aux hidden
  → accept-reject (scheduler.postprocess_spec)      ← 现有, 不变
  → propose (eagle3 head, k 次 draft forward)       ← 新增, 产出下轮 draft_tokens
  → 下一轮 verify
```

prefill 同理: prefill forward 也要返回 aux hidden (最后一个 prompt token 的), 供第一次 propose 用。

## 3. 引擎需要新增

1. **target forward 捕获 aux hidden states** — `Qwen3Model.forward` (nano-vllm 自己的实现,
   `nanovllm/models/qwen3.py`, 非官方代码, 随便改) 目前只返回最后 hidden; prefill 和 verify 路径都要抽。
   **layer ids 已查证** (vLLM 默认公式 `(2, n//2, n-3)`, Qwen3-1.7B n=28 → `(2, 14, 25)`):
   语义是 "index 0 = embedding 输出, index i = 第 i 层跑完后的 hidden+residual (pre-norm)",
   即取 `self.layers[1]`、`self.layers[13]`、`self.layers[24]` 跑完后的 `hidden_states + residual`。
2. **eagle3 head module + 权重加载** — fc / midlayer / norm / lm_head / d2t; midlayer 是
   llama-style attention (无 qk-norm), 不能直接复用 Qwen3Attention。
3. **draft 自回归 loop** — 替换 `llm_engine.py` 里的 ngram `propose_draft` 调用点;
   `draft_forward_count` 不再是 0 (= target_fwd × k)。
4. **draft head 自己的 KV cache** (单层) — 与 target 序列位置对齐; reject 后无需显式
   rollback (下一轮 window pass 重写同批 slot, 见 §2)。
5. **propose 的接口位置** — eagle3 draft 需要 GPU forward + 上一轮 verify 的 hidden, 不能像
   ngram 一样做成 utils 里的纯 CPU 函数; 天然在 model_runner 一侧 (具体放哪待用户决策, 见下)。

## 4. 决策点

| # | 决策 | 状态 |
|---|---|---|
| 1 | draft head KV cache 怎么管 | **decided: (a)** 复用 paged KV 加一层; block_manager 本身零改动 (logical block layer-agnostic), 改的是 `model_runner.allocate_kv_cache` |
| 2 | aux hidden states 怎么取 | **decided: 显式改 `Qwen3Model.forward`** 返回 aux |
| 3 | k 取多少 | **decided: k=2**, 对齐 AngelSlim 官方表和 vLLM baseline; eagle3 的 draft 有真实 GPU 成本, 不像 ngram 免费 |
| 4 | propose 代码放哪 | **decided: 独立 `Eagle3Proposer` module** (`nanovllm/utils/eagle3.py`), model_runner 持有并调用 |
| 5 | aux layer ids / position / token shift 等参考实现细节 | **resolved**: 查证 vLLM 源码, 结论在 §2/§3 |

### 方案对比

**#1 Draft KV cache: (a) 复用 paged KV, 加"第 29 层" vs (b) bs=1 独立 buffer**

| | (a) 复用 block_manager/paged cache | (b) 独立 contiguous buffer |
|---|---|---|
| 思路 | kv_cache 多分配一层, draft 用 target **同一份 block_table/slot_mapping** (位置 1:1 对齐) | 单独 `[max_model_len, 8, 128]` tensor, 自己记 seq_len |
| 代码 | 很少: window pass 的 slot_mapping 在 `prepare_verify` 已算好, draft 原样用; attention 走现有 varlen+block_tables 路径 | 要写第二套 append/attend 逻辑 (contiguous 不走 paged kernel), 手动记账 |
| 内存 | +1/28 ≈ +3.6% KV, 8GB 无压力 | 最省 |
| 后续 | bs>1 / prefix cache 天然兼容 | 锁死 bs=1 |

核心论据: draft KV 与 target 位置完全对齐, target 已有的全部 metadata (block_table /
slot_mapping / rollback 后的 num_cached_tokens) 可以白拿; (b) 反而要重新发明记账。→ 推荐 (a)。

**#2 aux hidden 取法: 显式改 `Qwen3Model.forward` vs forward hook**

- 显式: loop 里加 `if i in self.aux_layers: aux.append(hidden_states + residual)`,
  qwen3.py ~6-10 行 + model_runner 接返回值 ~5 行; `aux_layers` 为空时行为不变
- hook: 不动签名, 但对 1200 行的教学引擎属于过度设计, 且 residual 在 layer 返回值里,
  hook 拿 pre-norm hidden+residual 反而绕
- spec 模式下 prefill/verify 全走 eager, 不碰 cuda graph capture, 无兼容问题 → 推荐显式

**#4 proposer 位置: 独立 module vs inline 进 model_runner**

- 独立 `Eagle3Proposer` (nn.Module, 放 `spec_decoding/eagle3.py`): 持有 fc/midlayer/norm/
  lm_head/d2t + target embed 引用; 接口 `propose(window_token_ids, window_features,
  positions, k) -> draft_tokens`; model_runner 持有实例, verify/prefill 后调用
- 好处: 与 ngram 并排, 呼应"机制不动只换 proposer"; model_runner diff 小; 权重加载不污染 runner
- inline 唯一好处是少一层间接 → 推荐独立 module
- 注意: midlayer 不能直接复用 `Qwen3Attention` (无 qk-norm, qkv 输入 4096), 需在 eagle3.py
  写小的 llama-style layer, 复用 `layers/` 的 linear/rope/attention 原语

已定 (来自 CLAUDE.md spec-decoding design):
- verify 复用 varlen prefill 路径, 变长走 eager 不进 cuda graph
- greedy verify = argmax, lossless by construction
- benchmark 复用 bench_ngram_nanovllm.py 的 schema/prompt, 与 vllm_baseline.jsonl 真比 accept_len
