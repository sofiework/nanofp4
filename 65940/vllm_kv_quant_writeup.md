# vLLM KV Cache Quantization × Speculative Decoding — 研究结论

> 源码查证: vLLM `0.23.1rc1.dev` (`/home/feng/miniconda3/envs/vllm/.../site-packages/vllm/`)。
> 本文只写结论, 供 nano-vllm KV quant 实现对标 + resume/interview narrative 用。
> 所有 path 相对 vllm package root。

## 1. KV quant 现状 (支持哪些 dtype / granularity)

### 支持的 KV cache dtype
`CacheConfig.cache_dtype` (`config/cache.py:76`, CLI `--kv-cache-dtype`), 取值集 (`config/cache.py:19-36`):

```
auto, float16, bfloat16,
fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla,
int8_per_token_head, int4_per_token_head, fp8_per_token_head,
nvfp4, turboquant_*
```

注意: `fp8` = `fp8_e4m3` 别名; **裸 `int8` 不是合法 KV dtype**, INT8 只能经 `int8_per_token_head` 走。
量化格式基本 map 到 `torch.uint8` (byte-packed), 少数例外 (`fp8_inc`→`float8_e4m3fn`, `int8_per_token_head`→`int8`)。

### 量化模式 (granularity)
中枢 enum `KVQuantMode` (`v1/kv_cache_interface.py:33-73`):

| mode | granularity | scale | backend |
|---|---|---|---|
| `FP8_PER_TENSOR` | 每层 1 个 scalar (K 一个, V 一个) | static (标定/1.0) 或 dynamic 一次算 | FA (Hopper) / FlashInfer / Triton |
| `INT8_PER_TOKEN_HEAD` | 每 `(token, head)` 一个 | **写入时 kernel 动态算** | **仅 Triton** |
| `FP8_PER_TOKEN_HEAD` | 同上 | 同上 | **仅 Triton** |
| `INT4_PER_TOKEN_HEAD` | 同上 + head_dim 内 sub-byte pack | 同上 + Hadamard rotation + asym zero-point | **仅 Triton** |

关键: vLLM 已从「只有 per-tensor fp8」进化到 **per-token-head dynamic int8/int4/fp8**。per-token-head =
每个 head 的 `head_size` 向量共享一个 scale, 写入时按 token 动态标定。这是当前量化 KV 的主力粒度。

### backend gating (决定 Ampere 上能用什么)
- **FlashAttention**: fp8 KV 需 **FA3 + Hopper (SM90)** (`v1/attention/backends/flash_attn.py:165-176`);
  per-head scale 需 FA≥3。→ **Ampere(3060) 上 FA 走不了量化 KV**。
- **per-token-head (int8/int4/fp8)**: 只在 **Triton backend**
  (`v1/attention/backends/triton_attn.py`, store kernel `v1/attention/ops/triton_reshape_and_cache_flash.py`)。
- **结论**: **Ampere 上做量化 KV = 走 Triton path**。这正是本 project 第二步「写 kernel」的 port 对象。

### scale 来源
1. **static (标定)**: checkpoint 里带 `k_scale`/`v_scale` (llm-compressor / ModelOpt 标定),
   `BaseKVCacheMethod.process_weights_after_loading` (`model_executor/layers/quantization/kv_cache.py:74-158`) 加载;
   per-tensor 强制 (`kv_cache.py:128-131` 拒绝非 scalar scale)。
2. **default 1.0**: 无标定时 buffer 出生即 1.0 (`.../attention/attention.py:98-114`), 并 warn。
   这是 uncalibrated fp8 的下界基线。
3. **dynamic**: 老 per-tensor 一次性算 (`calc_kv_scales`, `attention.py:548-556`, `abs(x).max()/range`);
   per-token-head **每次写入**在 kernel 内算, 存进 side scale cache。

## 2. store path (量化在哪发生)

写入统一走 `reshape_and_cache` / `reshape_and_cache_flash` (`_custom_ops.py:2515-2556`), 参数带
`kv_cache_dtype: str`, `k_scale`, `v_scale`。各 backend 从 `layer._k_scale`/`_v_scale` 取 scale 传入。

- **per-tensor** (参考 kernel `triton_reshape_and_cache_flash.py:106-128`): `key/k_scale` 后 cast 到 cache dtype。
  FA 消费时把 scalar scale `.expand((num_seqs, num_kv_heads))` 折进 kernel 当 descale (`flash_attn.py:848-856`)。
- **per-token-head** (`triton_reshape_and_cache_flash.py:154-261`): 每 `(token,head)`:
  ```
  k_scale = max(abs(k_h).max() / QUANT_MAX, 1e-6)   # 动态 absmax 标定
  k_q     = k_h * (1/k_scale)                        # + round/clamp (int)
  ```
  scale 存进 `[num_blocks, block_size, num_kv_heads]` fp32 side cache; `layer._k_scale/_v_scale` 此时强制 1.0。
  **INT4** 转 `v1/attention/ops/int4_per_token_head.py`: Hadamard rotation (RHT) 平滑 outlier + 非对称
  zero-point + 2×int4/byte packing。

## 3. spec × quant: 为什么 draft 与 target 必须同 dtype

**结论: 结构性耦合, 不是显式 assert —— draft 根本没有独立的 dtype 来源。**

1. **`SpeculativeConfig` 没有 `kv_cache_dtype` 字段** (`config/speculative.py`, grep 无匹配)。它有
   per-draft 的 `attention_backend` / `moe_backend` / `quantization` escape hatch, 但**没有 KV dtype**。
   (社区 PR #48787 想加 `SpeculativeConfig.kv_cache_dtype` 作 escape hatch, 未进此 build。)

2. **proposer 复用 target 的同一个 `cache_config` 对象**:
   `_create_draft_vllm_config` (`v1/spec_decode/llm_base_proposer.py:1223`) 只 `replace` 掉
   `kernel_config` 和 `attention_config`, `cache_config` 原样透传。draft model 在这个 config 下 build。

3. **dtype 从这个共享字段流进每个 attention 层**: 每个 `Attention.__init__` (target 或 draft) 读
   `cache_config.cache_dtype` (`model_executor/layers/attention/attention.py:240`) → 得到相同的
   `kv_cache_torch_dtype`。

4. **单池布局强制统一**:
   - runner 全局 `get_kv_cache_spec` (`v1/worker/gpu_model_runner.py:7501-7538`) 扫 target+draft 所有层,
     每层 spec 的 `dtype = kv_cache_torch_dtype`。
   - 层按 spec 分组 (`v1/core/kv_cache_utils.py:1176`), 同组 `merge` 里 **assert dtype 全等**
     (`v1/kv_cache_interface.py:297`)。
   - 整个 KV pool 是**单块 `torch.zeros(size, dtype=int8)` buffer** 切片/别名分配给各层
     (`gpu_model_runner.py:7065-7106`)。
   - proposer 还 assert 所有 draft 层同组 (`llm_base_proposer.py:1609-1630`)。

   → draft 若换 dtype, 其 spec 与 target 不等 → 破坏 group 统一 + 单池布局。vLLM 干脆从源头不给 draft
   独立 dtype 来源 (共享 cache_config), 使「draft 异 dtype」**结构上不可能**, 而非「被禁止」。

5. **draft scale 缺省 1.0**: EAGLE head checkpoint 若无标定 KV scale, `_k_scale=_v_scale=1.0`
   (`attention.py:98-114`, `kv_cache.py:111-115`) —— 即目前 vLLM 里 draft KV 量化是被动继承的, 未被设计过。

6. **EAGLE3 的额外一层**: 就是并进 target KV group 的又一个成员 (`UniformTypeKVCacheSpecs`,
   `v1/kv_cache_interface.py:784-830`; draft 层注册见 `llm_base_proposer.py:1632-1692`), page size 累加,
   继承 target dtype。

## 4. 对 nano-vllm 的启示

- **nano-vllm 同样是单池**: `model_runner.allocate_kv_cache` 一个 `torch.empty(2, num_layers, ...)`,
  target = 层 `0..N-1`, draft (EAGLE3) = 层 `-1`。同样的结构约束: 真换 dtype 要拆单池布局。
- **fake-quant 绕过它**: quant→dequant→存回 bf16, 物理 dtype 全程统一, 量化只是注入误差。→ 可以自由让
  "draft int8/int4 而 target bf16", 测出 vLLM 架构上难以测的 **draft-dtype × accept-rate tradeoff**。
  这是本 project 的独特切口: vLLM 因单池而不给 draft 独立 dtype, 我用 fake-quant 恰好能测这个被架构挡住的问题。
- **accept rate 数值等价**: fake-quant 注入的误差与真低位存储读回的误差相同 → accept rate 结论无损,
  真省显存/加速留给第二步 kernel。
- **第二步 kernel 的 port 对象**: vLLM Triton per-token-head store kernel
  (`triton_reshape_and_cache_flash.py:154-261`) + Triton decode/prefill 消费路径 (`triton_attn.py`)。
  Ampere 上本就走 Triton, 无 FA3 依赖, port 可行。
