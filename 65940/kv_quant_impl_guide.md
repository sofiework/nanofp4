# KV Cache Quantization Implementation Guide (nano-vllm)

> 第一步: **fake-quant** 测 draft KV quant → 不同 dtype 的 tradeoff 与对 accept rate 的影响。
> 本轮 v1 只做 **int8**, granularity = **per-token-head** (对齐 vLLM, 之后 kernel 可 1:1 port)。
> vLLM 设计对标见 [vllm_kv_quant_writeup.md](vllm_kv_quant_writeup.md)。
>
> 体例: 新 class/func 只给 **signature + doc-comment**, body 你写; 要改的现有代码 **copy 原 block**,
> 在改动处用 `# >>>` 注释标出。全文按 tp=1 (bench 场景)。

## 0. 为什么 fake-quant 够用 (框架)

`store_kvcache` 前对 k、v 做 `quant→dequant`, 存回 bf16。物理 dtype 不变 → 不碰单池布局 (nano-vllm 的
`kv_cache = torch.empty(2, num_layers, ...)` 是一整块, draft = 层 `-1`; 与 vLLM 一样, 真换 dtype 要拆单池)。
量化误差被注入进被 attention 读回的 k、v —— **accept rate 与真低位存储数值等价**, 无需 kernel。
真省显存/加速留给第二步 (port vLLM Triton per-token-head kernel)。

**数据流** (一个 attention 层内):
```
q,k,v (bf16) ── forward ──> [if self.kv_quant]
                              k = fake_quant_dequant(k, cfg.k)   # 注入误差
                              v = fake_quant_dequant(v, cfg.v)
                            └─> store_kvcache(k,v,...)   写进 cache (被后续 decode 读)
                            └─> 本次 attention 也用这份 dequantized k,v (存-读一致)
```
默认 `self.kv_quant = None` → 完全不进 hook, 行为与现在逐位一致 (回归验证的基础)。

**target / draft 独立**: 两侧是不同的 `Attention` 实例 (draft = `proposer.midlayer.attn`, 挂在
`kv_cache[*,-1]`)。在 `allocate_kv_cache` 里分别挂 config 即可独立控制。v1: target = None(bf16),
draft = int8。机制本身对称, 之后可任意组合 (含 target 量化)。

**per-token-head 量化算法** (对齐 vLLM, 复用 65940 lab2 linear quant):
- k/v 在 hook 里 shape = `[N, H, D]` (N=token, H=num_kv_heads, D=head_dim)。
- per-token-head: 每 `(token, head)` 一个 scale → scale shape `[N, H, 1]`, 沿 D reduce。
- int8 symmetric: `qmax=127`; `S = absmax_over_D / qmax`; `q = round(x/S)` clamp 到 `[-128,127]`;
  dequant `x' = q * S`。与 vLLM per-token-head kernel 的 `absmax/QUANT_MAX` 完全一致。
- **坑**: lab2 `linear_quantize` 断言 fp32, k/v 是 bf16 → hook 内先 `.float()`, 算完 cast 回原 dtype。
- decode 单 token (N=1) 也成立 (scale 沿 D 算, 与 N 无关) → 增量安全, 无需 recency buffer。

---

## Step 1 — 新文件 `nanovllm/utils/kv_quant.py`

**加什么**: 两个 config dataclass + 一个纯函数 `fake_quant_dequant`。**只给 signature + comment, body 你写。**

```python
import torch
from dataclasses import dataclass


def get_quantized_range(bitwidth: int) -> tuple[int, int]:
    """复用 65940/lab2.md: 返回 (qmin, qmax)。int8 -> (-128, 127)。
    对称量化只用 qmax。"""
    # YOUR CODE: qmax = (1 << (bitwidth-1)) - 1 ; qmin = -(1 << (bitwidth-1))
    ...


@dataclass
class TensorQuantConfig:
    """单个张量 (K 或 V) 的量化配置。
    bits:        v1 = 8; 之后 4 / fp8。
    granularity: v1 = "per_token_head" (scale 每 (token,head) 一个, 沿 head_dim reduce);
                 预留 "per_tensor"。
    symmetric:   v1 = True (Z=0, S=absmax/qmax)。
    group_size:  int4 时在 head_dim 内再分组用; int8/0 = 整个 head_dim 一组, 忽略。
    """
    bits: int
    granularity: str = "per_token_head"
    symmetric: bool = True
    group_size: int = 0


@dataclass
class KVQuantConfig:
    """一侧 (target 或 draft) 的 KV 量化配置, K/V 分开配 (KIVI 式)。"""
    k: TensorQuantConfig
    v: TensorQuantConfig

    @staticmethod
    def from_str(spec: str) -> "KVQuantConfig | None":
        """把 config 字符串 (如 "int8") 解析成 KVQuantConfig; "" -> None (关闭)。
        v1 只认 "int8" -> K、V 都 TensorQuantConfig(bits=8, per_token_head, symmetric)。
        之后扩展 "int4"/"fp8"/K、V 不同粒度。"""
        # YOUR CODE
        ...


@torch.no_grad()
def fake_quant_dequant(x: torch.Tensor, cfg: TensorQuantConfig) -> torch.Tensor:
    """对 x = [N, H, D] (bf16) 做 quant->dequant, 返回同 shape 同 dtype 的有损张量。
    per_token_head + symmetric int8 步骤:
      1) xf = x.float()
      2) absmax = xf.abs().amax(dim=-1, keepdim=True)         # [N, H, 1]
      3) qmax = get_quantized_range(cfg.bits)[1]               # 127
         scale = (absmax / qmax).clamp_min(1e-6)               # 防 0 除
      4) q = (xf / scale).round().clamp(-qmax-1, qmax)         # int8 网格上的值
      5) deq = q * scale                                       # 回到 fp
      6) return deq.to(x.dtype)                                # cast 回 bf16
    注意: 全程不真的存 int8, 只是把 x 投影到 int8 可表示的格点上 (fake quant)。
    per_tensor 分支: absmax 改成对整个 x 求 (标量 scale)。int4/group 之后再加。"""
    # YOUR CODE
    ...
```

**dataflow**: 无状态纯函数, 输入输出都在 `Attention.forward` 内 (Step 2)。config 由 Step 4 构造并挂到 module。

---

## Step 2 — 改 `nanovllm/layers/attention.py`

### 2a. `Attention.__init__` 加一个默认 None 的字段

现状 (行 45-57):
```python
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # >>> 新增: KV 量化配置, 默认 None = 不量化 (行为不变); allocate_kv_cache 里按 config 挂上
        # self.kv_quant: KVQuantConfig | None = None
```

### 2b. `Attention.forward` 在 `store_kvcache` 前插 hook

现状 (行 59-64):
```python
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # if malloc KV cache
        if k_cache.numel() and v_cache.numel():
            # >>> 新增: 量化开启时, 先把 k、v quant->dequant (注入误差), 再写 cache。
            # >>> 同一份 dequantized k、v 继续用于下面的 attention (存-读一致)。
            # if self.kv_quant is not None:
            #     k = fake_quant_dequant(k, self.kv_quant.k)
            #     v = fake_quant_dequant(v, self.kv_quant.v)
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```
文件顶部加 import:
```python
# >>> from nanovllm.utils.kv_quant import KVQuantConfig, fake_quant_dequant
```

**为什么改这里**: `store_kvcache` 是 target 28 层 + draft 层唯一的写入口 (draft 的 `Eagle3Midlayer.attn`
也是这个 `Attention` 类)。在此注入误差, prefill/decode/verify/draft 全覆盖。
**为什么用 dequantized 的 k、v 继续算 attention**: prefill 无 prefix cache 时 attention 直接用局部 k、v;
若不 dequant 这份, 存进 cache 的 (有损) 与本次算的 (无损) 不一致, 后续 decode 读到的和 prefill 用的会错位。

---

## Step 3 — 改 `nanovllm/config.py`

现状 Config 尾部 (spec decoding 字段之后):
```python
    # speculative decoding
    enable_spec: bool = False
    num_spec_tokens: int = 0  # k = when disble
    spec_method: str = "ngram"  # "ngram" | "eagle3"
    draft_model: str = ""       # eagle3 head path

    # >>> 新增: KV cache quantization (fake-quant: quant->dequant->存 bf16)
    # kv_quant_target: str = ""       # "" 关闭 | "int8"
    # kv_quant_draft:  str = ""       # "" 关闭 | "int8"
    # kv_quant_gran:   str = "per_token_head"
    # kv_quant_group_size: int = 0    # int4 时用, int8 忽略
```
`__post_init__` 无需改 (fake-quant 不动 block 大小; 关闭时字段全默认)。可选加一个 assert:
```python
    # >>> 可选: assert self.kv_quant_target in ("", "int8") and self.kv_quant_draft in ("", "int8")
```

**dataflow**: `LLM(...)` kwargs → `Config` → `ModelRunner` (Step 4) 读这些字段构造 `KVQuantConfig`。

---

## Step 4 — 改 `nanovllm/engine/model_runner.py` (`allocate_kv_cache` 挂 config)

现状 (行 137-146, kv_cache 分配之后的挂载段):
```python
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                # >>> 新增: target 层挂 target 量化 config (None 则不量化)
                # module.kv_quant = tgt_cfg
                layer_id += 1
        # proposer is not a submodule of self.model, mount manually
        if self.proposer is not None:
            self.proposer.midlayer.attn.k_cache = self.kv_cache[0, -1]
            self.proposer.midlayer.attn.v_cache = self.kv_cache[1, -1]
            # >>> 新增: draft 层挂 draft 量化 config
            # self.proposer.midlayer.attn.kv_quant = draft_cfg
```
在这段**之前**由 config 构造两个 config (放 `allocate_kv_cache` 开头或方法顶部):
```python
        # >>> from nanovllm.utils.kv_quant import KVQuantConfig
        # >>> tgt_cfg = KVQuantConfig.from_str(config.kv_quant_target)   # None if ""
        # >>> draft_cfg = KVQuantConfig.from_str(config.kv_quant_draft)
```

**为什么在这里挂**: 这是唯一同时能拿到「每个 target `Attention` module」和「draft `proposer.midlayer.attn`」
的地方, 且 layer 身份明确 (draft = 单独挂的那个)。挂在 module 上, forward 直接读 `self.kv_quant`, 零额外传参。

**注意**: 若 `spec_method != "eagle3"` (无 proposer), draft 分支不执行, `kv_quant_draft` 自然无效 —— 
draft 量化只在 eagle3 下有意义。

---

## Step 5 — 新文件 `bench_kvquant.py` (draft dtype sweep)

**目标**: 复用 `bench_eagle3_nanovllm.py` 的 step-drive + accept_len 测量, 外层扫 draft 量化配置,
主指标 accept_len, 附带「量化输出 vs bf16 baseline 的 token 一致率」(衡量误差是否已影响生成)。
target 恒 bf16 (v1 只测 draft 侧)。**skeleton + comment, 你填。**

```python
"""nano-vllm draft KV-quant sweep。target bf16, draft ∈ {bf16(off), int8}。
每个 condition 跑同一批 prompt (同 seed), 记 accept_len + 输出 token 序列;
以 draft=bf16 为 baseline, 算 int8 输出的逐 token 一致率 (agreement)。
用法 (nanovllm env): python bench_kvquant.py
"""
import os, json, time, datetime
import torch
from nanovllm import LLM, SamplingParams

PATH  = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
DRAFT = os.path.expanduser("~/huggingface/Qwen3-1.7B_eagle3/")
MAX_TOKENS, K, TEMPERATURE, SEED = 1024, 2, 0.6, 0

# sweep 的 condition: (label, kv_quant_draft 字符串)
CONDITIONS = [
    ("draft_bf16", ""),       # baseline (量化关闭)
    ("draft_int8", "int8"),
]

PROMPTS = [ ... ]  # 复用 bench_eagle3_nanovllm.py 的 10 条 (code/json/qa/story/trans)


def run_condition(label: str, kv_quant_draft: str):
    """建一个 LLM (eagle3 + 指定 draft 量化), 跑所有 prompt。
    返回 [{prompt_id, accept_len, gen_tokens: [int,...], decode_tok_s}, ...]。
    要点:
      - LLM(PATH, enforce_eager=True, enable_mixbatch=False, enable_spec=True,
            num_spec_tokens=K, spec_method="eagle3", draft_model=DRAFT,
            kv_quant_draft=kv_quant_draft)   # <- 新 kwarg
      - accept_len 数法照 bench_eagle3_nanovllm.py: 自己 while step(), 数 target_fwd,
        accept_len = (gen-1)/target_fwd
      - 额外记 seq 最终 token_ids (completion 部分), 供一致率对比
    """
    # YOUR CODE
    ...


def token_agreement(base_tokens: list[int], test_tokens: list[int]) -> float:
    """逐 token 前缀一致率: 对齐到 min 长度, 相同位置相等的比例。
    量化无损时应 ≈ 1.0 (draft 量化不改正确性, 只改 accept rate → 理论上输出完全一致!)。
    < 1.0 说明有 bug 或 target 侧也被影响。"""
    # YOUR CODE
    ...


def main():
    # 1) 跑所有 condition, 收集 rows
    # 2) 以 draft_bf16 为 baseline, 对每个 condition 算 token_agreement
    # 3) 写 md + jsonl (照 bench_eagle3 的 schema/result-only 风格)
    # 关键结论应看到: draft_int8 的 accept_len < draft_bf16, 但 agreement ≈ 1.0
    #   (draft 量化只降 draft 命中率, 不改 verify 的 target argmax -> 输出不变)
    # YOUR CODE
    ...


if __name__ == "__main__":
    main()
```

**读数预期 (帮你判断对不对)**:
- `agreement ≈ 1.0`: draft 量化**不改输出** —— 印证「draft KV 是 error-safe zone, 只影响 accept rate 不影响正确性」。
  若 agreement 明显 <1, 说明 hook 误伤了 target 侧 (查 config 是否只挂到了 draft 层)。
- `accept_len(int8) < accept_len(bf16)`: draft 质量下降 → 命中率降。降多少 = 这就是 tradeoff 的第一个数据点。

---

## 验证顺序

1. **关闭回归**: 不传 `kv_quant_*` → 跑现有 `bench_eagle3_nanovllm.py`, accept_len/输出与当前完全一致
   (证明 `self.kv_quant=None` 时 hook 零影响)。
2. **int8 冒烟**: `kv_quant_draft="int8"` 跑 1-2 条 prompt, 能正常生成、accept_len 掉但 >1、无 shape/dtype 报错。
3. **sweep**: `python bench_kvquant.py` 出 `{draft_bf16, draft_int8}` 的 accept_len + agreement 表。
4. 结果写 benchmark md/jsonl (result-only)。

## 后续 (不在本轮)
- dtype 扫描: int4 (head_dim 内 group_size 32/64/128) + fp8 (e4m3/e5m2, fake 模拟)。
- target 侧量化 + (target,draft) 联合 condition matrix + context 长度曲线。
- 第二步: port vLLM Triton per-token-head `reshape_and_cache` kernel 做真省显存 (见 writeup §4)。
