# EAGLE-3 Implementation Guide (nano-vllm)

以算法 flow 为主线。每个改动按 **现状 → 改成什么 → 为什么** 展开,并标注 input/output 的来源与去向。
算法推导见 [eagle3_writeup.md](eagle3_writeup.md) §2。全文按 tp=1 写(bench 场景);tp>1 需另处理 rank 逻辑。

## 0. Engine-level 闭环 (谁在什么时候调谁)

**Top-level controller = `LLMEngine.step()`**。一个 iteration 四拍,propose 的产物被下一个 iteration 消费:

```
LLMEngine.step()
 ├─ 1. Scheduler.schedule()            → (prefill_seqs, decode_seqs)
 ├─ 2. ModelRunner.run()      [GPU]    → verify/prefill forward: 出 logits, 顺便暂存 aux
 ├─ 3. Scheduler.postprocess_spec()    → accept/reject, committed tokens 进 seq.token_ids
 └─ 4. ModelRunner.propose()  [GPU]    → 算法 step 3-6, 产出 seq.draft_tokens
                                          └→ 下一轮 step 2 的 prepare_verify 消费
```

propose 必须在 postprocess **之后**:window pass 的 shifted input 需要刚 commit 的 tokens。

顺便说明一个已有配合:`Scheduler._schedule_decode` 每轮固定
`seq.num_scheduled_tokens = 1 + self.k` 并 `may_append(seq, self.k)`——eagle3 每轮
恒出 k 个 draft(不像 ngram 会空手),所以这两处与 eagle3 天然一致,scheduler 不用改。

## 1. 算法 → module 对照

### [算法 step 1] Target forward 抽 3 层 aux hidden `[n, 2048]×3` — **done**

**改**: `models/qwen3.py::Qwen3Model`。

现状:`forward` 只返回最终 `hidden_states`。改后(已完成):

```python
# __init__
self.aux_layers: tuple[int, ...] = ()      # ModelRunner 构造后赋 (1, 13, 24), 运行期只读

# forward 的 layer loop
for i, layer in enumerate(self.layers):
    hidden_states, residual = layer(positions, hidden_states, residual)
    if i in self.aux_layers:
        aux.append(hidden_states + residual)
...
if self.aux_layers:
    return hidden_states, aux
return hidden_states
```

为什么是 `hidden + residual`:nano 的层间传递是 (hidden, residual) 两条,真正的
residual stream 值 = 两者之和(`RMSNorm.add_rms_forward` 也是先加再 norm);
EAGLE-3 head 训练时吃的就是这个 pre-norm 值。
为什么 `(1,13,24)`:vLLM 默认公式 `(2, n//2, n-3)` 以 embedding=0 起算,
换成 module index 各减 1。
`Qwen3ForCausalLM.forward` 是 `return self.model(...)`,tuple 自动透传,不改。

| | 来源/去向 |
|---|---|
| input | `input_ids, positions` ← `ModelRunner.run_model()`(调用方式不变) |
| output | `(hidden_states, aux)` → `ModelRunner.run_model()`;hidden 原路进 `compute_logits`,aux 进算法 step 2 |

### [算法 step 2] concat → `[n, 6144]`,按 seq 暂存

**改 1**: `model_runner.py::run_model` 的 eager 分支。现状是单行:

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))
```

改成拆开、tuple 时解包:

```python
out = self.model(input_ids, positions)
if isinstance(out, tuple):                  # eagle3: (hidden, aux)
    hidden, aux = out
    self._aux = torch.cat(aux, dim=-1)      # 3×[total_n,2048] → [total_n,6144]
else:
    hidden = out                            # vanilla: 行为不变
return self.model.compute_logits(hidden)
```

cuda graph 分支不动:它 replay 预 capture 的 graph、从固定 buffer 读输出,拿不到 aux;
但 eagle3 下所有需要 aux 的 forward(verify/prefill)都以 `is_prefill=True` 调用,
必然落在 eager 分支;graph 分支只服务 vanilla decode,eagle3 模式下不存在
(decode 就是 verify)。

**改 2**: `_run_verify` 与 `_run_one`(prefill 侧)把 `self._aux` 切回 per-seq 暂存。
`_run_verify` 现状已有一个按 seq 切 argmax 的 loop,同一个 loop 顺便切 aux:

```python
out, idx = [], 0
for seq in seqs:
    n = 1 + len(seq.draft_tokens)
    out.append(target_argmax[idx:idx + n])
    self.aux_features[seq.seq_id] = (self._aux[idx:idx + n], seq.num_cached_tokens)  # 新增
    idx += n
```

`_run_one` 的 prefill 调用后同理,按 `seq.num_scheduled_tokens` 切
(prepare_prefill 里 `seqlen_q = num_scheduled_tokens`,边界一致)。

为什么 start_pos(`seq.num_cached_tokens`)必须**此刻**记录:第 3 拍
`postprocess_spec` 会把它改写成 `len(seq)-1`(rollback 记账),到第 4 拍 propose 时
本轮起点已丢失。为什么暂存在 ModelRunner:aux 产生于第 2 拍、消费于第 4 拍,
中间隔着 engine 侧 accept/reject,GPU tensor 留在 runner 内跨拍传引用即可。

| | 来源/去向 |
|---|---|
| input | `aux` ← `Qwen3Model.forward`;per-seq 切分边界 = `_run_verify` 已有的 `n` / prefill 的 `num_scheduled_tokens` |
| output | `self.aux_features[seq_id] = (feats [n_i,6144], start_pos)` → `ModelRunner.propose()` 读取(`pop`,用完即删) |

### [算法 step 3] FC projection `[n, 6144] → [n, 2048]`

**新增**: `utils/eagle3.py::Eagle3Proposer` 的 `self.fc = nn.Linear(6144, 2048, bias=False)`,
在 `propose()` 第一行对整个 window 调用。

为什么 fc 不在暂存时(step 2)提前做:fc 是 head 权重,归 proposer 持有,
ModelRunner 不应依赖 proposer 的参数;同时保持"暂存的永远是 target 原料(fc 前 6144)"
这一边界。

| | 来源/去向 |
|---|---|
| input | `features [n_use, 6144]` ← `ModelRunner.propose()` 截断后作实参 |
| output | `feature [n_use, 2048]` → 同函数内进 midlayer(step 4),不跨 module |

### [算法 step 4] Window pass: 整个 window 过 midlayer,填 draft KV,出 draft_1

职责一分为二:**ModelRunner 负责"喂什么"(它懂 Sequence/block 语义),
Proposer 负责"怎么算"(只懂 tensor,不接触 paged layout)**。

**新增 A**: `model_runner.py::propose(seqs, k)`。骨架:

```python
@torch.inference_mode()
def propose(self, seqs: list[Sequence], k: int) -> list[list[int]]:
    drafts = []
    for seq in seqs:                                   # 逐 seq 串行 (bs=1 场景)
        feats, start = self.aux_features.pop(seq.seq_id)
        n_use  = min(len(feats), len(seq) - 1 - start)
        input_ids = seq[start+1 : start+n_use+1]       # Sequence.__getitem__ 支持 slice
        positions = list(range(start, start + n_use))
        num_draft = k if seq.num_completion_tokens > 0 else 0

        slot = lambda p: seq.block_table[p // self.block_size] * self.block_size + p % self.block_size
        slot_mapping = [slot(p) for p in positions]
        # window pass context: 照 prepare_verify 的写法造 tensor
        set_context(True, cu_seqlens_q=[0, n_use], cu_seqlens_k=[0, start + n_use],
                    max_seqlen_q=n_use, max_seqlen_k=start + n_use,
                    slot_mapping=..., block_tables=...)

        # single-token pass 每步的 (slot, context_len), j = 1..k-1
        steps = [(slot(p), p + 1) for p in range(len(seq) - 1, len(seq) - 1 + max(num_draft - 1, 0))]

        drafts.append(self.proposer.propose(input_ids_t, positions_t,
                                            feats[:n_use], num_draft, steps, block_tables))
        reset_context()
    return drafts
```

`n_use` 公式逐 case 验证:
- **decode 轮**: verify 处理 1+m 个、accept c 个 → `len(seq)-1-start = c+1`,
  截到 accepted 长度;此时 `input_ids` 恰好 = 本轮 committed 的 c+1 个 tokens
- **最后一个 prefill chunk**: postprocess 刚 append 首个 sampled token →
  `len(seq)-1-start` = chunk 长度,全用,最后一个 input = 该 sampled token
- **非最后 chunk**: `len(seq)-1-start > len(feats)` → 取 `len(feats)`,
  shifted 的下一个 token 是已知 prompt token;`num_draft=0`,只填 KV 不出 draft

context 各字段为什么这样填:`slot_mapping` 用 `seq.block_table` 的相同 slot 编号——
draft KV 写第 29 层,与 target 无冲突(layer 是 `kv_cache` 的独立维);
`cu_seqlens_k = start+n_use` 含 prefix,因为 draft attention 要看 0..start-1 的
历史 draft KV(之前轮次 window pass 写的),`block_tables` 非 None 触发
`Attention.forward` 的 prefix-cache 分支(k/v 从 cache 读,`flash_attn_varlen_func`
的 causal 语义把 q 右对齐到 k 末尾——verify 已依赖同一行为)。

**新增 B**: `Eagle3Proposer.propose()` 前半:

```python
feat = self.fc(features)                               # step 3
x, residual = self.midlayer(positions, self.embed_tokens(input_ids), feat)
hidden, prenorm = self.norm(x, residual)               # 此刻 draft KV 已写好
if num_draft == 0:
    return []
```

draft KV 写入零额外代码:midlayer 内的 `Attention` 读 context 自动
`store_kvcache`,与 target 28 层同路径。

| | 来源/去向 |
|---|---|
| input (`ModelRunner.propose`) | `seqs` ← `LLMEngine.step()` 第 4 拍(token_ids 已含 committed);`aux_features` ← step 2 |
| input (`Eagle3Proposer.propose`) | `input_ids/positions/features/num_draft/steps/block_tables` ← `ModelRunner.propose()`;window 的 attention metadata ← 已 set 的 context |
| output | draft KV → `kv_cache[:, -1]`(经 `Attention.forward` → `store_kvcache`);`hidden[-1]`/`prenorm[-1]` → 本函数循环体(step 6) |

### [算法 step 5] Midlayer 内部

**新增**: `utils/eagle3.py::Eagle3Midlayer`。构造(维度按 head config 硬编码):

```python
self.input_layernorm = RMSNorm(2048); self.hidden_norm = RMSNorm(2048)
self.qkv_proj = QKVParallelLinear(4096, 128, 16, 8)    # 第一个参数 = 输入维 4096
self.o_proj   = RowParallelLinear(2048, 2048, bias=False)
self.rotary_emb = get_rope(128, rotary_dim=128, max_position=40960, base=1e6)
self.attn = Attention(16, 128, 128 ** -0.5, 8)         # k_cache/v_cache 由 allocate_kv_cache 事后挂
self.post_attention_layernorm = RMSNorm(2048)
self.mlp = Qwen3MLP(2048, 6144, "silu")                 # 结构相同, 直接 import
```

forward——attention 部分照抄 `Qwen3Attention.forward` 的
split/view/rope/attn/flatten 调用,只删 q_norm/k_norm、把输入换成双流 concat:

```python
def forward(self, positions, embeds, feature):
    e = self.input_layernorm(embeds)          # 单参数调用 = 纯 norm
    residual = feature                        # residual = pre-norm feature (norm 前原值)
    h = self.hidden_norm(feature)
    qkv = self.qkv_proj(torch.cat([e, h], dim=-1))     # embed 在前, 权重布局依赖顺序
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = q.view(-1, 16, 128); k = k.view(-1, 8, 128); v = v.view(-1, 8, 128)
    q, k = self.rotary_emb(positions, q, k)
    o = self.attn(q, k, v)                    # 读 context: 写 draft KV + varlen/decode attention
    x = self.o_proj(o.flatten(1, -1))
    x, residual = self.post_attention_layernorm(x, residual)   # first residual add
    x = self.mlp(x)
    return x, residual                        # second residual add 留给 proposer 的 self.norm
```

`RMSNorm.forward(x, residual)` 返回 `(norm(x+residual), x+residual)`——所以 proposer 里
`self.norm(mlp_out, residual)` 一个调用同时完成 second residual add,并给出
lm_head 输入(第一返回值)和下一步 feature(第二返回值 = pre-norm sum)。

不能复用 `Qwen3DecoderLayer`:qkv 输入 4096 ≠ 2048、无 q_norm/k_norm、双 norm 双流。
但所有原语(`QKVParallelLinear`/`RowParallelLinear`/`get_rope`/`Attention`/`RMSNorm`/`Qwen3MLP`)照用。

| | 来源/去向 |
|---|---|
| input | `positions, embeds, feature` ← `Eagle3Proposer.propose()`(window 与 single-token pass 都调它,靠 context 区分路径) |
| output | `(mlp_out, residual)` → `Eagle3Proposer.propose()` 的 `self.norm(x, residual)` → 拆成 (lm_head 输入, 下一步 feature) |

### [算法 step 6] lm_head → d2t → 自回归 loop

**新增 A**: `Eagle3Proposer.propose()` 循环体。窗口 pass 之后:

```python
draft_id = self.lm_head(hidden[-1:]).argmax(dim=-1)
tok = (draft_id + self.d2t[draft_id]).item()           # draft vocab → target vocab
feat, out = prenorm[-1:], [tok]
for slot_j, ctx_len in steps:                          # k-1 次 single-token pass
    set_context(False, slot_mapping=..., context_lens=[ctx_len], block_tables=block_tables)
    x, residual = self.midlayer(positions=[ctx_len - 1],           # p_j = ctx_len-1, target 编号续
                                embeds=self.embed_tokens([tok]),   # fresh lookup, norm 在 midlayer 做
                                feature=feat)
    hidden, feat = self.norm(x, residual)              # feat = 本步 pre-norm hidden
    draft_id = self.lm_head(hidden).argmax(dim=-1)
    tok = (draft_id + self.d2t[draft_id]).item()
    out.append(tok)
return out
```

`set_context(False, ...)` 走 `Attention.forward` 的 decode 分支
(`flash_attn_with_kvcache`),attend 全部 draft KV。分工:每步的 `(slot, ctx_len)`
由 ModelRunner 预计算传入(step 4 的 `steps`),proposer 只拿现成 slot 号,不懂 block 语义。

**改 B**: `llm_engine.py::step()` 接线。现状 spec 分支在 run **之前** 做 ngram propose
(`llm_engine.py:73-76`);eagle3 分支把 propose 移到 postprocess **之后**:

```python
# run 之前: spec_method == "ngram" 才走原 propose_draft 逻辑
# postprocess_spec(...) 之后:
if self.config.spec_method == "eagle3":
    live = [s for s in decode_seqs + prefill_seqs if not s.is_finished]
    for s in live:
        self.scheduler.block_manager.may_append(s, self.scheduler.k)   # 见 pitfall 4
    for s, d in zip(live, self.model_runner.call("propose", live, self.scheduler.k)):
        s.draft_tokens = d
```

finished 的 seq 没有下一轮,不 propose。`num_scheduled_tokens` 不用在这里设:
`_schedule_decode` 每轮固定设 `1 + self.k`,而 eagle3 恒出 k 个 draft,两者一致
(ngram 需要在 engine 里改小它,是因为 ngram 可能空手)。

**改 C**: `config.py` 加两个字段,`__post_init__` 加对应 assert:

```python
spec_method: str = "ngram"     # "ngram" | "eagle3"
draft_model: str = ""          # eagle3 head 路径
```

| | 来源/去向 |
|---|---|
| input | `hidden[-1]/prenorm[-1]` ← step 4 window pass;每步 `(slot, ctx_len)` ← `ModelRunner.propose()` 预计算 |
| output | `draft_tokens: list[int]` → `ModelRunner.propose()` 汇成 `list[list[int]]` → `LLMEngine.step()` 写 `seq.draft_tokens` → 下一轮 `prepare_verify`(拼 q_ids)与 `postprocess_spec`(比对)消费 |

## 2. 文件清单 (按实现顺序)

| 文件 | 做什么 | 对应算法 step | LOC |
|---|---|---|---|
| `models/qwen3.py` | aux 抽取 | 1 | ~10, **done** |
| `utils/eagle3.py` | `Eagle3Midlayer` + `Eagle3Proposer`(fc/norm/lm_head/d2t/load_weights/propose) | 3-6 | ~130 |
| `engine/model_runner.py` | run_model 解包、aux 暂存、KV +1 层、`propose()` | 2, 4, 6 的喂数侧 | ~80 |
| `engine/llm_engine.py` | step() 第 4 拍接线 | 闭环 | ~10 |
| `config.py` | `spec_method` / `draft_model` | — | ~5 |
| `scheduler.py` / `block_manager.py` | **零改动**(机制复用的验证点) | — | 0 |

## 3. Pitfalls (写的时候查这里)

1. **init 顺序**: `ModelRunner.__init__` 里 proposer 构造 + `load_weights` 必须在
   `allocate_kv_cache` **之前**——block 数 = (总显存×util − used − peak) / block_bytes,
   权重不先占显存,block 会算多,运行时 OOM。同时在此处设
   `self.model.model.aux_layers = (1, 13, 24)`。
2. **KV +1 层**: `allocate_kv_cache` 里 `block_bytes` 与 `self.kv_cache` 的 layer 维
   `num_hidden_layers` → `+1`;现有挂载 loop 只遍历 `self.model.modules()`,
   proposer 不在其中,loop 后手动挂 `kv_cache[0, -1]` / `kv_cache[1, -1]`。
3. **load_weights**(手写 ~25 LOC;不走 `utils/loader.py`,因为 head 是
   pytorch_model.bin 且 key 带 `midlayer.` 前缀): fp16 → `.to(bfloat16)`;
   q/k/v `cat(dim=0)` 进 `qkv_proj.weight`、gate/up `cat(dim=0)` 进
   `gate_up_proj.weight`(顺序对照 `Qwen3MLP`);`d2t` 存 int64 buffer;t2d 跳过。
4. **block 边界**: single-token pass 写 draft KV 最远到 position `len(seq)+k-3`。
   schedule 时的 `may_append` 只保到 verify 用量,且 propose 发生在下一次 schedule
   之前(prefill 后的首次 propose 更是完全没预留)→ engine 侧 propose 前
   `may_append(seq, k)` 扩块;`_new_blocks` 幂等,块够时是 no-op,下一轮 verify
   反正需要这些块,不浪费。
5. **rope 共享**: `get_rope` 带 `lru_cache`,head 与 target 参数相同
   (128/40960/1e6),命中同一实例,正确且省显存。
6. **cuda graph**: eagle3 下 verify/prefill/propose 全 eager,aux 解包只发生在
   `run_model` 的 eager 分支,graph 分支不改也不会命中。

## 4. 验证顺序

Step 1 跑 vanilla 无回归(done)→ proposer 单测(随机 feature 查 shape 和 d2t)→
接通后看 **accept_len**:>1.5 基本对;≈1.0 说明 shift/position/norm 有错
(最灵敏指标)→ bench(k=2,同 schema 对比 vLLM 1.52×/1.72 与 ngram 1.14×/1.17)。
