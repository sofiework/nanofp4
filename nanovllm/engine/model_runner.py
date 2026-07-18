import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.eagle3 import Eagle3Proposer


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        # eagle3: model returns tuple (aux), cuda graph capture would fail; runtime is all-eager anyway
        self.enforce_eager = config.enforce_eager or config.spec_method == "eagle3"
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        # eagle3
        self._aux: torch.Tensor | None = None
        self.aux_features: dict[int, tuple[torch.Tensor, int]] = {}
        # proposer weights must occupy memory before allocate_kv_cache
        self.proposer = None
        if config.spec_method == "eagle3":
            self.proposer = Eagle3Proposer(self.model.model.embed_tokens)
            self.proposer.load_weights(config.draft_model)
            self.model.model.aux_layers = (1, 13, 24)


        # one time initiate
        self.warmup_model()
        # calculate & malloc blocks in GPU memory for KV cache
        self.allocate_kv_cache()


        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # = self.model_runner.run(seqs, is_prefill)
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, [])
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        num_layers = hf_config.num_hidden_layers + (1 if self.proposer is not None else 0)  # eagle3: +1 draft layer
        block_bytes = 2 * num_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, num_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        # proposer is not a submodule of self.model, mount manually
        if self.proposer is not None:
            self.proposer.midlayer.attn.k_cache = self.kv_cache[0, -1]
            self.proposer.midlayer.attn.v_cache = self.kv_cache[1, -1]

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # start (skip cached) token_id
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q

            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:    # warmup
                continue
            
            # start/end block idx
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size

            for i in range(start_block, end_block):
                # block physical address
                slot_start = seq.block_table[i] * self.block_size

                # first block / end block
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                # regular block
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    # prepare data for verify
    # [last_token, d1, d2, ..., dm] * seq_len
    def prepare_verify(self, seqs: list[Sequence]):
        input_ids, positions = [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0] # cumulative seq length for flash attn
        max_seqlen_q = max_seqlen_k = 0
        slot_mapping = []

        for seq in seqs:
            start = seq.num_cached_tokens          # L-1
            q_ids = [seq.last_token] + seq.draft_tokens  # 1+m 个 token
            seqlen_q = len(q_ids)
            end = start + seqlen_q
            seqlen_k = end

            input_ids.extend(q_ids)
            positions.extend(range(start, end))

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            for pos in range(start, end):
                block = seq.block_table[pos // self.block_size]
                slot_mapping.append(block * self.block_size + pos % self.block_size)

        block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables, is_verify=True)
        return input_ids, positions


    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # last block start + offset
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures
    
    @torch.inference_mode()
    # prepare each seq for Eagle3Proposer.proposer
    def propose(self, seqs: list[Sequence], k:int) -> list[list[int]]:
        drafts = []

        for seq in seqs:
            # self.aux_features: dict[int, tuple[torch.Tensor, int]]
            feats, start = self.aux_features.pop(seq.seq_id)

            # num_of_pair (feature, token)
            # feature[i] + embed(token[i+1]) → draft token[i+2]
            # i = start ... start + n_use - 1
            n_use = min(len(feats), len(seq) - 1 - start)

            input_ids = seq[start + 1 : start + n_use + 1]
            positions = list(range(start, start + n_use))
            num_draft = k if seq.num_completion_tokens > 0 else 0

            slot = lambda p: seq.block_table[p // self.block_size] * self.block_size + p % self.block_size
            slot_mapping = [slot(p) for p in positions]

            input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            cu_seqlens_q = torch.tensor([0, n_use], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            cu_seqlens_k = torch.tensor([0, start + n_use], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            block_tables = self.prepare_block_tables([seq])

            # window pass context
            set_context(True, cu_seqlens_q, cu_seqlens_k,
                    max_seqlen_q=n_use, max_seqlen_k=start + n_use,
                    slot_mapping=slot_mapping, block_tables=block_tables)
            # single-token pass context
            steps = [(slot(p), p + 1) for p in range(len(seq) - 1, len(seq) - 1 + max(num_draft - 1, 0))]

            drafts.append(self.proposer.propose(input_ids, positions,
                                            feats[:n_use], num_draft, steps, block_tables))
            reset_context()

        return drafts



    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # eager mode
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            out = self.model(input_ids, positions)
            # eagle3
            if isinstance(out, tuple):
                hidden, aux = out
                self._aux = torch.cat(aux, dim=-1) # 3×[n, 2048] → [n, 6144]
            # vanilla
            else:
                hidden = out
            return self.model.compute_logits(hidden)
        
        # CUDA graph
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])
    

    # decode forward + prefill forward separately, token_ids order = decode first then prefill
    def run(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence]) -> list[int]:
        
        # spec decoding enable
        if self.config.enable_spec:
            return self._run_spec(prefill_seqs, decode_seqs)

        token_ids = []

        # decode (use CUDA graph)
        if decode_seqs:
            token_ids += self._run_one(decode_seqs, is_prefill=False)

        # prefill (varlen kernel)
        if prefill_seqs:
            token_ids += self._run_one(prefill_seqs, is_prefill=True)
        return token_ids
    

    # verify forawrd (n-gram & eagle3)
    @torch.inference_mode()
    def _run_verify(self, seqs: list[Sequence]) -> list[list[int]]: 
        # [last token] + draft_tokens (1 + m) -> 1D tensor
        input_ids, positions = self.prepare_verify(seqs) 

        # complete logits
        # shape = [total_tokens, vocab]
        logits = self.run_model(input_ids, positions, is_prefill=True)
        reset_context()

        if self.rank != 0 :
            return []
        
        # greedy argmax
        target_argmax: list[int] = logits.argmax(dim=-1).tolist()  # token_ids

        out, idx = [], 0
        for seq in seqs:
            n = 1 + len(seq.draft_tokens)

            # each seq token_id
            out.append(target_argmax[idx:idx + n])
            # each seq aux_feature
            if self._aux is not None:
                self.aux_features[seq.seq_id] = (self._aux[idx:idx + n], seq.num_cached_tokens)

            idx += n
        return out
    
    
    # spec decoding
    def _run_spec(self, prefill_seqs, decode_seqs):
        self.aux_features.clear()

        # decode -> 1+m token_ids
        decode_verify: list[list[int]] = self._run_verify(decode_seqs) if decode_seqs else []

        # prefill (same)
        prefill_tokens = self._run_one(prefill_seqs, is_prefill=True) if prefill_seqs else []
        return prefill_tokens, decode_verify


    def _run_one(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        # eagle3: stash per-seq aux + start_pos (postprocess will rewrite num_cached_tokens)
        if is_prefill and self._aux is not None:
            idx = 0
            for seq in seqs:
                if seq.block_table:    # skip warmup seqs (never consumed by propose)
                    self.aux_features[seq.seq_id] = (self._aux[idx:idx + seq.num_scheduled_tokens], seq.num_cached_tokens)
                idx += seq.num_scheduled_tokens
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids if self.rank == 0 else []

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
