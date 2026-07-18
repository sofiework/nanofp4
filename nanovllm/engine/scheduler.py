from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        # mix batch
        self.enable_mixbatch = config.enable_mixbatch

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        # iteration-level scheduling: return prefill / decode seqs separately,
        # model_runner runs one forward each (decode via cuda graph, prefill via varlen).
        if self.enable_mixbatch:
            # decode first (cheap, 1 token/seq), then fill the batch with prefill
            decode_seqs, budget = self._schedule_decode()
            prefill_seqs = self._schedule_prefill(budget)
        else:
            # off: prefill-first, decode only when nothing to prefill (original behavior)
            prefill_seqs = self._schedule_prefill()
            decode_seqs = [] if prefill_seqs else self._schedule_decode()[0]
        assert prefill_seqs or decode_seqs
        return prefill_seqs, decode_seqs

    def _schedule_decode(self, num_batched_tokens: int = 0) -> tuple[list[Sequence], int]:
        decode_seqs = []
        while self.running and len(decode_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # need a new block but none free -> preempt LRU (or self, then stop)
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
                num_batched_tokens += 1
        self.running.extendleft(reversed(decode_seqs))
        return decode_seqs, num_batched_tokens

    def _schedule_prefill(self, num_batched_tokens: int = 0) -> list[Sequence]:
        prefill_seqs = []
        while self.waiting and len(prefill_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and prefill_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            prefill_seqs.append(seq)
        return prefill_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence], token_ids: list[int]):
        # token_ids order must match model_runner.run(): decode first, then prefill
        n = len(decode_seqs)
        for seq, token_id in zip(decode_seqs, token_ids[:n]):
            self._commit(seq, token_id, is_prefill=False)
        for seq, token_id in zip(prefill_seqs, token_ids[n:]):
            self._commit(seq, token_id, is_prefill=True)

    def _commit(self, seq: Sequence, token_id: int, is_prefill: bool):
        self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            return  # chunked prefill not done yet
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
