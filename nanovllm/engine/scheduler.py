from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


import time

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        # mix_batch
        self.enable_mixbatch = config.enable_mixbatch

        # priority scheduling
        self.priority_deadline = config.priority_deadline

        # speculative decoding
        self.k = config.num_spec_tokens if config.enable_spec else 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        seq.arrival_time = time.monotonic()
        self.waiting.append(seq)
    
    # priority scheduling
    # update seq priority (if > deadline)
    def _priority(self, seq, now) -> int:
        prio = seq.priority
        if self.priority_deadline > 0 and now - seq.arrival_time > self.priority_deadline:
            prio = max(prio, 1) # normal -> fast
        return prio
    
    # pick next seq to schedule
    def _pick_waiting(self, now, scheduled_seqs) -> Sequence:
        best, best_key = None, None 
        # tuple[started: bool, priority: int, -arrival_time: float]
        
        for seq in self.waiting:
            if seq in scheduled_seqs:
                continue

            started = bool(seq.block_table)
            key = (started, self._priority(seq, now), -seq.arrival_time)
            
            if best_key is None or key > best_key:
                best, best_key = seq, key
        return best

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        # self.max_num_seqs - "seq limit in batch" - "Decode"
        # self.max_num_batched_tokens - "token limit in batch" - "Prefill"


        # Mixed batch on / off
        # Mixed batch - return prefil/decode scheduled_seqs separately
        if self.enable_mixbatch:
            # decode first            
            decode_seqs, num_batched_tokens = self._schedule_decode(0)
            prefill_seqs, _ = self._schedule_prefill(num_batched_tokens)

        # off, fall back: prefill / decode only
        else: 
            prefill_seqs, _ = self._schedule_prefill(0)
            decode_seqs = [] if prefill_seqs else self._schedule_decode(0)[0]
        
        assert prefill_seqs or decode_seqs
        return prefill_seqs, decode_seqs
    
    def _schedule_decode(self, num_batched_tokens: int):
        decode_seqs = []

        while self.running and len(decode_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # need new block & no free block -> preempt
            while not self.block_manager.can_append(seq, self.k):
                if self.running:
                    # preempt LRU
                    self.preempt(self.running.pop())
                else:
                    # preempt self
                    self.preempt(seq)
                    break

            else: # can_append = True
                seq.num_scheduled_tokens = 1 + self.k
                seq.is_prefill = False
                self.block_manager.may_append(seq, self.k)
                decode_seqs.append(seq)
                # schedule 1 each time / 1 + k in spec-decoding
                num_batched_tokens += 1 + self.k 
                
        # put back to running list
        self.running.extendleft(reversed(decode_seqs))
        return decode_seqs, num_batched_tokens
        
        
    def _schedule_prefill(self, num_batched_tokens: int):
        prefill_seqs = []
        now = time.monotonic()

        # Prefill
        while self.waiting and len(prefill_seqs) < self.max_num_seqs:
            seq = self._pick_waiting(now, prefill_seqs)
            if seq is None:
                break

            # check max_token in batch
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            
            # Chunked prefil
            # new Sequence -> can_allocate() -> return num_cached_block
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                # prefix cache -> remain num_tokens to process
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            # preempted Sequence (blocks allocated)
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # Scheduling
            # break if > max_token_in_batch, but not break if new Sequence
            if remaining < num_tokens and prefill_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table: # new Sequence
                self.block_manager.allocate(seq, num_cached_blocks)

            # schedule remaining tokens
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # Sequence prefill done -> running
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            prefill_seqs.append(seq)

        return prefill_seqs, num_batched_tokens
    

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence], token_ids: list[int]):
        # decode (MUST same order -> model_runner.run())
        decode_token_ids  = token_ids[:len(decode_seqs)]
        prefill_token_ids = token_ids[len(decode_seqs):]
        
        for seq, token_id in zip(decode_seqs, decode_token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            
            seq.append_token(token_id)
            # decode done / context limit
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                

        # prefill
        for seq, token_id in zip(prefill_seqs, prefill_token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # not done prefill
            if seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # prefill generate EOS -> finished
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            

    # spec decoding
    def postprocess_spec(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence], token_ids: list[int], decode_verify):
        # Decode (MUST same order -> model_runner.run())
        for seq, target_argmax in zip(decode_seqs, decode_verify):
            # target_argmax [a0, a1, ... am, am+1]
            # drafts        [d1, d2, ... dm]
            drafts = seq.draft_tokens 
            committed = []

            # compare target / draft
            for j in range(len(drafts)):
                committed.append(target_argmax[j])
                if target_argmax[j] != drafts[j]:
                    break
            
            else: # accept all
                committed.append(target_argmax[len(drafts)])
            c = len(committed) - 1 # num_accepted

            # append token -> seq
            finished = False
            for t in committed:
                seq.append_token(t)
                if (not seq.ignore_eos and t == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    finished = True
                    break
            # reset draft_tokens
            seq.draft_tokens = []


            ### KV rollback
            # early terminate
            if finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                continue
        
            # verify forward write 1+m to KV, but 1+c valid
            seq.num_scheduled_tokens = 1 + c
            # hash 1+c to prefix cache
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens = len(seq) - 1 # last_token = token_ids[len(seq)]
            seq.num_scheduled_tokens = 0
        
        
        # Prefill
        for seq, token_id in zip(prefill_seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # not done prefill
            if seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # prefill generate EOS -> finished
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)