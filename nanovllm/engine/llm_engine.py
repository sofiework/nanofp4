import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

from nanovllm.utils.spec_decoding.ngram import propose_draft


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config

        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

        # per-token timing for benchmark (off by default, set True before generate)
        self.collect_metrics = False
        self.submit_time: dict[int, float] = {}
        self.token_times: dict[int, list[float]] = {}
        self._recorded: dict[int, int] = {}

    def reset_metrics(self):
        self.submit_time.clear()
        self.token_times.clear()
        self._recorded.clear()

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # tokenize prompt -> make it Sequence -> Scheuler.add()
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt) # str -> token_ids
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        if self.collect_metrics:
            self.submit_time[seq.seq_id] = perf_counter()
            self.token_times[seq.seq_id] = []
            self._recorded[seq.seq_id] = 0

    # schedule each sequence
    def step(self) -> tuple[list[tuple[int, list[int]]], int, int]:
        # schedule
        prefill_seqs, decode_seqs = self.scheduler.schedule()

        # n-gram / spec decoding propose_draft
        if self.scheduler.k and self.config.spec_method == "ngram":
            for seq in decode_seqs:
                seq.draft_tokens = propose_draft(seq.token_ids, self.scheduler.k, n=3)
                seq.num_scheduled_tokens = 1 + len(seq.draft_tokens)

        # for throughput print: prefill = scheduled tokens, decode = 1 per seq
        num_prefill_tokens = sum(seq.num_scheduled_tokens for seq in prefill_seqs)
        num_decode_tokens = len(decode_seqs)

        
        # n-gram / spec decoding verify (compare token_ids)
        if self.scheduler.k:
            prefill_tokens, decode_verify = self.model_runner.call("run", prefill_seqs, decode_seqs)
            self.scheduler.postprocess_spec(prefill_seqs, decode_seqs, prefill_tokens, decode_verify)

            # eagle3: propose for next round (after postprocess: window needs committed tokens)
            if self.config.spec_method == "eagle3":
                live = [s for s in decode_seqs + prefill_seqs if not s.is_finished]
                for s in live:
                    self.scheduler.block_manager.may_append(s, self.scheduler.k)
                for s, d in zip(live, self.model_runner.call("propose", live, self.scheduler.k)):
                    s.draft_tokens = d
        # regular forward
        else:
            # call GPU -> run one forward pass -> return next token_ids
            token_ids = self.model_runner.call("run", prefill_seqs, decode_seqs)
            # Scheduler append token_ids to Sequence
            self.scheduler.postprocess(prefill_seqs, decode_seqs, token_ids)


        # per-token timestamps: record now for every seq that produced new token(s) this step
        if self.collect_metrics:
            now = perf_counter()
            for seq in decode_seqs + prefill_seqs:
                n = seq.num_completion_tokens
                rec = self._recorded.get(seq.seq_id, 0)
                if n > rec:
                    self.token_times[seq.seq_id].extend([now] * (n - rec))
                    self._recorded[seq.seq_id] = n

        # for print result -> (seq_id, completed token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in prefill_seqs + decode_seqs if seq.is_finished]
        return outputs, num_prefill_tokens, num_decode_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        
        # for each prompt -> add_request() add Sequence to Scheduler
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.

        while not self.is_finished():
            t = perf_counter()

            # Core function
            output, num_prefill_tokens, num_decode_tokens = self.step()

            dt = perf_counter() - t
            if num_prefill_tokens:
                prefill_throughput = num_prefill_tokens / dt
            if num_decode_tokens:
                decode_throughput = num_decode_tokens / dt

            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)

        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
