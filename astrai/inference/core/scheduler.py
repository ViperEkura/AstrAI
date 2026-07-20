import logging
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch

from astrai.inference.core.cache import ContiguousCache, KVCache
from astrai.inference.core.executor import Executor
from astrai.inference.core.task import STOP, Task, TaskManager, TaskStatus
from astrai.model.automodel import AutoModel
from astrai.tokenize.tokenizer import AutoTokenizer

logger = logging.getLogger(__name__)


class InferenceScheduler:
    """Continuous batching loop: cleanup -> refill -> prefill -> decode (all groups)."""

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 16,
        max_seq_len: Optional[int] = None,
        max_prompt_len: int = 2048,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache: Optional[KVCache] = None,
    ):
        config = model.config

        if max_seq_len is not None:
            self.max_seq_len = max_seq_len
        elif config.max_len is not None:
            self.max_seq_len = config.max_len
        else:
            raise ValueError(
                "max_seq_len must be provided either as argument "
                "or in model config (config.max_len)"
            )
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        head_dim = config.dim // config.n_heads

        if cache is not None:
            self._cache = cache
        else:
            self._cache = ContiguousCache(
                config.n_layers,
                max_batch_size,
                self.max_seq_len,
                config.n_kv_heads,
                head_dim,
                self.device,
                self.dtype,
            )

        self._task_mgr = TaskManager(
            tokenizer=tokenizer,
            max_batch_size=max_batch_size,
            max_seq_len=self.max_seq_len,
            max_prompt_len=max_prompt_len,
        )

        self._executor = Executor(
            model=model,
            tokenizer=tokenizer,
            kv_cache=self._cache,
            device=self.device,
            dtype=self.dtype,
        )

        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

    def add_task(self, prompt: str, **kwargs) -> str:
        return self._task_mgr.add_task(prompt, **kwargs)

    def remove_task(self, task_id: str):
        for task in self._task_mgr.remove_task(task_id):
            self._cache.task_free(task.task_id)

    def get_stats(self) -> Dict[str, Any]:
        return self._task_mgr.get_stats()

    def _run_generation_loop(self):
        stop_ids = self._task_mgr.tokenizer.stop_ids
        cache = self._cache
        try:
            while not self._stop_event.is_set():
                finished = self._task_mgr.remove_finished_tasks(stop_ids)
                for task in finished:
                    cache.task_free(task.task_id)

                active = self._task_mgr.get_active_tasks()
                available = self._task_mgr.max_batch_size - len(active)
                if available > 0:
                    candidates = self._task_mgr.pull_candidates(available)
                    failed = []
                    for task in candidates:
                        if cache.task_alloc(task.task_id, task.prompt_ids):
                            self._task_mgr.activate(task)
                        else:
                            failed.append(task)
                    if failed:
                        self._task_mgr.return_to_waiting(failed)

                if not self._task_mgr.has_work():
                    self._task_mgr.wait_for_tasks(timeout=1.0)
                    continue

                to_prefill = [
                    t
                    for t in self._task_mgr.get_active_tasks()
                    if t.output_tokens == 0
                    and cache.task_cached(t.task_id) < len(t.prompt_ids)
                ]
                if to_prefill:
                    for t in to_prefill:
                        t.input_tokens = len(t.prompt_ids)

                    groups: Dict[Tuple[int, int], List[Task]] = {}
                    for t in to_prefill:
                        key = (
                            len(t.prompt_ids),
                            cache.task_cached(t.task_id),
                        )
                        groups.setdefault(key, []).append(t)

                    for (prompt_len, start_pos), group in groups.items():
                        self._executor.execute_prefill(group, prompt_len, start_pos)
                        start_logical_page = start_pos // getattr(
                            cache, "page_size", 64
                        )
                        for t in group:
                            cache.task_record_hashes(
                                t.task_id, t.prompt_ids, start_logical_page
                            )

                decode_tasks = self._task_mgr.get_active_tasks()

                valid: List[Task] = []
                for t in sorted(decode_tasks, key=lambda t: t.task_id):
                    if cache.task_extend(t.task_id, t.next_pos):
                        valid.append(t)
                    else:
                        t.status = TaskStatus.ABORTED
                        self._task_mgr.invoke_callback(t.task_id, STOP)

                if valid:
                    next_tokens = self._executor.execute_decode(valid)

                    for t, ntok in zip(valid, next_tokens):
                        t.output_ids.append(ntok)
                        t.output_tokens += 1
                        new_text = t.decode_new_token(self._task_mgr.tokenizer)
                        if new_text:
                            self._task_mgr.invoke_callback(t.task_id, new_text)

                    for t in valid:
                        if t.is_finished(stop_ids):
                            remaining = t.flush_remaining(self._task_mgr.tokenizer)
                            if remaining:
                                self._task_mgr.invoke_callback(t.task_id, remaining)
                            self._task_mgr.invoke_callback(t.task_id, STOP)

        except Exception as e:
            self._stop_event.set()
            logger.error(f"Scheduler loop crashed: {e}", exc_info=True)
            for task in self._task_mgr.get_active_tasks():
                self._task_mgr.invoke_callback(task.task_id, STOP)
                cache.task_free(task.task_id)
            for task in self._task_mgr.get_waiting_tasks():
                self._task_mgr.invoke_callback(task.task_id, STOP)
            self._task_mgr.clear_queues()

    def start(self):
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._stop_event.clear()
        t = threading.Thread(target=self._run_generation_loop, daemon=True)
        t.start()
        self._loop_thread = t

    def stop(self):
        self._stop_event.set()
        self._task_mgr.wake()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
        self._loop_thread = None
        for task in self._task_mgr.get_active_tasks():
            self._task_mgr.invoke_callback(task.task_id, STOP)
            self._cache.task_free(task.task_id)
        for task in self._task_mgr.get_waiting_tasks():
            self._task_mgr.invoke_callback(task.task_id, STOP)
            self._cache.task_free(task.task_id)
        self._task_mgr.clear_queues()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_batch(
        self,
        prompt_ids_list: List[List[int]],
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        frequency_penalty: float = 0.0,
        rep_window: int = 64,
        return_logprobs: bool = False,
    ) -> List[List[int]]:
        """Synchronous batch generation without the scheduler thread.

        Accepts already-tokenized prompts (no string round-trip) and runs
        prefill + decode to completion on the calling thread.  Designed for
        RL rollout, where logprobs of the behaviour policy must be collected
        alongside generated tokens.

        Args:
            prompt_ids_list: ``B`` prompts, each a list of token IDs.
            max_tokens: Maximum tokens to generate per prompt.  ``None``
                uses ``self.max_seq_len - len(prompt_ids)``.
            temperature/top_p/top_k/frequency_penalty/rep_window: Sampling
                parameters (uniform across the batch).
            return_logprobs: If ``True``, return ``(token_ids, logprobs)``
                tuples per prompt (logprobs aligned 1-to-1 with token_ids).

        Returns:
            ``List[List[int]]`` of generated token IDs per prompt, or —
            when ``return_logprobs`` is ``True`` —
            ``List[Tuple[List[int], List[float]]]``.
        """
        stop_ids = self._task_mgr.tokenizer.stop_ids
        cache = self._cache
        seq_cap = self.max_seq_len

        tasks: List[Task] = []
        for ids in prompt_ids_list:
            if len(ids) >= seq_cap:
                tasks.append(None)
                continue
            t_max = max_tokens
            if t_max is None:
                t_max = seq_cap - len(ids)
            else:
                t_max = min(t_max, seq_cap - len(ids))
            task = Task(
                task_id=f"batch_{uuid.uuid4().hex[:8]}",
                prompt_ids=list(ids),
                max_tokens=t_max,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                rep_window=rep_window,
            )
            if not cache.task_alloc(task.task_id, task.prompt_ids):
                tasks.append(None)
                continue
            task.input_tokens = len(task.prompt_ids)
            tasks.append(task)

        try:
            live = [t for t in tasks if t is not None]
            prefill_groups: Dict[Tuple[int, int], List[Task]] = {}
            for t in live:
                key = (len(t.prompt_ids), cache.task_cached(t.task_id))
                prefill_groups.setdefault(key, []).append(t)
            for (prompt_len, start_pos), group in prefill_groups.items():
                self._executor.execute_prefill(group, prompt_len, start_pos)

            while live:
                valid: List[Task] = []
                for t in sorted(live, key=lambda x: x.task_id):
                    if cache.task_extend(t.task_id, t.next_pos):
                        valid.append(t)
                    else:
                        t.status = TaskStatus.ABORTED
                if not valid:
                    break

                step_out = self._executor.execute_decode(
                    valid, return_logprobs=return_logprobs
                )
                if return_logprobs:
                    for t, (ntok, _lp) in zip(valid, step_out):
                        t.output_ids.append(ntok)
                        t.output_tokens += 1
                else:
                    for t, ntok in zip(valid, step_out):
                        t.output_ids.append(ntok)
                        t.output_tokens += 1

                live = [t for t in valid if not t.is_finished(stop_ids)]
        finally:
            for t in tasks:
                if t is not None:
                    cache.task_free(t.task_id)

        results: List[Any] = []
        for t in tasks:
            if t is None:
                results.append(([], []) if return_logprobs else [])
            elif return_logprobs:
                results.append((list(t.output_ids), list(t.output_logprobs)))
            else:
                results.append(list(t.output_ids))
        return results
