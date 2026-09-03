import gc
import logging
import threading
import uuid
from contextlib import nullcontext
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from astrai.extension import (
    ATTN_BACKEND,
    AttentionBackend,
    attn_backend,
    get_backend,
)
from astrai.inference.cache import PagePool, TaskCacheManager
from astrai.inference.metrics import MetricsCollector
from astrai.inference.runtime.executor import Executor
from astrai.inference.task import (
    STOP,
    GenerationResult,
    Task,
    TaskManager,
    TaskStatus,
)
from astrai.model.automodel import AutoModel
from astrai.tokenize.tokenizer import AutoTokenizer

logger = logging.getLogger(__name__)


def _with_weight_lock(method):
    @wraps(method)
    def synchronized(self, *args, **kwargs):
        with self._weight_lock:
            return method(self, *args, **kwargs)

    return synchronized


class InferenceScheduler:
    """Continuous batching loop: cleanup -> refill -> prefill -> decode (all groups)."""

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 16,
        max_seq_len: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache: Optional[PagePool] = None,
        enable_cuda_graph: bool = True,
        backend: Optional[Union[str, ATTN_BACKEND, AttentionBackend, type]] = None,
        policy_version: int = 0,
    ):
        if (
            isinstance(policy_version, bool)
            or not isinstance(policy_version, int)
            or policy_version < 0
        ):
            raise ValueError("policy_version must be a non-negative integer")
        config = model.config
        self._model = model
        self._enable_cuda_graph = enable_cuda_graph
        self._owns_cache = cache is None
        self._released = False
        self._resume_running = False

        if max_seq_len is not None:
            self.max_seq_len = max_seq_len
        elif config.max_position_embeddings is not None:
            self.max_seq_len = config.max_position_embeddings
        else:
            raise ValueError(
                "max_seq_len must be provided either as argument "
                "or in model config (config.max_position_embeddings)"
            )
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        head_dim = config.hidden_size // config.num_attention_heads

        self._cache_spec = {
            "n_layers": config.num_hidden_layers,
            "n_kv_heads": config.num_key_value_heads,
            "head_dim": head_dim,
            "max_batch_size": max_batch_size,
            "max_seq_len": self.max_seq_len,
            "device": self.device,
            "dtype": self.dtype,
        }

        if cache is None:
            cache = PagePool(
                n_layers=config.num_hidden_layers,
                n_kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=self.max_seq_len,
                device=self.device,
                dtype=self.dtype,
            )
        self._cache = cache

        self._metrics = MetricsCollector()

        self._task_cache = TaskCacheManager(self._cache)

        self._task_mgr = TaskManager(
            tokenizer=tokenizer,
            max_batch_size=max_batch_size,
            max_seq_len=self.max_seq_len,
            metrics=self._metrics,
        )

        if backend is None:
            self._backend = None
            active_backend = get_backend()
        else:
            active_backend = backend
        with attn_backend(active_backend):
            if backend is not None:
                self._backend = get_backend()
            self._runtime_backend = get_backend()
            self._backend_name = type(self._runtime_backend).__name__
            self._executor = Executor(
                model=self._model,
                kv_cache=self._cache,
                task_cache=self._task_cache,
                device=self.device,
                dtype=self.dtype,
                enable_cuda_graph=enable_cuda_graph,
            )

        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None
        self._weight_lock = threading.RLock()
        self._policy_version = policy_version

    @property
    def policy_version(self) -> int:
        """Version of the model weights used for subsequent generations."""
        return self._policy_version

    @property
    def model(self) -> AutoModel:
        """Model whose weights survive inference-runtime release/resume cycles."""
        return self._model

    @property
    def runtime_released(self) -> bool:
        """Whether KV, workspace, and CUDA-graph runtime memory is released."""
        return self._released

    def _require_runtime(self) -> None:
        if self._released:
            raise RuntimeError("Inference runtime is released; call resume() first")

    @_with_weight_lock
    def update_weights(self, policy_version: int) -> int:
        """Acknowledge an in-place weight update and invalidate stale KV state.

        The scheduler owns the same model object as the in-process trainer, so
        weights have already changed when this method is called.  The explicit
        version update makes that lifecycle visible and prevents prefix KV
        entries produced by older weights from being reused.
        """
        if (
            isinstance(policy_version, bool)
            or not isinstance(policy_version, int)
            or policy_version < 0
        ):
            raise ValueError("policy_version must be a non-negative integer")
        if policy_version < self._policy_version:
            raise ValueError(
                f"policy_version cannot move backwards from "
                f"{self._policy_version} to {policy_version}"
            )
        if policy_version == self._policy_version:
            return self._policy_version
        if self._loop_thread is not None and self._loop_thread.is_alive():
            raise RuntimeError("Stop the scheduler before updating model weights")
        if self._task_mgr.get_active_tasks() or self._task_mgr.get_waiting_tasks():
            raise RuntimeError("Cannot update model weights while tasks are queued")

        if not self._released:
            self._task_cache.invalidate_cache()
        self._policy_version = policy_version
        return self._policy_version

    def add_task(self, prompt: str, **kwargs) -> str:
        self._require_runtime()
        return self._task_mgr.add_task(prompt, **kwargs)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a waiting or active task without freeing in-use KV state."""
        immediate, cancelled = self._task_mgr.cancel_task(task_id)
        for task in immediate:
            self._metrics.mark_finished(
                task.task_id, task.input_tokens, task.output_tokens
            )
        if cancelled:
            self._task_mgr.wake()
        return cancelled

    def remove_task(self, task_id: str) -> bool:
        """Backward-compatible alias for cancellation."""
        return self.cancel_task(task_id)

    def get_stats(self) -> Dict[str, Any]:
        stats = self._task_mgr.get_stats()
        stats["kv_cache_tasks"] = (
            0 if self._task_cache is None else self._task_cache.task_count
        )
        stats["policy_version"] = self._policy_version
        stats["runtime_released"] = self._released
        return stats

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def cuda_graph_enabled(self) -> bool:
        return self._executor is not None and self._executor.cuda_graph_enabled

    def _backend_context(self):
        if self._backend is None:
            return nullcontext()
        return attn_backend(self._backend)

    @staticmethod
    def _task_backend_groups(tasks: List[Task]):
        groups = {}
        for task in tasks:
            groups.setdefault(task.backend, (task.backend, []))[1].append(task)
        return groups.values()

    def _step(
        self, tasks: List[Task], return_logprobs: bool = False
    ) -> Tuple[List[Task], List[Task]]:
        """Advance every active task by one token (prefill + decode).

        Single shared primitive for both the continuous-batching loop and
        the synchronous ``run_batch`` path, so the two cannot drift.

        Tasks must already be allocated in the KV cache. Tasks without output
        are prefilled first and sample their first token from the final prompt
        position. Tasks with output extend the cache by one position and decode
        from their latest generated token.

        Args:
            tasks: Active tasks to advance by one token.
            return_logprobs: Forwarded to ``execute_decode``; per-token
                logprobs are recorded on each task's ``output_logprobs``.

        Returns:
            ``(decoded, aborted)``: tasks that produced a new token (its ID
            already appended to ``output_ids``) and tasks that hit the
            sequence cap and were marked ``ABORTED``.
        """
        to_prefill = [t for t in tasks if not t.prefill_done and t.prompt_ids]
        prefilled_ids = set()
        produced: List[Task] = []
        if to_prefill:
            for t in to_prefill:
                t.input_tokens = len(t.prompt_ids)

            groups: Dict[Tuple[int, Optional[AttentionBackend]], List[Task]] = {}
            for t in to_prefill:
                start_pos = min(
                    self._task_cache.task_cached(t.task_id), len(t.prompt_ids) - 1
                )
                groups.setdefault((start_pos, t.backend), []).append(t)

            for (start_pos, _), group in groups.items():
                backend = group[0].backend
                backend_context = (
                    attn_backend(backend) if backend is not None else nullcontext()
                )
                with (
                    backend_context,
                    self._metrics.record([t.task_id for t in group], "prefill"),
                ):
                    prefilled, step_out = self._executor.execute_prefill(
                        group, start_pos=start_pos, return_logprobs=return_logprobs
                    )

                for t, out in zip(prefilled, step_out):
                    t.output_ids.append(out[0] if return_logprobs else out)
                    t.output_tokens += 1
                    t.mark_prefill_done()
                    prefilled_ids.add(t.task_id)
                    produced.append(t)

                start_logical_page = start_pos // self._cache.page_size
                for t in group:
                    self._task_cache.task_record_hashes(
                        t.task_id, t.prompt_ids, start_logical_page
                    )

        decoded: List[Task] = []
        aborted: List[Task] = []
        for t in tasks:
            if t.task_id in prefilled_ids:
                continue
            if self._task_cache.task_extend(t.task_id, t.next_pos):
                decoded.append(t)
            else:
                t.status = TaskStatus.ABORTED
                aborted.append(t)

        for backend, group in self._task_backend_groups(decoded):
            backend_context = (
                attn_backend(backend) if backend is not None else nullcontext()
            )
            with (
                backend_context,
                self._metrics.record([t.task_id for t in group], "decode"),
            ):
                step_out = self._executor.execute_decode(
                    group, return_logprobs=return_logprobs
                )
            for t, out in zip(group, step_out):
                t.output_ids.append(out[0] if return_logprobs else out)
                t.output_tokens += 1
                t.advance_kv()
                produced.append(t)

        return produced, aborted

    def _run_generation_loop(self):
        stop_ids = self._task_mgr.tokenizer.stop_ids
        try:
            with self._backend_context():
                while not self._stop_event.is_set():
                    finished = self._task_mgr.remove_finished_tasks(stop_ids)
                    for task in finished:
                        if task.status == TaskStatus.FINISHED:
                            self._task_cache.task_record_hashes(
                                task.task_id,
                                self._task_cache.task_cacheable_ids(
                                    task.task_id, task.prompt_ids, task.output_ids
                                ),
                            )
                        self._task_cache.task_free(task.task_id)

                    active = self._task_mgr.get_active_tasks()
                    available = self._task_mgr.max_batch_size - len(active)
                    if available > 0:
                        candidates = self._task_mgr.pull_candidates(available)
                        failed = []
                        for task in candidates:
                            if self._task_cache.task_alloc(
                                task.task_id, task.prompt_ids
                            ):
                                if not self._task_mgr.activate(task):
                                    self._task_cache.task_free(task.task_id)
                                    self._metrics.mark_finished(
                                        task.task_id,
                                        task.input_tokens,
                                        task.output_tokens,
                                    )
                            else:
                                failed.append(task)
                        if failed:
                            self._task_mgr.return_to_waiting(failed)

                    if not self._task_mgr.has_work():
                        self._task_mgr.wait_for_tasks(timeout=1.0)
                        continue

                    active = [
                        task
                        for task in self._task_mgr.get_active_tasks()
                        if task.status != TaskStatus.ABORTED
                    ]

                    decoded, aborted = self._step(active)

                    for t in aborted:
                        self._task_mgr.invoke_callback(t.task_id, STOP)

                    for t in decoded:
                        if t.status == TaskStatus.ABORTED:
                            continue
                        new_text = t.decode_new_token(self._task_mgr.tokenizer)
                        if new_text:
                            self._task_mgr.invoke_callback(t.task_id, new_text)
                        if t.is_finished(stop_ids):
                            self._task_mgr.invoke_callback(t.task_id, STOP)

        except Exception as e:
            self._stop_event.set()
            logger.error(f"Scheduler loop crashed: {e}", exc_info=True)
            self._abort_and_clear(free_waiting=False)

    def start(self):
        self._require_runtime()
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
        self._abort_and_clear(free_waiting=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @_with_weight_lock
    def release(self) -> bool:
        """Release inference-only GPU memory while preserving model weights.

        The scheduler must own its cache (the default).  Releasing an injected
        cache would have ambiguous ownership and could invalidate another
        consumer, so that configuration fails before changing scheduler state.

        Active and waiting requests are cancelled using the normal ``stop``
        path.  Prefix-cache entries, KV storage, decode workspace, and captured
        CUDA graphs are then dropped.  ``resume`` reconstructs them from the
        original bounds and restarts the loop when it was running before the
        release.

        Returns ``True`` when resources were released and ``False`` when the
        scheduler was already released.
        """
        if self._released:
            return False
        if not self._owns_cache:
            raise RuntimeError(
                "Cannot release a scheduler backed by an externally owned cache"
            )

        self._resume_running = (
            self._loop_thread is not None and self._loop_thread.is_alive()
        )
        loop_thread = self._loop_thread
        self.stop()
        if loop_thread is not None and loop_thread.is_alive():
            self._loop_thread = loop_thread
            raise RuntimeError("Scheduler loop did not stop; runtime remains resident")
        self._task_cache.invalidate_cache()

        if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        self._executor = None
        self._task_cache = None
        self._cache = None
        self._released = True
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    @_with_weight_lock
    def resume(self) -> bool:
        """Recreate released inference memory and restore the prior loop state.

        Returns ``True`` when resources were reconstructed and ``False`` when
        the runtime was already resident.
        """
        if not self._released:
            return False

        cache = PagePool(**self._cache_spec)
        task_cache = TaskCacheManager(cache)
        with attn_backend(self._runtime_backend):
            executor = Executor(
                model=self._model,
                kv_cache=cache,
                task_cache=task_cache,
                device=self.device,
                dtype=self.dtype,
                enable_cuda_graph=self._enable_cuda_graph,
            )

        self._cache = cache
        self._task_cache = task_cache
        self._executor = executor
        self._released = False
        restart = self._resume_running
        self._resume_running = False
        if restart:
            self.start()
        return True

    def _abort_and_clear(self, free_waiting: bool):
        """Invoke STOP callbacks, release cache slots, and clear task queues."""
        active = self._task_mgr.get_active_tasks()
        waiting = self._task_mgr.get_waiting_tasks()
        for task in active:
            self._task_mgr.invoke_callback(task.task_id, STOP)
            if self._task_cache is not None:
                self._task_cache.task_free(task.task_id)
            self._metrics.mark_finished(
                task.task_id, task.input_tokens, task.output_tokens
            )
        for task in waiting:
            self._task_mgr.invoke_callback(task.task_id, STOP)
            if free_waiting:
                if self._task_cache is not None:
                    self._task_cache.task_free(task.task_id)
            self._metrics.mark_finished(
                task.task_id, task.input_tokens, task.output_tokens
            )
        self._task_mgr.clear_queues()

    @_with_weight_lock
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
        return_details: bool = False,
    ) -> List[Any]:
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
            return_details: If ``True``, return a structured result per prompt
                with terminal and error reasons. Logprobs are populated when
                ``return_logprobs`` is also ``True``.

        Returns:
            Structured results when ``return_details`` is ``True``;
            otherwise generated token IDs per prompt, or token/logprob tuples
            when ``return_logprobs`` is ``True``.
        """
        self._require_runtime()
        stop_ids = self._task_mgr.tokenizer.stop_ids
        seq_cap = self.max_seq_len
        request_backend = get_backend(use_default=False)

        tasks: List[Optional[Task]] = []
        error_reasons: List[Optional[str]] = []
        for ids in prompt_ids_list:
            if not ids:
                tasks.append(None)
                error_reasons.append("prompt_empty")
                continue
            if len(ids) >= seq_cap:
                tasks.append(None)
                error_reasons.append("prompt_too_long")
                continue
            t_max = max_tokens
            if t_max is None:
                t_max = seq_cap - len(ids)
            else:
                t_max = min(t_max, seq_cap - len(ids))
            if t_max <= 0:
                tasks.append(None)
                error_reasons.append("max_tokens_non_positive")
                continue
            task = Task(
                task_id=f"batch_{uuid.uuid4().hex[:8]}",
                prompt_ids=list(ids),
                max_tokens=t_max,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                rep_window=rep_window,
                backend=request_backend,
            )
            if not self._task_cache.task_alloc(task.task_id, task.prompt_ids):
                tasks.append(None)
                error_reasons.append("kv_cache_allocation_failed")
                continue
            task.input_tokens = len(task.prompt_ids)
            self._metrics.register(task.task_id)
            tasks.append(task)
            error_reasons.append(None)

        runtime_errors: Dict[str, str] = {}
        try:
            live = [t for t in tasks if t is not None]

            with self._backend_context():
                while live:
                    decoded, aborted = self._step(live, return_logprobs=return_logprobs)
                    for task in aborted:
                        runtime_errors[task.task_id] = "kv_cache_extension_failed"
                    live = [t for t in decoded if not t.is_finished(stop_ids)]
        finally:
            for t in tasks:
                if t is not None:
                    self._metrics.mark_finished(
                        t.task_id, t.input_tokens, t.output_tokens
                    )
                    self._task_cache.task_free(t.task_id)

        details: List[GenerationResult] = []
        for t, setup_error in zip(tasks, error_reasons):
            if t is None:
                details.append(
                    GenerationResult(
                        token_ids=[],
                        logprobs=[],
                        finish_reason="rejected",
                        error_reason=setup_error,
                    )
                )
            else:
                runtime_error = runtime_errors.get(t.task_id)
                stopped = bool(t.output_ids and t.output_ids[-1] in stop_ids)
                if runtime_error:
                    finish_reason = "rejected"
                elif stopped:
                    finish_reason = "stop"
                else:
                    finish_reason = "length"
                details.append(
                    GenerationResult(
                        token_ids=list(t.output_ids),
                        logprobs=list(t.output_logprobs),
                        finish_reason=finish_reason,
                        error_reason=runtime_error,
                    )
                )

        if return_details:
            return details
        if return_logprobs:
            return [(result.token_ids, result.logprobs) for result in details]
        return [result.token_ids for result in details]
