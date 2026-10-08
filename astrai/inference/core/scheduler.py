"""Request scheduling and result application; EngineCore owns execution."""

import logging
import uuid
from collections import deque
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import torch

from astrai.config.inference_config import InferenceConfig
from astrai.extension import ATTN_BACKEND, AttentionBackend, attn_backend, get_backend
from astrai.inference.contracts import ModelRunnerOutput, SchedulerOutput
from astrai.inference.core.cache.pool import BlockPool, KVCacheManager
from astrai.inference.core.engine_core import EngineCore
from astrai.inference.core.events import (
    FINISH_ABORTED,
    FINISH_CANCELLED,
    FINISH_LENGTH,
    FINISH_REJECTED,
    FINISH_STOP_TOKEN,
    RequestError,
    RequestFinished,
    TokenDelta,
)
from astrai.inference.core.metrics import MetricsCollector
from astrai.inference.core.request import (
    GenerationResult,
    Request,
    RequestManager,
    RequestStatus,
)
from astrai.inference.core.stepper import SchedulerStep
from astrai.inference.core.versioning import PolicyVersionGuard
from astrai.inference.worker.model_runner import GPUModelRunner
from astrai.model.automodel import AutoModel
from astrai.tokenize.tokenizer import AutoTokenizer

logger = logging.getLogger(__name__)
T = TypeVar("T")
_config = InferenceConfig()


def _with_weight_lock(method):
    @wraps(method)
    def synchronized(self, *args, **kwargs):
        with self._weight_lock:
            return method(self, *args, **kwargs)

    return synchronized


def _synchronous(method):
    @wraps(method)
    def synchronized(self, *args, **kwargs):
        with self.engine_core.exclusive():
            return method(self, *args, **kwargs)

    return synchronized


class OutputEventSink:
    """Fast, exception-safe outlet for token and exactly-once terminal events."""

    def __call__(self, events: List[Any]) -> None:
        raise NotImplementedError


class Scheduler:
    """Own request/KV scheduling state, not the thread or GPU execution loop."""

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 16,
        max_seq_len: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache: Optional[BlockPool] = None,
        enable_cuda_graph: bool = True,
        backend: Optional[Union[str, ATTN_BACKEND, AttentionBackend, type]] = None,
        policy_version: int = 0,
        enable_overlap: bool = False,
        page_size: Optional[int] = None,
        kv_tokens: Optional[int] = None,
        token_budget: Optional[int] = None,
    ):
        if (
            isinstance(policy_version, bool)
            or not isinstance(policy_version, int)
            or policy_version < 0
        ):
            raise ValueError("policy_version must be a non-negative integer")
        config = model.config
        self.max_seq_len = (
            max_seq_len if max_seq_len is not None else config.max_position_embeddings
        )
        if self.max_seq_len is None:
            raise ValueError("max_seq_len must be provided as argument or model config")
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype
        self._enable_overlap = enable_overlap
        if cache is None:
            pool_kwargs = {}
            if page_size is not None:
                pool_kwargs["page_size"] = page_size
            if kv_tokens is not None:
                pool_kwargs["n_tokens"] = kv_tokens
            cache = BlockPool(
                n_layers=config.num_hidden_layers,
                n_kv_heads=config.num_key_value_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                max_batch_size=max_batch_size,
                max_seq_len=self.max_seq_len,
                device=self.device,
                dtype=self.dtype,
                **pool_kwargs,
            )
        self._cache = cache
        self._metrics = MetricsCollector()
        self._kv_manager = KVCacheManager(cache)
        self._requests = RequestManager(
            tokenizer, max_batch_size, self.max_seq_len, self._metrics
        )
        self._stop_ids = frozenset(tokenizer.stop_ids)
        self._backend = None
        with attn_backend(backend if backend is not None else get_backend()):
            if backend is not None:
                self._backend = get_backend()
            self._backend_name = type(get_backend()).__name__
            self._executor = GPUModelRunner(
                model=model,
                kv_cache=cache,
                cache_mgr=self._kv_manager,
                device=self.device,
                dtype=self.dtype,
                enable_cuda_graph=enable_cuda_graph,
            )
        effective_budget = token_budget or _config.max_num_batched_tokens or None
        self._stepper = SchedulerStep(self, token_budget=effective_budget)
        self._event_sink: Optional[OutputEventSink] = None
        self._states: Dict[str, Request] = {}
        self._step_id = 0
        # Keys are (step id, request incarnation), never mutable object identity.
        self._planned = {}
        self._pending_order = {}
        self._ready = {}
        self._policy_guard = PolicyVersionGuard(
            policy_version,
            ensure_ready=self._ensure_weight_update_ready,
            on_commit=self._kv_manager.invalidate_cache,
        )
        self._weight_lock = self._policy_guard.lock
        self._engine_core = EngineCore(self, self._weight_lock)

    @property
    def engine_core(self) -> EngineCore:
        return self._engine_core

    @property
    def _loop_thread(self):
        return self.engine_core.loop_thread

    @property
    def _stop_event(self):
        return self.engine_core.stop_event

    @property
    def stop_ids(self):
        return self._stop_ids

    @property
    def policy_version(self):
        return self._policy_guard.policy_version

    @property
    def model(self):
        return self._executor.model

    @property
    def max_batch_size(self):
        return self._requests.max_batch_size

    @property
    def backend_name(self):
        return self._backend_name

    @property
    def cuda_graph_enabled(self):
        return self._executor.cuda_graph_enabled

    def _backend_context(self):
        return (
            attn_backend(self._backend) if self._backend is not None else nullcontext()
        )

    def set_event_sink(self, sink):
        with self._weight_lock:
            self._event_sink = sink

    def _emit_events(self, events):
        if not events or self._event_sink is None:
            return
        try:
            self._event_sink(events)
        except Exception:
            logger.exception("output event sink failed")

    def _ensure_weight_update_ready(self):
        self.engine_core.ensure_weight_update_ready()

    def update_weights(self, policy_version: int) -> int:
        return self._policy_guard.update_weights(policy_version)

    def apply_weight_update(
        self, policy_version: Optional[int], update: Callable[[int], T]
    ) -> T:
        return self._policy_guard.apply_weight_update(policy_version, update)

    def with_policy_snapshot(self, inspect: Callable[[int], T]) -> T:
        return self._policy_guard.with_policy_snapshot(inspect)

    def _remember(self, request):
        current = self._states.get(request.request_id)
        if current is not None and current is not request:
            raise ValueError(f"duplicate live request id: {request.request_id}")
        self._states[request.request_id] = request
        request.input_tokens = len(request.prompt_ids)

    @_with_weight_lock
    def add_request(self, prompt: str, **kwargs) -> str:
        self.engine_core.ensure_accepting()
        rid = self._requests.add_request(prompt, **kwargs)
        request = self._requests.get_request(rid)
        self._remember(request)
        if request.max_tokens is not None and request.max_tokens <= 0:
            self.finish(request, FINISH_LENGTH)
            self.release_finished()
        return rid

    @_with_weight_lock
    def add_requests(self, prompts: List[str], **kwargs) -> List[str]:
        self.engine_core.ensure_accepting()
        ids = self._requests.add_requests(prompts, **kwargs)
        for rid in ids:
            request = self._requests.get_request(rid)
            self._remember(request)
            if request.max_tokens is not None and request.max_tokens <= 0:
                self.finish(request, FINISH_LENGTH)
        self.release_finished()
        return ids

    @_with_weight_lock
    def cancel_request(self, request_id: str) -> bool:
        request = self._states.get(request_id)
        if request is None or request.terminal_emitted:
            return False
        _, cancelled = self._requests.cancel_request(request_id)
        if cancelled:
            self.finish(request, FINISH_CANCELLED)
            self.release_finished()
            self._requests.wake()
        return cancelled

    @_with_weight_lock
    def get_stats(self):
        stats = self._requests.get_stats()
        stats["kv_cache_tasks"] = self._kv_manager.request_count
        stats["policy_version"] = self.policy_version
        return stats

    def finish_reason(self, request):
        return (
            FINISH_STOP_TOKEN
            if request.output_ids and request.output_ids[-1] in self.stop_ids
            else FINISH_LENGTH
        )

    def finish(self, request, reason, *, error_reason=None, error=None, events=None):
        if request.terminal_emitted:
            return
        request.terminal_emitted = True
        request.finish_reason = reason
        request.error_reason = error_reason
        request.status = (
            RequestStatus.FINISHED
            if reason in (FINISH_LENGTH, FINISH_STOP_TOKEN) and error is None
            else RequestStatus.ABORTED
        )
        if not request.emit_events:
            return
        event = (
            RequestError(request.request_id, "execution_failed", str(error))
            if error is not None
            else RequestFinished(
                request.request_id, reason, request.input_tokens, request.output_tokens
            )
        )
        if events is None:
            self._emit_events([event])
        else:
            events.append(event)

    def abort_all(self, reason, error=None):
        events = []
        for request in list(self._states.values()):
            self.finish(request, reason, error=error, events=events)
        self._emit_events(events)

    def release_finished(self):
        if self.engine_core._shutdown_failed:
            return
        for rid, request in list(self._states.items()):
            if not request.terminal_emitted or self._pending_order.get(
                request.identity
            ):
                continue
            if (
                request.status == RequestStatus.FINISHED
                and request.num_materialized_tokens
            ):
                self._kv_manager.record_block_hashes(
                    rid,
                    request.prompt_ids + request.output_ids,
                    materialized_end=request.num_materialized_tokens,
                )
            self._kv_manager.free_slots(rid)
            self._metrics.mark_finished(
                rid, request.input_tokens, request.output_tokens
            )
            self._requests.discard(request)
            self._states.pop(rid, None)

    def active_requests(self):
        return [
            r for r in self._requests.get_running_requests() if not r.terminal_emitted
        ]

    def admit_requests(self):
        available = self.max_batch_size - len(self._requests.get_running_requests())
        if available <= 0:
            return
        failed = []
        for request in self._requests.pull_waiting(available):
            if request.terminal_emitted:
                continue
            if not self._kv_manager.can_ever_fit(len(request.prompt_ids)):
                self.finish(request, FINISH_REJECTED)
            elif self._kv_manager.alloc_slots(request.request_id, request.prompt_ids):
                if not self._requests.activate(request):
                    self.finish(request, FINISH_CANCELLED)
            else:
                failed.append(request)
        if failed:
            self._requests.return_to_waiting(failed)
        self.release_finished()

    @_with_weight_lock
    def schedule(self, requests=None, *, return_logprobs=False) -> SchedulerOutput:
        requests = self.active_requests() if requests is None else requests
        for request in requests:
            self._remember(request)
        entries = self._stepper.plan(requests)
        self._step_id += 1
        plan = SchedulerOutput(
            self._step_id, self.policy_version, tuple(entries), return_logprobs
        )
        for entry in entries:
            key = (plan.step_id, entry.identity)
            self._planned[key] = (plan.policy_version, entry)
            self._pending_order.setdefault(entry.identity, deque()).append(plan.step_id)
            self._states[entry.request_id].num_computed_tokens = entry.materialized_end
        return plan

    @_with_weight_lock
    def update_from_output(self, output: ModelRunnerOutput):
        """Apply identity-addressed rows in per-request step order, exactly once."""
        identities = []
        seen = set()
        for result in output.results:
            identity = result.identity
            key = (output.step_id, identity)
            planned = self._planned.get(key)
            if planned is None or output.policy_version != planned[0]:
                continue
            self._ready.setdefault(key, result)
            if identity not in seen:
                seen.add(identity)
                identities.append(identity)
        events, produced = [], []
        for identity in identities:
            order = self._pending_order.get(identity)
            while order:
                key = (order[0], identity)
                result = self._ready.get(key)
                if result is None:
                    break
                version, entry = self._planned[key]
                if result.materialized_end != entry.materialized_end:
                    raise RuntimeError(
                        "worker materialization does not match scheduled window"
                    )
                if entry.samples != (result.token_id is not None):
                    raise RuntimeError(
                        "worker sampling rows do not match scheduled window"
                    )
                self._ready.pop(key)
                self._planned.pop(key)
                order.popleft()
                request = self._states.get(identity.request_id)
                if request is None or request.identity != identity:
                    continue
                if version != self.policy_version:
                    self.finish(
                        request,
                        FINISH_ABORTED,
                        error_reason="stale_execution",
                        events=events,
                    )
                    continue
                request.num_materialized_tokens = max(
                    request.num_materialized_tokens, result.materialized_end
                )
                if request.terminal_emitted:
                    continue
                if entry.phase == "prefill":
                    self._kv_manager.record_block_hashes(
                        request.request_id,
                        request.prompt_ids,
                        entry.position // self._cache.page_size,
                        materialized_end=result.materialized_end,
                    )
                if result.token_id is not None:
                    request.output_ids.append(result.token_id)
                    request.output_tokens += 1
                    if result.logprob is not None:
                        request.output_logprobs.append(result.logprob)
                    produced.append(request)
                    if request.emit_events:
                        events.append(
                            TokenDelta(
                                request.request_id,
                                result.token_id,
                                request.output_tokens,
                            )
                        )
                    if request.is_finished(self.stop_ids):
                        self.finish(request, self.finish_reason(request), events=events)
            if not order:
                self._pending_order.pop(identity, None)
        self._emit_events(events)
        return produced

    def rollback_schedule(self, plan):
        """Undo only an unlaunched plan; allocated capacity can be reused.

        Older in-flight steps keep their slots and optimistic watermarks. This
        is required before re-planning a shrunken overlap batch after a drain.
        """
        for entry in plan.requests:
            key = (plan.step_id, entry.identity)
            self._planned.pop(key, None)
            self._ready.pop(key, None)
            order = self._pending_order.get(entry.identity)
            if order is not None:
                order.remove(plan.step_id)
                if not order:
                    self._pending_order.pop(entry.identity, None)
            request = self._states.get(entry.request_id)
            if request is not None and request.identity == entry.identity:
                request.num_computed_tokens = max(
                    [request.num_materialized_tokens]
                    + [
                        self._planned[(step, entry.identity)][1].materialized_end
                        for step in order or ()
                    ]
                )

    def discard_outstanding(self):
        self._planned.clear()
        self._pending_order.clear()
        self._ready.clear()
        for request in self._states.values():
            request.num_computed_tokens = request.num_materialized_tokens

    def run_busy_loop(self):
        return self.engine_core.run_busy_loop()

    def start(self):
        self._stop_ids = frozenset(self._requests.tokenizer.stop_ids)
        return self.engine_core.start()

    def stop(self):
        return self.engine_core.stop()

    @_synchronous
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
        self._stop_ids = frozenset(self._requests.tokenizer.stop_ids)
        requests, errors = [], []
        backend = get_backend(use_default=False)
        batch_id = uuid.uuid4().hex
        for index, ids in enumerate(prompt_ids_list):
            error = None
            if not ids:
                error = "prompt_empty"
            elif len(ids) >= self.max_seq_len:
                error = "prompt_too_long"
            limit = (
                self.max_seq_len - len(ids)
                if max_tokens is None
                else min(max_tokens, self.max_seq_len - len(ids))
            )
            if error is None and limit <= 0:
                error = "max_tokens_non_positive"
            request = None
            if error is None:
                request = Request(
                    f"batch_{batch_id}_{index:08d}",
                    ids,
                    limit,
                    temperature,
                    top_p,
                    top_k,
                    frequency_penalty,
                    rep_window,
                    backend,
                )
                if not self._kv_manager.alloc_slots(
                    request.request_id, request.prompt_ids
                ):
                    request, error = None, "kv_cache_allocation_failed"
                else:
                    request.emit_events = False
                    self._remember(request)
                    self._metrics.register(request.request_id)
            requests.append(request)
            errors.append(error)
        try:
            live = [r for r in requests if r is not None]
            with self._backend_context():
                while live:
                    decoded, aborted = self._stepper.step(
                        live, return_logprobs=return_logprobs
                    )
                    for request in aborted:
                        self.finish(
                            request,
                            FINISH_ABORTED,
                            error_reason="kv_cache_extension_failed",
                        )
                        request.error_reason = (
                            request.error_reason or "kv_cache_extension_failed"
                        )
                    live = [
                        r
                        for r in decoded
                        if not r.terminal_emitted and not r.is_finished(self.stop_ids)
                    ]
        finally:
            self.engine_core.drain()
            for request in requests:
                if request is not None and not request.terminal_emitted:
                    self.finish(
                        request, FINISH_ABORTED, error_reason="execution_failed"
                    )
            self.release_finished()
        details = []
        for request, error in zip(requests, errors):
            if request is None:
                details.append(GenerationResult([], [], "rejected", error))
            else:
                reason = (
                    "rejected"
                    if request.error_reason
                    else (
                        "stop"
                        if request.finish_reason == FINISH_STOP_TOKEN
                        else "length"
                    )
                )
                details.append(
                    GenerationResult(
                        list(request.output_ids),
                        list(request.output_logprobs),
                        reason,
                        request.error_reason,
                    )
                )
        if return_details:
            return details
        if return_logprobs:
            return [(r.token_ids, r.logprobs) for r in details]
        return [r.token_ids for r in details]

    @_synchronous
    def score_ids(self, prompt_ids_list, continuation_ids_list, per_token=False):
        if len(prompt_ids_list) != len(continuation_ids_list):
            raise ValueError("prompt and continuation lists must have equal length")
        requests = []
        backend = get_backend(use_default=False)
        for prompt_ids, cont_ids in zip(prompt_ids_list, continuation_ids_list):
            if (
                not prompt_ids
                or not cont_ids
                or len(prompt_ids) + len(cont_ids) > self.max_seq_len
            ):
                requests.append(None)
                continue
            request = Request(
                f"score_{uuid.uuid4().hex}",
                list(prompt_ids) + list(cont_ids),
                max_tokens=0,
                backend=backend,
            )
            request.cont_len = len(cont_ids)
            if not self._kv_manager.alloc_slots(request.request_id, request.prompt_ids):
                requests.append(None)
                continue
            request.emit_events = False
            self._remember(request)
            requests.append(request)
        live = [r for r in requests if r is not None]
        results = {}
        try:
            if live:
                with self._backend_context():
                    scored = self._executor.execute_score(
                        [r.execution("score", 0, len(r.prompt_ids)) for r in live],
                        per_token=per_token,
                    )
                results = {r.request_id: value for r, value in zip(live, scored)}
        finally:
            # Score has no PendingExecution, including when forward throws.
            self.engine_core.fence()
            for request in live:
                self._kv_manager.free_slots(request.request_id)
                self._states.pop(request.request_id, None)
        return [results.get(r.request_id) if r is not None else None for r in requests]
