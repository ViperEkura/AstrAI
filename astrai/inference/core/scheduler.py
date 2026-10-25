"""Request state, KV accounting, scheduling, and result application."""

import logging
from collections import deque
from typing import Any, Dict, List, Optional

from astrai.config.inference_config import InferenceConfig
from astrai.inference.contracts import ModelRunnerOutput, SchedulerOutput
from astrai.inference.core.cache.pool import BlockPool, KVCacheManager
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
from astrai.inference.core.planner import StepPlanner
from astrai.inference.core.request import Request, RequestManager, RequestStatus

logger = logging.getLogger(__name__)
_config = InferenceConfig()


class OutputEventSink:
    """Fast, exception-safe outlet for token and terminal events."""

    def __call__(self, events: List[Any]) -> None:
        raise NotImplementedError


class Scheduler:
    """Own request and logical KV state; never launch model execution."""

    def __init__(
        self,
        tokenizer,
        max_batch_size: int,
        max_seq_len: int,
        cache: BlockPool,
        token_budget: Optional[int] = None,
        policy_version: int = 0,
    ):
        self.max_seq_len = max_seq_len
        self._cache = cache
        self._metrics = MetricsCollector()
        self._kv_manager = KVCacheManager(cache)
        self._requests = RequestManager(
            tokenizer, max_batch_size, max_seq_len, self._metrics
        )
        self._stop_ids = frozenset(tokenizer.stop_ids)
        effective_budget = token_budget or _config.max_num_batched_tokens or None
        self._planner = StepPlanner(self._kv_manager, effective_budget)
        self._event_sink: Optional[OutputEventSink] = None
        self._states: Dict[str, Request] = {}
        self._step_id = 0
        self._policy_version = policy_version
        self._planned = {}
        self._pending_order = {}
        self._ready = {}

    @property
    def stop_ids(self):
        return self._stop_ids

    @property
    def max_batch_size(self):
        return self._requests.max_batch_size

    @property
    def kv_manager(self):
        return self._kv_manager

    @property
    def cache(self):
        return self._cache

    @property
    def metrics(self):
        return self._metrics

    def has_requests(self):
        return self._requests.has_requests()

    def wait_for_requests(self, timeout):
        return self._requests.wait_for_requests(timeout=timeout)

    def wake(self):
        self._requests.wake()

    def has_outstanding(self):
        return bool(self._planned)

    def set_policy_version(self, policy_version):
        self._policy_version = policy_version

    def refresh_stop_ids(self):
        self._stop_ids = frozenset(self._requests.tokenizer.stop_ids)

    def set_event_sink(self, sink):
        self._event_sink = sink

    def _emit_events(self, events):
        if not events or self._event_sink is None:
            return
        try:
            self._event_sink(events)
        except Exception:
            logger.exception("output event sink failed")

    def remember(self, request):
        current = self._states.get(request.request_id)
        if current is not None and current is not request:
            raise ValueError(f"duplicate live request id: {request.request_id}")
        self._states[request.request_id] = request
        request.input_tokens = len(request.prompt_ids)

    def add_request(self, prompt: str, **kwargs) -> str:
        rid = self._requests.add_request(prompt, **kwargs)
        request = self._requests.get_request(rid)
        self.remember(request)
        if request.max_tokens is not None and request.max_tokens <= 0:
            self.finish(request, FINISH_LENGTH)
            self.release_finished()
        return rid

    def add_requests(self, prompts: List[str], **kwargs) -> List[str]:
        ids = self._requests.add_requests(prompts, **kwargs)
        for rid in ids:
            request = self._requests.get_request(rid)
            self.remember(request)
            if request.max_tokens is not None and request.max_tokens <= 0:
                self.finish(request, FINISH_LENGTH)
        self.release_finished()
        return ids

    def enqueue_tokenized_requests(self, requests: List[Request]) -> None:
        self._requests.enqueue_tokenized_requests(requests)
        for request in requests:
            request.emit_events = False
            self.remember(request)

    def reject_waiting_for_capacity(self) -> None:
        """End a synchronous batch when no active request can free capacity."""
        waiting = self._requests.pull_waiting(self._requests.get_stats()["waiting"])
        for request in waiting:
            self.finish(
                request,
                FINISH_REJECTED,
                error_reason="kv_cache_allocation_failed",
            )
        self._requests.discard_waiting(waiting)
        self.release_finished()

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

    def get_stats(self):
        stats = self._requests.get_stats()
        stats["kv_cache_tasks"] = self._kv_manager.request_count
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

    def release_finished(self, safe_to_release=True):
        if not safe_to_release:
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

    def admit_requests(self, safe_to_release=True):
        available = self.max_batch_size - len(self._requests.get_running_requests())
        if available <= 0:
            return
        failed = []
        for request in self._requests.pull_waiting(available):
            if request.terminal_emitted:
                continue
            if not self._kv_manager.can_ever_fit(len(request.prompt_ids)):
                self.finish(
                    request,
                    FINISH_REJECTED,
                    error_reason="kv_cache_allocation_failed",
                )
            elif self._kv_manager.alloc_slots(request.request_id, request.prompt_ids):
                if not self._requests.activate(request):
                    self.finish(request, FINISH_CANCELLED)
            else:
                failed.append(request)
        if failed:
            self._requests.return_to_waiting(failed)
        self.release_finished(safe_to_release)

    def schedule(
        self,
        requests=None,
        policy_version=None,
        *,
        return_logprobs=False,
        defer_kv_failures=False,
    ) -> SchedulerOutput:
        if policy_version is None:
            policy_version = self._policy_version
        requests = self.active_requests() if requests is None else requests
        eligible = []
        for request in requests:
            self.remember(request)
            if request.terminal_emitted:
                continue
            request.input_tokens = len(request.prompt_ids)
            if request.is_finished(self.stop_ids):
                self.finish(request, self.finish_reason(request))
            elif request.prefill_complete and request.next_pos >= self.max_seq_len:
                self.finish(request, FINISH_LENGTH)
            else:
                eligible.append(request)
        entries, failed = self._planner.plan(eligible)
        if not defer_kv_failures:
            for request in failed:
                self.finish(
                    request,
                    FINISH_ABORTED,
                    error_reason="kv_cache_extension_failed",
                )
        self._step_id += 1
        plan = SchedulerOutput(
            self._step_id, policy_version, tuple(entries), return_logprobs
        )
        for entry in entries:
            key = (plan.step_id, entry.identity)
            self._planned[key] = (plan.policy_version, entry)
            self._pending_order.setdefault(entry.identity, deque()).append(plan.step_id)
            self._states[entry.request_id].num_computed_tokens = entry.materialized_end
        return plan

    def update_from_output(self, output: ModelRunnerOutput, policy_version=None):
        """Apply identity-addressed rows in per-request step order, exactly once."""
        if policy_version is None:
            policy_version = self._policy_version
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
                if version != policy_version:
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

    def reserve_tokenized_request(self, request, register_metrics=False):
        if not self._kv_manager.alloc_slots(request.request_id, request.prompt_ids):
            return False
        request.emit_events = False
        self.remember(request)
        if register_metrics:
            self._metrics.register(request.request_id)
        return True

    def release_tokenized_request(self, request):
        self._kv_manager.free_slots(request.request_id)
        self._states.pop(request.request_id, None)
