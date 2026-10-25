"""Single-process driver for shared training and inference model execution."""

import logging
import threading
import uuid
from collections import deque
from contextlib import contextmanager
from functools import wraps
from typing import Callable, List, Optional, TypeVar

import torch

from astrai.extension.runtime.context import (
    snapshot_op_selection,
    use_op_selection,
)
from astrai.inference.core.cache.pool import BlockPool
from astrai.inference.core.events import (
    FINISH_ABORTED,
    FINISH_CANCELLED,
    FINISH_STOP_TOKEN,
)
from astrai.inference.core.request import GenerationResult, Request, RequestStatus
from astrai.inference.core.scheduler import Scheduler
from astrai.inference.core.versioning import PolicyVersionGuard
from astrai.inference.worker.model_runner import GPUModelRunner
from astrai.model.automodel import AutoModel
from astrai.tokenize.tokenizer import AutoTokenizer

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _synchronous(method):
    @wraps(method)
    def synchronized(self, *args, **kwargs):
        with self.exclusive():
            return method(self, *args, **kwargs)

    return synchronized


class EngineCore:
    """Own the Scheduler, model runner, loop, and policy operation boundary."""

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
        self.enable_overlap = enable_overlap
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
        self.scheduler = Scheduler(
            tokenizer,
            max_batch_size,
            self.max_seq_len,
            cache,
            token_budget,
            policy_version,
        )
        self.op_selection = snapshot_op_selection()
        with use_op_selection(self.op_selection):
            self.model_runner = GPUModelRunner(
                model=model,
                kv_cache=cache,
                cache_mgr=self.scheduler.kv_manager,
                device=self.device,
                dtype=self.dtype,
                enable_cuda_graph=enable_cuda_graph,
            )
        self._policy_guard = PolicyVersionGuard(
            policy_version,
            ensure_ready=self.ensure_weight_update_ready,
            on_commit=self.scheduler.kv_manager.invalidate_cache,
        )
        self.lock = self._policy_guard.lock
        self.stop_event = threading.Event()
        self.loop_thread = None
        self._lifecycle_lock = threading.Lock()
        self._pending = deque()
        self._closing = False
        self._shutdown_failed = False

    @property
    def model(self):
        return self.model_runner.model

    @property
    def max_batch_size(self):
        return self.scheduler.max_batch_size

    @property
    def policy_version(self):
        return self._policy_guard.policy_version

    @property
    def cuda_graph_enabled(self):
        return self.model_runner.cuda_graph_enabled

    def set_event_sink(self, sink):
        with self.lock:
            self.scheduler.set_event_sink(sink)

    def add_request(self, prompt: str, **kwargs) -> str:
        with self.lock:
            self.ensure_accepting()
            return self.scheduler.add_request(prompt, **kwargs)

    def add_requests(self, prompts: List[str], **kwargs) -> List[str]:
        with self.lock:
            self.ensure_accepting()
            return self.scheduler.add_requests(prompts, **kwargs)

    def cancel_request(self, request_id: str) -> bool:
        with self.lock:
            return self.scheduler.cancel_request(request_id)

    def get_stats(self):
        with self.lock:
            stats = self.scheduler.get_stats()
            stats["policy_version"] = self.policy_version
            return stats

    def update_weights(self, policy_version: int) -> int:
        with self.lock:
            version = self._policy_guard.update_weights(policy_version)
            self.scheduler.set_policy_version(version)
            return version

    def apply_weight_update(
        self, policy_version: Optional[int], update: Callable[[int], T]
    ) -> T:
        with self.lock:
            result = self._policy_guard.apply_weight_update(policy_version, update)
            self.scheduler.set_policy_version(self.policy_version)
            return result

    def with_policy_snapshot(self, inspect: Callable[[int], T]) -> T:
        return self._policy_guard.with_policy_snapshot(inspect)

    def _release_finished(self):
        self.scheduler.release_finished(safe_to_release=not self._shutdown_failed)

    @property
    def pending(self):
        return tuple(self._pending)

    def ensure_accepting(self):
        if self._closing or self._shutdown_failed:
            raise RuntimeError(
                "engine is stopping or stopped; start it before submitting"
            )

    def ensure_weight_update_ready(self):
        if self.loop_thread is not None and self.loop_thread.is_alive():
            raise RuntimeError("Stop the scheduler before updating model weights")
        if self.scheduler.has_requests():
            raise RuntimeError("Cannot update model weights while requests are queued")
        self.drain()
        self._release_finished()
        if self._shutdown_failed or self.scheduler.has_outstanding():
            raise RuntimeError("Cannot update weights before execution is drained")

    @contextmanager
    def exclusive(self):
        """Synchronous score/rollout may share an idle loop, never live work."""
        with self.lock:
            self.drain()
            self._release_finished()
            if self.scheduler.has_requests():
                raise RuntimeError(
                    "Cannot run synchronous inference with queued requests"
                )
            if self._shutdown_failed:
                raise RuntimeError("engine execution has not drained safely")
            try:
                yield
            except BaseException as error:
                # Include partial setup and forwards that failed before a
                # handle existed. Failed fences retain the request/KV registry.
                self.fence()
                self._shutdown(error)
                raise

    def fence(self):
        try:
            self.model_runner.synchronize()
        except BaseException:
            self._shutdown_failed = True
            raise

    def submit(self, plan):
        for pending in self.model_runner.execute_model(plan):
            self._pending.append(pending)

    def resolve(self, pending):
        return self.scheduler.update_from_output(pending.commit(), self.policy_version)

    def drain(self, count=None):
        remaining = len(self._pending) if count is None else count
        while remaining and self._pending:
            pending = self._pending[0]
            self.resolve(pending)
            self._pending.popleft()
            remaining -= 1
        latest = self.model_runner.peek_pending()
        if latest is not None and latest.committed:
            self.model_runner.clear_pending()

    def execute_synchronously(self, requests, return_logprobs=False):
        with self.lock:
            self.drain()
            plan = self.scheduler.schedule(
                requests, self.policy_version, return_logprobs=return_logprobs
            )
            try:
                self.submit(plan)
                self.drain()
            except BaseException:
                self.fence()
                self._discard_pending()
                raise
        return (
            [r for r in requests if r.status != RequestStatus.ABORTED],
            [r for r in requests if r.status == RequestStatus.ABORTED],
        )

    def _can_overlap(self, active):
        if not self.enable_overlap or not active:
            return False
        if any(
            not r.prefill_complete
            or r.frequency_penalty
            or r.next_pos >= self.scheduler.max_seq_len
            or (r.max_tokens is not None and r.output_tokens + 1 >= r.max_tokens)
            for r in active
        ):
            return False
        if self._pending:
            last = self._pending[-1].snapshot
            return (
                len(self._pending) == 1
                and all(r.phase == "decode" for r in last.requests)
                and last.identities == tuple(r.identity for r in active)
            )
        return True

    def tick(self, *, return_logprobs=False):
        scheduler = self.scheduler
        self._release_finished()
        scheduler.admit_requests(safe_to_release=not self._shutdown_failed)
        active = scheduler.active_requests()
        overlap = self._can_overlap(active)
        if self._pending and not overlap:
            self.drain()
            self._release_finished()
            scheduler.admit_requests(safe_to_release=not self._shutdown_failed)
            active = scheduler.active_requests()
            overlap = self._can_overlap(active)
        if not active:
            self.drain()
            self._release_finished()
            return False
        old_count = len(self._pending)
        plan = scheduler.schedule(
            active,
            self.policy_version,
            return_logprobs=return_logprobs,
            defer_kv_failures=bool(old_count and overlap),
        )
        if (
            old_count
            and overlap
            and plan.identities != self._pending[-1].snapshot.identities
        ):
            # KV extension may shrink a previously steady batch. Its host
            # input snapshots are then stale and the D2D relay cannot match.
            # Commit pending tokens before declaring extension failures:
            # a pending EOS may already have completed a failed request.
            scheduler.rollback_schedule(plan)
            self.drain()
            self._release_finished()
            active = scheduler.active_requests()
            plan = scheduler.schedule(
                active, self.policy_version, return_logprobs=return_logprobs
            )
            overlap = False
        self.submit(plan)
        if (
            overlap
            and plan.requests
            and all(r.phase == "decode" for r in plan.requests)
        ):
            # Launch N+1 before resolving N; at most one handle stays in flight.
            self.drain(old_count)
        else:
            self.drain()
        self._release_finished()
        return True

    def _discard_pending(self):
        """Only after synchronize succeeded, forget results that cannot apply."""
        for pending in self._pending:
            if pending._ring is not None:
                pending._ring.release(pending)
        self._pending.clear()
        self.model_runner.clear_pending()
        self.scheduler.discard_outstanding()

    def _shutdown(self, error=None):
        try:
            self._drain_shutdown(error)
        except BaseException as failure:
            self._shutdown_failed = True
            # Consumer completion and resource retirement are independent.
            # A failed fence quarantines KV, never the terminal error event.
            self.scheduler.abort_all(FINISH_CANCELLED, error=error or failure)
            raise

    def _drain_shutdown(self, error=None):
        if self._shutdown_failed:
            self.fence()
            self._shutdown_failed = False
        try:
            self.drain()
        except BaseException as drain_error:
            # A bad output must not skip fencing or prevent other resources
            # from being reclaimed. If the fence fails, retain all KV state.
            try:
                self.fence()
            except BaseException:
                self._shutdown_failed = True
                raise
            self._discard_pending()
            error = error or drain_error
        if self.scheduler.has_outstanding():
            # Includes a group whose forward raised before yielding a handle.
            try:
                self.fence()
            except BaseException:
                self._shutdown_failed = True
                raise
            self._discard_pending()
        self.scheduler.abort_all(FINISH_CANCELLED, error=error)
        self._release_finished()
        self._shutdown_failed = False

    def run_busy_loop(self):
        error = None
        try:
            with use_op_selection(self.op_selection):
                while not self.stop_event.is_set():
                    with self.lock:
                        work = self.tick()
                    if not work:
                        if self.scheduler.has_requests():
                            # Allocator pressure with a nonempty waiting queue.
                            self.stop_event.wait(0.005)
                        else:
                            self.scheduler.wait_for_requests(timeout=0.05)
        except BaseException as exc:
            error = exc
            self._closing = True
            self.stop_event.set()
            logger.exception("EngineCore loop crashed")
        finally:
            with self.lock:
                try:
                    self._shutdown(error)
                except BaseException:
                    logger.exception("EngineCore could not safely drain; retaining KV")

    def start(self):
        with self.lock, self._lifecycle_lock:
            if self.loop_thread is not None and self.loop_thread.is_alive():
                return
            if self._shutdown_failed:
                raise RuntimeError("engine must drain safely before restart")
            self.scheduler.refresh_stop_ids()
            self._closing = False
            self.stop_event.clear()
            self.loop_thread = threading.Thread(target=self.run_busy_loop, daemon=True)
            self.loop_thread.start()

    def stop(self, timeout=2.0):
        with self._lifecycle_lock:
            self._closing = True
            self.stop_event.set()
            self.scheduler.wake()
            thread = self.loop_thread
        # Never join under the generation mutex: loop/finally needs it to drain.
        if thread is threading.current_thread():
            return False
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        if thread is not None and thread.is_alive():
            logger.warning("EngineCore stop timed out; retaining thread and KV state")
            return False
        with self.lock:
            if self.loop_thread is not thread:
                # A completed predecessor may have been explicitly restarted
                # while join ran without the lock. Never clean its successor.
                return False
            self._shutdown()
            self.loop_thread = None
        return True

    @_synchronous
    def generate(
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
        seeds: Optional[List[int]] = None,
    ) -> List[GenerationResult]:
        """Generate tokenized prompts, returning terminal results in input order.

        Invalid prompts retain a rejected result slot. ``return_logprobs``
        controls computation only; the result type is always the same.
        All submitted execution drains before the policy boundary is released.
        """
        if seeds is not None and (
            len(seeds) != len(prompt_ids_list)
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed < 2**63
                for seed in seeds
            )
        ):
            raise ValueError("seeds must contain one integer in [0, 2**63) per request")
        requests, errors = [], []
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
                    f"batch_{uuid.uuid4().hex}",
                    ids,
                    limit,
                    temperature,
                    top_p,
                    top_k,
                    frequency_penalty,
                    rep_window,
                    seed=None if seeds is None else seeds[index],
                )
            requests.append(request)
            errors.append(error)
        self._execute_requests(
            [request for request in requests if request is not None],
            return_logprobs=return_logprobs,
        )
        results = []
        for request, error in zip(requests, errors):
            if request is None:
                results.append(GenerationResult([], [], "rejected", error))
            else:
                reason = (
                    "rejected"
                    if request.error_reason
                    else "stop"
                    if request.finish_reason == FINISH_STOP_TOKEN
                    else "length"
                )
                results.append(
                    GenerationResult(
                        list(request.output_ids),
                        list(request.output_logprobs),
                        reason,
                        request.error_reason,
                    )
                )
        return results

    def _execute_requests(
        self, requests: List[Request], return_logprobs: bool = False
    ) -> None:
        """Drive tokenized requests with the serving loop's execution pipeline."""
        self.scheduler.refresh_stop_ids()
        self.scheduler.enqueue_tokenized_requests(requests)
        try:
            with use_op_selection(self.op_selection):
                while any(not request.terminal_emitted for request in requests):
                    if not self.tick(return_logprobs=return_logprobs):
                        self.scheduler.reject_waiting_for_capacity()
                        break
        except BaseException:
            self.fence()
            self._discard_pending()
            raise
        finally:
            self.drain()
            for request in requests:
                if not request.terminal_emitted:
                    self.scheduler.finish(
                        request, FINISH_ABORTED, error_reason="execution_failed"
                    )
            self._release_finished()

    @_synchronous
    def score_ids(self, prompt_ids_list, continuation_ids_list, per_token=False):
        if len(prompt_ids_list) != len(continuation_ids_list):
            raise ValueError("prompt and continuation lists must have equal length")
        requests = []
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
            )
            request.cont_len = len(cont_ids)
            if not self.scheduler.reserve_tokenized_request(request):
                requests.append(None)
                continue
            requests.append(request)
        live = [r for r in requests if r is not None]
        results = {}
        try:
            if live:
                with use_op_selection(self.op_selection):
                    scored = self.model_runner.execute_score(
                        [r.execution("score", 0, len(r.prompt_ids)) for r in live],
                        per_token=per_token,
                    )
                results = {r.request_id: value for r, value in zip(live, scored)}
        finally:
            # Score has no PendingExecution, including when forward throws.
            self.fence()
            for request in live:
                self.scheduler.release_tokenized_request(request)
        return [results.get(r.request_id) if r is not None else None for r in requests]
