"""Single-process execution owner for the shared training/inference model."""

import logging
import threading
from collections import deque
from contextlib import contextmanager
from typing import TYPE_CHECKING

from astrai.inference.core.events import FINISH_CANCELLED

if TYPE_CHECKING:
    from astrai.inference.core.scheduler import Scheduler

logger = logging.getLogger(__name__)


class EngineCore:
    """Own the loop, in-flight resources, draining and the policy lock domain.

    Scheduler owns requests and applies results. This driver is the only caller
    that launches worker work. It is deliberately an object/thread, not a process.
    The lock is PolicyVersionGuard's RLock, also used by synchronous execution.
    """

    def __init__(self, scheduler: "Scheduler", lock):
        self.scheduler = scheduler
        self.lock = lock
        self.stop_event = threading.Event()
        self.loop_thread = None
        self._lifecycle_lock = threading.Lock()
        self._pending = deque()
        self._closing = False
        self._shutdown_failed = False

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
        if self.scheduler._requests.has_requests():
            raise RuntimeError("Cannot update model weights while requests are queued")
        self.drain()
        self.scheduler.release_finished()
        if self._shutdown_failed or self.scheduler._planned:
            raise RuntimeError("Cannot update weights before execution is drained")

    @contextmanager
    def exclusive(self):
        """Synchronous score/rollout may share an idle loop, never live work."""
        with self.lock:
            self.drain()
            self.scheduler.release_finished()
            if self.scheduler._requests.has_requests():
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
            self.scheduler._executor.synchronize()
        except BaseException:
            self._shutdown_failed = True
            raise

    def submit(self, plan):
        for pending in self.scheduler._executor.execute_model(plan):
            self._pending.append(pending)

    def resolve(self, pending):
        return self.scheduler.update_from_output(pending.commit())

    def drain(self, count=None):
        remaining = len(self._pending) if count is None else count
        while remaining and self._pending:
            pending = self._pending[0]
            self.resolve(pending)
            self._pending.popleft()
            remaining -= 1
        latest = self.scheduler._executor.peek_pending()
        if latest is not None and latest.committed:
            self.scheduler._executor.clear_pending()

    def execute_synchronously(self, requests, return_logprobs=False):
        with self.lock:
            self.drain()
            plan = self.scheduler.schedule(requests, return_logprobs=return_logprobs)
            try:
                self.submit(plan)
                self.drain()
            except BaseException:
                self.fence()
                self._discard_pending()
                raise

    def _can_overlap(self, active):
        if not self.scheduler._enable_overlap or not active:
            return False
        first_backend = active[0].backend
        if any(
            not r.prefill_complete
            or r.backend is not first_backend
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

    def tick(self, requests=None, *, return_logprobs=False):
        """Advance serving admission or a fixed synchronous batch.

        Both callers share submit/commit, the device token relay and
        depth-two draining. Fixed batches do not admit unrelated requests.
        """
        scheduler = self.scheduler

        def refresh():
            scheduler.release_finished()
            if requests is None:
                scheduler.admit_requests()
                return scheduler.active_requests()
            return [request for request in requests if not request.terminal_emitted]

        active = refresh()
        overlap = self._can_overlap(active)
        if self._pending and not overlap:
            self.drain()
            active = refresh()
            overlap = self._can_overlap(active)
        if not active:
            self.drain()
            scheduler.release_finished()
            return False
        old_count = len(self._pending)
        plan = scheduler.schedule(active, return_logprobs=return_logprobs)
        if (
            old_count
            and overlap
            and plan.identities != self._pending[-1].snapshot.identities
        ):
            # KV extension may shrink a previously steady batch. Its host
            # input snapshots are then stale and the D2D relay cannot match.
            scheduler.rollback_schedule(plan)
            self.drain()
            active = refresh()
            plan = scheduler.schedule(active, return_logprobs=return_logprobs)
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
        scheduler.release_finished()
        return True

    def _discard_pending(self):
        """Only after synchronize succeeded, forget results that cannot apply."""
        for pending in self._pending:
            if pending._ring is not None:
                pending._ring.release(pending)
        self._pending.clear()
        self.scheduler._executor.clear_pending()
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
        if self.scheduler._planned:
            # Includes a group whose forward raised before yielding a handle.
            try:
                self.fence()
            except BaseException:
                self._shutdown_failed = True
                raise
            self._discard_pending()
        self.scheduler.abort_all(FINISH_CANCELLED, error=error)
        self.scheduler.release_finished()
        self._shutdown_failed = False

    def run_busy_loop(self):
        error = None
        try:
            with self.scheduler._backend_context():
                while not self.stop_event.is_set():
                    with self.lock:
                        work = self.tick()
                    if not work:
                        if self.scheduler._requests.has_requests():
                            # Allocator pressure with a nonempty waiting queue.
                            self.stop_event.wait(0.005)
                        else:
                            self.scheduler._requests.wait_for_requests(timeout=0.05)
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
            self._closing = False
            self.stop_event.clear()
            self.loop_thread = threading.Thread(target=self.run_busy_loop, daemon=True)
            self.loop_thread.start()

    def stop(self, timeout=2.0):
        with self._lifecycle_lock:
            self._closing = True
            self.stop_event.set()
            self.scheduler._requests.wake()
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
