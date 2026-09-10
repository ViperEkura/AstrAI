"""Policy version protocol for weights shared between trainer and server.

The guard owns the monotonic version counter, the RLock that serializes
weight publication against generation (``run_batch`` acquires the same
lock), and the validation/commit rules around both.  Scheduler-specific
preconditions (no in-flight generation, no queued tasks) and side effects
(dropping stale KV entries) are injected as callables so the guard stays
free of inference-subsystem knowledge.
"""

import threading
from functools import wraps
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def _locked(method):
    @wraps(method)
    def synchronized(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return synchronized


class PolicyVersionGuard:
    """Monotonic policy-version protocol over shared in-place weights.

    The scheduler and the in-process trainer mutate the same model object;
    every weight mutation must publish a new version atomically under the
    generation lock.  Versions never move backwards and every mutation
    advances (``apply_weight_update``) or repeats the live version
    (``update_weights`` idempotently).
    """

    def __init__(
        self,
        policy_version: int,
        ensure_ready: Callable[[], None],
        on_commit: Callable[[], None],
    ):
        if (
            isinstance(policy_version, bool)
            or not isinstance(policy_version, int)
            or policy_version < 0
        ):
            raise ValueError("policy_version must be a non-negative integer")
        self._lock = threading.RLock()
        self._policy_version = policy_version
        self._ensure_ready = ensure_ready
        self._on_commit = on_commit

    @property
    def lock(self) -> threading.RLock:
        """Generation/weight mutex; synchronous generation acquires it too."""
        return self._lock

    @property
    def policy_version(self) -> int:
        """Version of the model weights used for subsequent generations."""
        return self._policy_version

    def _validate(self, policy_version: int, *, require_advance: bool = False) -> None:
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
        if require_advance and policy_version == self._policy_version:
            raise ValueError(
                f"policy_version must advance beyond {self._policy_version} "
                "when model weights are mutated"
            )

    @_locked
    def update_weights(self, policy_version: int) -> int:
        """Acknowledge an in-place weight update and invalidate stale KV state.

        The scheduler owns the same model object as the in-process trainer,
        so weights have already changed when this method is called. The
        explicit version update makes that lifecycle visible and prevents
        prefix KV entries produced by older weights from being reused.
        """
        self._validate(policy_version)
        if policy_version == self._policy_version:
            return self._policy_version
        self._ensure_ready()
        return self._commit(policy_version)

    @_locked
    def apply_weight_update(
        self, policy_version: Optional[int], update: Callable[[], T]
    ) -> T:
        """Mutate shared weights and publish their version without generation.

        ``policy_version=None`` derives ``live + 1`` under the same lock, for
        callers that only need "advance by one" (e.g. ``optimizer.step()``)
        without a read-compute-write race on the current version.
        """
        if not callable(update):
            raise TypeError("update must be callable")
        if policy_version is None:
            policy_version = self._policy_version + 1
        else:
            self._validate(policy_version, require_advance=True)
        self._ensure_ready()

        result = update()
        self._commit(policy_version)
        return result

    @_locked
    def with_policy_snapshot(self, inspect: Callable[[int], T]) -> T:
        """Inspect state while the policy version remains stable."""
        if not callable(inspect):
            raise TypeError("inspect must be callable")
        return inspect(self._policy_version)

    def _commit(self, policy_version: int) -> int:
        self._on_commit()
        self._policy_version = policy_version
        return self._policy_version
