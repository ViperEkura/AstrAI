"""Deferred execution resources; no mutable core request objects cross here."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import Tensor

from astrai.inference.contracts import (
    ModelRunnerOutput,
    RequestIdentity,
    RequestOutput,
    SchedulerOutput,
)


@dataclass
class PendingExecution:
    """GPU resources plus an immutable plan, resolved exactly once on host.

    Row order belongs to ``sampled_identities``, not to the scheduler's input
    ordering. Continuation-only prefills also carry a completion fence.
    """

    snapshot: SchedulerOutput
    tokens: Optional[Tensor] = None
    logprobs: Optional[Tensor] = None
    sampled_identities: Tuple[RequestIdentity, ...] = ()
    completion_event: Optional["torch.cuda.Event"] = None
    copy_event: Optional["torch.cuda.Event"] = None
    host_tokens: Optional[Tensor] = None
    host_logprobs: Optional[Tensor] = None
    _ring: Optional["ResultRing"] = field(default=None, repr=False)
    _payload: Optional[ModelRunnerOutput] = field(default=None, init=False, repr=False)

    def commit(self) -> ModelRunnerOutput:
        if self._payload is not None:
            return self._payload
        if self.copy_event is not None and self.host_tokens is not None:
            self.copy_event.synchronize()
            tokens = self.host_tokens.tolist()
            logprobs = (
                self.host_logprobs.tolist() if self.host_logprobs is not None else None
            )
        else:
            if self.completion_event is not None:
                self.completion_event.synchronize()
            tokens = self.tokens.tolist() if self.tokens is not None else []
            logprobs = self.logprobs.tolist() if self.logprobs is not None else None
        if len(tokens) != len(self.sampled_identities):
            raise RuntimeError("sampled rows do not match their execution identities")
        if logprobs is not None and len(logprobs) != len(tokens):
            raise RuntimeError("logprob rows do not match sampled rows")
        sampled = {
            identity: (token, logprobs[i] if logprobs is not None else None)
            for i, (identity, token) in enumerate(zip(self.sampled_identities, tokens))
        }
        if len(sampled) != len(self.sampled_identities):
            raise RuntimeError("duplicate sampled execution identity")
        if not sampled.keys() <= self.snapshot._identity_set:  # noqa: SLF001
            raise RuntimeError("sampled identity is not present in execution plan")
        self._payload = ModelRunnerOutput(
            self.snapshot.step_id,
            self.snapshot.policy_version,
            tuple(
                RequestOutput(
                    r.identity,
                    r.materialized_end,
                    *sampled.get(r.identity, (None, None)),
                )
                for r in self.snapshot.requests
            ),
        )
        if self._ring is not None:
            self._ring.release(self)
        return self._payload

    @property
    def committed(self) -> bool:
        return self._payload is not None


class ResultRing:
    """Depth-2 pinned host slots fed by a dedicated copy stream.

    ``post`` stages a pending step's device results into the free slot:
    the copy stream first waits on the producing work (current stream
    ordering is inherited — tokens were produced on the compute stream),
    copies ``[B]`` tokens (and optional ``[B]`` logprobs) into pinned
    host buffers and records a completion event onto the pending step.
    Depth 2 lets the scheduler submit step t+1's copy while step t's copy
    may still be in flight; a slot is only reused once its previous
    occupant's event has been waited on (the pending step itself holds
    the only reference that matters, and ``post`` synchronises the old
    event before overwriting).

    On CPU (or when CUDA is unavailable) the ring is inert and pending
    steps fall back to synchronous ``tolist`` commits.
    """

    def __init__(self, max_batch_size: int, device):
        self._enabled = (
            torch.cuda.is_available() and torch.device(device).type == "cuda"
        )
        self._slots = []
        if self._enabled:
            self._copy_stream = torch.cuda.Stream()
            for _ in range(2):
                self._slots.append(
                    {
                        "tokens": torch.empty(
                            max_batch_size, dtype=torch.long, pin_memory=True
                        ),
                        "logprobs": torch.empty(
                            max_batch_size, dtype=torch.float32, pin_memory=True
                        ),
                        "event": torch.cuda.Event(),
                        "in_use": False,
                    }
                )
        else:
            self._copy_stream = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def post(self, pending: PendingExecution) -> bool:
        """Stage ``pending``'s results into a ring slot; False if refused.

        Refusal means the caller should keep the synchronous ``tolist``
        fallback (used for CPU runs and for oversized batches).
        """
        if not self._enabled:
            return False
        b = pending.tokens.shape[0]
        if b > self._slots[0]["tokens"].shape[0]:
            return False
        slot = None
        for candidate in self._slots:
            if not candidate["in_use"]:
                slot = candidate
                break
        if slot is None:
            # A completed copy is not necessarily consumed. Never overwrite
            # an uncommitted payload; retain its device tensor as fallback.
            return False
        tokens_pin = slot["tokens"][:b]
        logprobs_pin = slot["logprobs"][:b] if pending.logprobs is not None else None
        # The copy stream must not race the producing work: tokens were
        # sampled on the compute stream (possibly still in flight). Record
        # the current stream position and make the copy stream wait on it
        # before the D2H, so the pinned slot never lands a stale read.
        produced = torch.cuda.Event()
        produced.record()
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(produced)
            tokens_pin.copy_(pending.tokens, non_blocking=True)
            if logprobs_pin is not None:
                logprobs_pin.copy_(pending.logprobs, non_blocking=True)
            slot["event"].record()
        slot["in_use"] = True
        pending.copy_event = slot["event"]
        pending.host_tokens = tokens_pin
        pending.host_logprobs = logprobs_pin
        pending._ring = self
        return True

    def release(self, pending: PendingExecution) -> None:
        """Mark the slot a committed step occupied as reusable."""
        if not self._enabled:
            return
        for slot in self._slots:
            if slot["event"] is pending.copy_event:
                slot["in_use"] = False
                break
