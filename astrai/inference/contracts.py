"""Immutable, in-process scheduling/execution messages.

No request lifecycle objects or core/worker imports belong in this module.
An identity includes an incarnation so a late result cannot hit a reused id.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple


@dataclass(frozen=True, eq=False)
class RequestIdentity:
    """Stable per-incarnation request address.

    ``eq=False`` opts out of the dataclass-generated field-by-field
    comparison: identities are dict keys in every scheduler hot map
    (``_planned`` / ``_pending_order`` / ``_ready``) and in the commit-side
    duplicate check, so hashing and equality drop to tuple speed instead of
    a generated ``__eq__`` call per lookup.
    """

    request_id: str
    incarnation: str

    def __hash__(self):
        return hash((self.request_id, self.incarnation))

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, RequestIdentity):
            return NotImplemented
        return (self.request_id, self.incarnation) == (
            other.request_id,
            other.incarnation,
        )


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    frequency_penalty: float = 0.0
    rep_window: int = 64
    seed: Optional[int] = None


@dataclass(frozen=True)
class ExecutionRequest:
    identity: RequestIdentity
    prompt_ids: Tuple[int, ...]
    input_token_id: int
    position: int
    num_tokens: int
    phase: Literal["prefill", "decode", "score"]
    sampling: SamplingParams = SamplingParams()
    # Only frequency-penalty execution needs a host output-history snapshot.
    output_ids: Tuple[int, ...] = ()
    cont_len: int = 0
    backend: Any = None

    @property
    def request_id(self) -> str:
        return self.identity.request_id

    @property
    def next_pos(self) -> int:
        return self.position

    @property
    def materialized_end(self) -> int:
        return self.position + self.num_tokens

    @property
    def samples(self) -> bool:
        return self.phase == "decode" or (
            self.phase == "prefill" and self.materialized_end == len(self.prompt_ids)
        )

    @property
    def temperature(self) -> float:
        return self.sampling.temperature

    @property
    def top_p(self) -> float:
        return self.sampling.top_p

    @property
    def top_k(self) -> int:
        return self.sampling.top_k

    @property
    def frequency_penalty(self) -> float:
        return self.sampling.frequency_penalty

    @property
    def rep_window(self) -> int:
        return self.sampling.rep_window

    @property
    def sampling_position(self) -> int:
        return self.materialized_end - len(self.prompt_ids)


@dataclass(frozen=True)
class SchedulerOutput:
    step_id: int
    policy_version: int
    requests: Tuple[ExecutionRequest, ...]
    return_logprobs: bool = False

    @property
    def request_ids(self) -> Tuple[str, ...]:
        return tuple(r.request_id for r in self.requests)

    @property
    def identities(self) -> Tuple[RequestIdentity, ...]:
        cached = self.__dict__.get("_identities_cache")
        if cached is None:
            cached = tuple(r.identity for r in self.requests)
            object.__setattr__(self, "_identities_cache", cached)
        return cached

    @property
    def _identity_set(self) -> frozenset:
        # Commit-side membership checks use this instead of rebuilding a
        # set from the tuple on every pending commit.
        cached = self.__dict__.get("_identity_set_cache")
        if cached is None:
            cached = frozenset(self.identities)
            object.__setattr__(self, "_identity_set_cache", cached)
        return cached

    def select(self, requests) -> "SchedulerOutput":
        return SchedulerOutput(
            self.step_id, self.policy_version, tuple(requests), self.return_logprobs
        )


@dataclass(frozen=True)
class RequestOutput:
    identity: RequestIdentity
    materialized_end: int
    token_id: Optional[int] = None
    logprob: Optional[float] = None


@dataclass(frozen=True)
class ModelRunnerOutput:
    step_id: int
    policy_version: int
    results: Tuple[RequestOutput, ...]
