import uuid
from enum import Enum
from typing import (
    List,
    Optional,
    Sequence,
)

from astrai.config.inference_config import (
    InferenceConfig,
    normalize_eos_token_ids,
)
from astrai.inference.contracts import ExecutionRequest, RequestIdentity, SamplingParams
from astrai.inference.contracts import GenerationResult as GenerationResult
from astrai.tokenize.tokenizer import StreamDecoder as StreamDecoder

_config = InferenceConfig()


class RequestStatus(Enum):
    """Request lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class Request:
    """Single generation request: prompt, sampling params, output state."""

    def __init__(
        self,
        request_id: str,
        prompt_ids: List[int],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        frequency_penalty: float = 0.0,
        rep_window: int = _config.default_rep_window,
        *,
        seed: Optional[int] = None,
        eos_token_ids: Optional[Sequence[int]] = None,
    ):
        self.request_id = request_id
        self.identity = RequestIdentity(request_id, uuid.uuid4().hex)
        self.prompt_ids = list(prompt_ids)
        self._prompt_snapshot = tuple(prompt_ids)
        self.max_tokens = max_tokens
        self.eos_token_ids = (
            None
            if eos_token_ids is None
            else frozenset(normalize_eos_token_ids(eos_token_ids))
        )
        self.sampling = SamplingParams(
            temperature, top_p, top_k, frequency_penalty, rep_window, seed
        )

        # Scoring only: how many trailing tokens of ``prompt_ids`` form the
        # continuation to score.  Zero for generation requests.
        self.cont_len: int = 0

        self.status = RequestStatus.PENDING
        self.output_ids: List[int] = []
        self.output_logprobs: List[float] = []
        self.cache_slot = -1
        self.cached_tokens = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.num_computed_tokens: int = 0
        self.num_materialized_tokens: int = 0
        self.finish_reason: Optional[str] = None
        self.error_reason: Optional[str] = None
        self.terminal_emitted = False
        self.emit_events = True

    def execution(self, phase: str, position: int, num_tokens: int) -> ExecutionRequest:
        """Detach all mutable request state at the core/worker boundary."""
        return ExecutionRequest(
            identity=self.identity,
            cache_slot=self.cache_slot,
            prompt_ids=self._prompt_snapshot,
            input_token_id=self.output_ids[-1]
            if self.output_ids
            else self.prompt_ids[-1],
            position=position,
            num_tokens=num_tokens,
            phase=phase,
            sampling=self.sampling,
            output_ids=tuple(self.output_ids)
            if self.sampling.frequency_penalty
            else (),
            cont_len=self.cont_len,
        )

    @property
    def next_pos(self) -> int:
        """KV position where the next decode step will write."""
        return self.num_computed_tokens

    @property
    def prefill_complete(self) -> bool:
        """True when all prompt KV entries are materialized."""
        return self.num_computed_tokens >= self.input_tokens > 0

    def is_finished(self, stop_ids) -> bool:
        """Terminal check; ``stop_ids`` may be a set (O(1) membership)."""
        if self.max_tokens is not None and self.output_tokens >= self.max_tokens:
            return True
        if self.output_ids and self.output_ids[-1] in stop_ids:
            return True
        return False
