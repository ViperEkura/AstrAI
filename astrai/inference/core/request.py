import uuid
from dataclasses import dataclass
from enum import Enum
from typing import (
    List,
    Literal,
    Optional,
)

from tokenizers.decoders import DecodeStream

from astrai.config.inference_config import InferenceConfig
from astrai.inference.contracts import ExecutionRequest, RequestIdentity, SamplingParams
from astrai.tokenize.tokenizer import AutoTokenizer

_config = InferenceConfig()


@dataclass(frozen=True)
class GenerationResult:
    """Structured terminal result for one synchronous generation request."""

    token_ids: List[int]
    logprobs: List[float]
    finish_reason: Literal["stop", "length", "cancelled", "rejected"]
    error_reason: Optional[str] = None


class StreamDecoder:
    """Incremental decoder backed by the tokenizers library's DecodeStream.

    Delegates to the Rust-native streaming decoder which maintains an
    O(1) bounded token buffer internally (via prefix drain), avoiding
    the O(n²) cost of re-decoding the full history on each step.

    Multi-byte UTF-8 sequences split across token boundaries are
    buffered until complete; ``push`` returns "" while the trailing
    sequence is still incomplete.
    """

    __slots__ = ("_stream", "_tok")

    def __init__(self, tokenizer: AutoTokenizer):
        # Test doubles and lightweight tokenizers may lack the Rust handle;
        # ``push`` degrades to id-as-string so streams still terminate.
        self._tok = getattr(tokenizer, "_tokenizer", None)
        self._stream = (
            DecodeStream(skip_special_tokens=True) if self._tok is not None else None
        )

    def push(self, token_id: int) -> str:
        """Append a token ID and return newly completed text.

        Returns "" while a multi-byte character is still incomplete.
        """
        if self._stream is None:
            return str(token_id)
        chunk = self._stream.step(self._tok, token_id)
        return chunk or ""


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
    ):
        self.request_id = request_id
        self.identity = RequestIdentity(request_id, uuid.uuid4().hex)
        self.prompt_ids = list(prompt_ids)
        self._prompt_snapshot = tuple(prompt_ids)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.rep_window = rep_window
        self._sampling = SamplingParams(
            temperature, top_p, top_k, frequency_penalty, rep_window, seed
        )

        # Scoring only: how many trailing tokens of ``prompt_ids`` form the
        # continuation to score.  Zero for generation requests.
        self.cont_len: int = 0

        self.status = RequestStatus.PENDING
        self.output_ids: List[int] = []
        self.output_logprobs: List[float] = []
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
            prompt_ids=self._prompt_snapshot,
            input_token_id=self.output_ids[-1]
            if self.output_ids
            else self.prompt_ids[-1],
            position=position,
            num_tokens=num_tokens,
            phase=phase,
            sampling=self._sampling,
            output_ids=tuple(self.output_ids) if self.frequency_penalty else (),
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
