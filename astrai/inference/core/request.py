import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Deque,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

from tokenizers.decoders import DecodeStream

from astrai.config.inference_config import InferenceConfig
from astrai.inference.contracts import ExecutionRequest, RequestIdentity, SamplingParams
from astrai.inference.core.metrics import MetricsCollector
from astrai.tokenize.tokenizer import AutoTokenizer

if TYPE_CHECKING:
    from astrai.extension import AttentionBackend

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
        backend: Optional["AttentionBackend"] = None,
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
        self.backend = backend
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
        self._decoder: Optional[StreamDecoder] = None

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
            backend=self.backend,
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


class RequestManager:
    """Thread-safe request queues and lifecycle transitions (no page ops)."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 16,
        max_seq_len: int = 8192,
        metrics: Optional["MetricsCollector"] = None,
    ):
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []
        self._requests: Dict[str, Request] = {}

        self._request_event = threading.Event()
        self._lock = threading.Lock()

        self._total_requests = 0
        self._total_tokens = 0
        self._cancelled_total = 0

        self._metrics = metrics

    def add_request(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        frequency_penalty: float = 0.0,
        rep_window: int = 64,
        backend: Optional["AttentionBackend"] = None,
        request_id: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
    ) -> str:
        request_id = request_id or f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        if prompt_ids is None:
            prompt_ids = self.tokenizer.encode(prompt)
            # Some tokenizers answer a bare string with the batched shape
            # ([[ids]]); unwrap so one prompt is always a flat id list.
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
        if not prompt_ids:
            # An empty prompt never completes prefill (``prefill_complete`` stays
            # False) and would crash the decode path on ``prompt_ids[-1]``;
            # rejecting it here keeps the scheduling loop alive.
            raise ValueError("prompt encoded to zero tokens; refusing to schedule")
        if len(prompt_ids) > self.max_seq_len:
            prompt_ids = prompt_ids[-self.max_seq_len :]

        if max_tokens is None:
            max_tokens = self.max_seq_len - len(prompt_ids)
        else:
            max_tokens = min(max_tokens, self.max_seq_len - len(prompt_ids))

        request = Request(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            rep_window=rep_window,
            backend=backend,
        )

        self._register_request(request)
        return request_id

    def add_requests(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        frequency_penalty: float = 0.0,
        rep_window: int = 64,
        backend: Optional["AttentionBackend"] = None,
        request_ids: Optional[List[str]] = None,
        prompts_ids: Optional[List[List[int]]] = None,
    ) -> List[str]:
        """Batch add: one ``encode_batch`` call for all prompts.

        Per-prompt ``add_request`` serializes tokenization — measurable at
        serving batch sizes (128 x 512-token prompts: ~148 ms sequential vs
        ~65 ms batched, and the batch path also leaves the GPU queue free
        for the prefill launches to overlap). Sampling params are shared
        across the batch; per-request overrides go through ``add_request``.

        ``request_ids`` / ``prompts_ids`` let a frontend that already
        tokenized (and minted ids) submit without re-encoding; both must
        match ``prompts`` in length when given.
        """
        if not prompts:
            return []
        if prompts_ids is not None:
            if len(prompts_ids) != len(prompts):
                raise ValueError("prompts_ids must match prompts in length")
            encoded = prompts_ids
        else:
            encoded = self.tokenizer.encode(list(prompts))
            if not isinstance(encoded, list) or len(encoded) != len(prompts):
                raise ValueError("batch tokenizer returned unexpected shape")
        if request_ids is not None and len(request_ids) != len(prompts):
            raise ValueError("request_ids must match prompts in length")

        request_ids_out: List[str] = []
        requests: List[Request] = []
        stamp = time.time()
        for i, prompt_ids in enumerate(encoded):
            if not prompt_ids:
                raise ValueError(
                    f"prompt {i} encoded to zero tokens; refusing to schedule"
                )
            request_id = (
                request_ids[i]
                if request_ids is not None
                else f"req_{int(stamp)}_{uuid.uuid4().hex[:8]}"
            )
            if len(prompt_ids) > self.max_seq_len:
                prompt_ids = prompt_ids[-self.max_seq_len :]
            task_max = (
                self.max_seq_len - len(prompt_ids)
                if max_tokens is None
                else min(max_tokens, self.max_seq_len - len(prompt_ids))
            )
            request = Request(
                request_id=request_id,
                prompt_ids=prompt_ids,
                max_tokens=task_max,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                rep_window=rep_window,
                backend=backend,
            )
            requests.append(request)
            request_ids_out.append(request_id)

        # Validate and install the whole batch in one critical section. A
        # duplicate later in the batch must not leave an unowned prefix queued.
        with self._lock:
            if len(set(request_ids_out)) != len(request_ids_out):
                raise ValueError("duplicate request ids in batch")
            if any(rid in self._requests for rid in request_ids_out):
                raise ValueError("duplicate live request id in batch")
            for request in requests:
                self.waiting.append(request)
                self._requests[request.request_id] = request
                self._total_requests += 1
                if self._metrics is not None:
                    self._metrics.register(request.request_id)
        self._request_event.set()
        return request_ids_out

    def _register_request(self, request: "Request") -> None:
        with self._lock:
            if request.request_id in self._requests:
                raise ValueError(f"duplicate live request id: {request.request_id}")
            self.waiting.append(request)
            self._requests[request.request_id] = request
            self._total_requests += 1

        if self._metrics is not None:
            self._metrics.register(request.request_id)

        self._request_event.set()

    def cancel_request(self, request_id: str) -> Tuple[List[Request], bool]:
        """Mark a request cancelled and return requests safe to clean immediately."""
        cancelled = False
        immediate: List[Request] = []
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.status in (
                RequestStatus.FINISHED,
                RequestStatus.ABORTED,
            ):
                return [], False

            request.status = RequestStatus.ABORTED
            self._cancelled_total += 1
            cancelled = True
            if request in self.waiting:
                self.waiting = deque(
                    waiting for waiting in self.waiting if waiting is not request
                )
                self._requests.pop(request_id, None)
                immediate = [request]

        return immediate, cancelled

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            waiting = len(self.waiting)
            stats: Dict[str, Any] = {
                "total_tasks": self._total_requests,
                "total_tokens": self._total_tokens,
                "running": len(self.running),
                "waiting_tasks": waiting,
                "waiting": waiting,
                "cancelled_total": self._cancelled_total,
            }
        if self._metrics is not None:
            stats.update(self._metrics.get_stats())
        return stats

    def pull_waiting(self, n: int) -> List[Request]:
        to_add: List[Request] = []
        with self._lock:
            take = min(n, len(self.waiting))
            for _ in range(take):
                to_add.append(self.waiting.popleft())
        return to_add

    def activate(self, request: Request) -> bool:
        with self._lock:
            if request.status == RequestStatus.ABORTED:
                self._requests.pop(request.request_id, None)
                return False
            request.status = RequestStatus.RUNNING
            self.running.append(request)
            return True

    def discard_waiting(self, requests: List[Request]):
        """Drop already-pulled waiting requests without re-queueing.

        Used by the admission livelock guard: requests that can never fit
        the pool are terminated (terminal event already emitted) rather
        than returned to the queue to spin forever.
        """
        with self._lock:
            for request in requests:
                self._requests.pop(request.request_id, None)

    def return_to_waiting(self, requests: List[Request]):
        cancelled = []
        with self._lock:
            for request in reversed(requests):
                if request.status == RequestStatus.ABORTED:
                    self._requests.pop(request.request_id, None)
                    cancelled.append(request)
                else:
                    self.waiting.appendleft(request)
        if self._metrics is not None:
            for request in cancelled:
                self._metrics.mark_finished(
                    request.request_id, request.input_tokens, request.output_tokens
                )

    def has_requests(self) -> bool:
        with self._lock:
            return bool(self.running or self.waiting)

    def wait_for_requests(self, timeout: float = 1.0):
        with self._lock:
            if self.waiting or self.running:
                return
            self._request_event.clear()
        self._request_event.wait(timeout=timeout)

    def get_running_requests(self) -> List[Request]:
        with self._lock:
            return list(self.running)

    def get_request(self, request_id: str) -> Optional[Request]:
        with self._lock:
            return self._requests.get(request_id)

    def discard(self, request: Request) -> None:
        """Remove a terminal request after its event and execution fence."""
        with self._lock:
            self.waiting = deque(r for r in self.waiting if r is not request)
            self.running = [r for r in self.running if r is not request]
            if self._requests.get(request.request_id) is request:
                self._requests.pop(request.request_id, None)
            self._total_tokens += request.output_tokens

    def wake(self):
        self._request_event.set()
