import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

from tokenizers.decoders import DecodeStream

from astrai.config.inference_config import InferenceConfig
from astrai.inference.metrics import MetricsCollector
from astrai.tokenize.tokenizer import AutoTokenizer

if TYPE_CHECKING:
    from astrai.extension import AttentionBackend

STOP = object()
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
        self._tok = tokenizer._tokenizer
        self._stream = DecodeStream(skip_special_tokens=True)

    def push(self, token_id: int) -> str:
        """Append a token ID and return newly completed text.

        Returns "" while a multi-byte character is still incomplete.
        """
        chunk = self._stream.step(self._tok, token_id)
        return chunk or ""


class TaskStatus(Enum):
    """Task lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class Task:
    """Single generation request: prompt, sampling params, output state."""

    def __init__(
        self,
        task_id: str,
        prompt_ids: List[int],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        frequency_penalty: float = 0.0,
        rep_window: int = _config.default_rep_window,
        backend: Optional["AttentionBackend"] = None,
    ):
        self.task_id = task_id
        self.prompt_ids = prompt_ids
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.rep_window = rep_window
        self.backend = backend

        self.status = TaskStatus.PENDING
        self.output_ids: List[int] = []
        self.output_logprobs: List[float] = []
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self._kv_len: int = 0
        self._decoder: Optional[StreamDecoder] = None

    def mark_prefill_done(self):
        """Prompt KV is materialized by prefill; first output sampled but
        not yet written to KV."""
        self._kv_len = self.input_tokens

    def advance_kv(self):
        """One more position written to KV (after a decode forward)."""
        self._kv_len += 1

    def decode_new_token(self, tokenizer: AutoTokenizer) -> str:
        """Decode the last appended output token, buffering incomplete
        multi-byte sequences across calls.

        Lazily creates a :class:`StreamDecoder` on first use.
        """
        if self._decoder is None:
            self._decoder = StreamDecoder(tokenizer)
        return self._decoder.push(self.output_ids[-1])

    @property
    def next_pos(self) -> int:
        """KV position where the next decode step will write."""
        return self._kv_len

    @property
    def prefill_done(self) -> bool:
        """True when all prompt KV entries are materialized."""
        return self._kv_len >= self.input_tokens > 0

    def is_finished(self, stop_ids: List[int]) -> bool:
        if self.max_tokens is not None and self.output_tokens >= self.max_tokens:
            return True
        if self.output_ids and self.output_ids[-1] in stop_ids:
            return True
        return False


class BatchedStreamCallback(ABC):
    """Stream sink that receives a whole scheduler step's events in one call.

    The scheduling loop dispatches once per decode step: every
    ``(task_id, token)`` event routed to the same sink object is delivered
    as a single list, so batch-aware consumers take their lock and wake
    waiters once per step instead of once per token. Plain per-token
    callbacks keep the ``Callable[[str], None]`` contract.
    """

    @abstractmethod
    def __call__(self, events: List[Tuple[str, Any]]) -> None:
        """Consume ``[(task_id, token), ...]`` produced by one decode step."""
        raise NotImplementedError


class TaskManager:
    """Thread-safe task queues and lifecycle transitions (no page ops)."""

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

        self.waiting_queue: Deque[Task] = deque()
        self.active_tasks: List[Task] = []
        self._callbacks: Dict[str, Callable[[str], None]] = {}
        self._tasks: Dict[str, Task] = {}

        self._task_event = threading.Event()
        self._lock = threading.Lock()

        self._total_tasks = 0
        self._total_tokens = 0
        self._cancelled_total = 0

        self._metrics = metrics

    def add_task(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        frequency_penalty: float = 0.0,
        rep_window: int = 64,
        backend: Optional["AttentionBackend"] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        task_id = f"task_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            # An empty prompt never completes prefill (``prefill_done`` stays
            # False) and would crash the decode path on ``prompt_ids[-1]``;
            # rejecting it here keeps the scheduling loop alive.
            raise ValueError("prompt encoded to zero tokens; refusing to schedule")
        if len(prompt_ids) > self.max_seq_len:
            prompt_ids = prompt_ids[-self.max_seq_len :]

        if max_tokens is None:
            max_tokens = self.max_seq_len - len(prompt_ids)
        else:
            max_tokens = min(max_tokens, self.max_seq_len - len(prompt_ids))

        task = Task(
            task_id=task_id,
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            rep_window=rep_window,
            backend=backend,
        )

        with self._lock:
            self.waiting_queue.append(task)
            self._tasks[task_id] = task
            self._total_tasks += 1
            if stream_callback:
                self._callbacks[task_id] = stream_callback

        if self._metrics is not None:
            self._metrics.register(task_id)

        self._task_event.set()
        return task_id

    def cancel_task(self, task_id: str) -> Tuple[List[Task], bool]:
        """Mark a task cancelled and return tasks safe to clean immediately.

        Registered stream callbacks receive the terminal ``STOP`` sentinel
        for every live cancellation: the scheduling loop drains ABORTED
        tasks without invoking callbacks, so skipping it here would leave
        consumers (e.g. ``GenerateResult.wait_completion``) waiting forever.
        """
        callback = None
        cancelled = False
        immediate: List[Task] = []
        with self._lock:
            task = self._tasks.get(task_id)
            callback = self._callbacks.pop(task_id, None)
            if task is None or task.status in (
                TaskStatus.FINISHED,
                TaskStatus.ABORTED,
            ):
                return [], False

            task.status = TaskStatus.ABORTED
            self._cancelled_total += 1
            cancelled = True
            if task in self.waiting_queue:
                self.waiting_queue = deque(
                    waiting for waiting in self.waiting_queue if waiting is not task
                )
                self._tasks.pop(task_id, None)
                immediate = [task]

        if cancelled and callback is not None:
            if isinstance(callback, BatchedStreamCallback):
                callback([(task_id, STOP)])
            else:
                callback(STOP)
        return immediate, cancelled

    def remove_task(self, task_id: str) -> List[Task]:
        """Backward-compatible alias for cancellation."""
        immediate, _ = self.cancel_task(task_id)
        return immediate

    def invoke_callback(self, task_id: str, token: Any):
        with self._lock:
            cb = self._callbacks.get(task_id)
        if isinstance(cb, BatchedStreamCallback):
            cb([(task_id, token)])
        elif cb:
            cb(token)

    def invoke_callbacks(self, events: List[Tuple[str, Any]]) -> None:
        """Dispatch one decode step's ``(task_id, token)`` events.

        Callbacks resolve under a single lock acquisition; events aimed at
        the same batched sink are delivered as one list (one consumer-side
        lock/notify per step), while plain per-token callbacks receive one
        call per event.
        """
        grouped: Dict[int, Tuple[BatchedStreamCallback, List[Any]]] = {}
        plain: List[Tuple[Callable[[str], None], Any]] = []
        with self._lock:
            for task_id, token in events:
                cb = self._callbacks.get(task_id)
                if cb is None:
                    continue
                if isinstance(cb, BatchedStreamCallback):
                    entry = grouped.get(id(cb))
                    if entry is None:
                        grouped[id(cb)] = (cb, [(task_id, token)])
                    else:
                        entry[1].append((task_id, token))
                else:
                    plain.append((cb, token))
        for cb, batch in grouped.values():
            cb(batch)
        for cb, token in plain:
            cb(token)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            waiting = len(self.waiting_queue)
            stats: Dict[str, Any] = {
                "total_tasks": self._total_tasks,
                "total_tokens": self._total_tokens,
                "active_tasks": len(self.active_tasks),
                "waiting_tasks": waiting,
                "waiting_queue": waiting,
                "cancelled_total": self._cancelled_total,
            }
        if self._metrics is not None:
            stats.update(self._metrics.get_stats())
        return stats

    def remove_finished_tasks(self, stop_ids: List[int]) -> List[Task]:
        with self._lock:
            finished = []
            for task in self.active_tasks:
                if task.status == TaskStatus.ABORTED:
                    finished.append(task)
                elif task.is_finished(stop_ids):
                    task.status = TaskStatus.FINISHED
                    finished.append(task)
                    self._total_tokens += task.output_tokens

            self.active_tasks = [
                t
                for t in self.active_tasks
                if t.status not in (TaskStatus.FINISHED, TaskStatus.ABORTED)
            ]
            for task in finished:
                self._tasks.pop(task.task_id, None)
                self._callbacks.pop(task.task_id, None)

        if self._metrics is not None:
            for task in finished:
                self._metrics.mark_finished(
                    task.task_id, task.input_tokens, task.output_tokens
                )
        return finished

    def pull_candidates(self, n: int) -> List[Task]:
        to_add: List[Task] = []
        with self._lock:
            take = min(n, len(self.waiting_queue))
            for _ in range(take):
                to_add.append(self.waiting_queue.popleft())
        return to_add

    def activate(self, task: Task) -> bool:
        with self._lock:
            if task.status == TaskStatus.ABORTED:
                self._tasks.pop(task.task_id, None)
                self._callbacks.pop(task.task_id, None)
                return False
            task.status = TaskStatus.RUNNING
            self.active_tasks.append(task)
            return True

    def return_to_waiting(self, tasks: List[Task]):
        cancelled = []
        with self._lock:
            for task in reversed(tasks):
                if task.status == TaskStatus.ABORTED:
                    self._tasks.pop(task.task_id, None)
                    self._callbacks.pop(task.task_id, None)
                    cancelled.append(task)
                else:
                    self.waiting_queue.appendleft(task)
        if self._metrics is not None:
            for task in cancelled:
                self._metrics.mark_finished(
                    task.task_id, task.input_tokens, task.output_tokens
                )

    def has_work(self) -> bool:
        with self._lock:
            return bool(self.active_tasks or self.waiting_queue)

    def wait_for_tasks(self, timeout: float = 1.0):
        with self._lock:
            if self.waiting_queue or self.active_tasks:
                return
            self._task_event.clear()
        self._task_event.wait(timeout=timeout)

    def get_active_tasks(self) -> List[Task]:
        with self._lock:
            return list(self.active_tasks)

    def get_waiting_tasks(self) -> List[Task]:
        with self._lock:
            return list(self.waiting_queue)

    def clear_queues(self):
        with self._lock:
            self.waiting_queue.clear()
            self.active_tasks.clear()
            self._callbacks.clear()
            self._tasks.clear()

    def wake(self):
        self._task_event.set()
