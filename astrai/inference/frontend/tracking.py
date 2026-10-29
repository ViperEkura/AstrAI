"""Frontend request tracking: event queues, stream chunks, finish mapping.

Split out of ``engine.py`` so the facade stays a facade: the tracker owns
the bounded per-request queues the scheduler loop appends into, and the
chunk type carries structured output to protocol adapters.
"""

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from astrai.inference.core.events import (
    FINISH_ABORTED,
    FINISH_CANCELLED,
    FINISH_LENGTH,
    FINISH_REJECTED,
    FINISH_STOP_TOKEN,
    RequestError,
    RequestFinished,
)
from astrai.inference.core.scheduler import OutputEventSink


def _deliver_async_events(deliveries) -> None:
    for queue, events in deliveries:
        queue.put_nowait(events)


class EventQueueSink(OutputEventSink):
    """Bounded per-request event queues, fed by the scheduler loop thread.

    The scheduler thread appends to bounded queues for blocking consumers
    and forwards subscribed async streams to their event loop. Consumer code
    (detokenize, protocol formatting, user callbacks) runs outside the sink.

    When a terminal event (:class:`RequestFinished` / :class:`RequestError`)
    is queued, ``on_terminal`` fires for that request — the core's event
    contract guarantees exactly one terminal event, queued after all of the
    request's tokens, so the flag doubles as a completion signal for
    consumers that only want the final result.
    """

    def __init__(
        self,
        maxlen: int = 4096,
        on_terminal: Optional[Callable[[str], None]] = None,
    ):
        self._lock = threading.Lock()
        self._queues: Dict[str, Deque[Any]] = {}
        self._terminal: Set[str] = set()
        self._async_subscribers: Dict[
            str, Tuple[asyncio.AbstractEventLoop, asyncio.Queue]
        ] = {}
        self._maxlen = maxlen
        self._on_terminal = on_terminal

    def register(self, request_id: str, maxlen: Optional[int] = None) -> None:
        with self._lock:
            bound = self._maxlen if maxlen is None else maxlen
            self._queues[request_id] = deque(maxlen=bound)
            self._terminal.discard(request_id)

    def unregister(self, request_id: str) -> None:
        with self._lock:
            self._queues.pop(request_id, None)
            self._async_subscribers.pop(request_id, None)
            self._terminal.discard(request_id)

    def subscribe_async(
        self, request_id: str, loop: asyncio.AbstractEventLoop
    ) -> Tuple[asyncio.Queue, List[Any]]:
        queue = asyncio.Queue()
        with self._lock:
            events = self._queues.get(request_id)
            if events is None:
                raise KeyError(f"request {request_id!r} is not registered")
            pending = list(events)
            events.clear()
            self._async_subscribers[request_id] = (loop, queue)
        return queue, pending

    def unsubscribe_async(self, request_id: str) -> None:
        with self._lock:
            self._async_subscribers.pop(request_id, None)

    def __call__(self, events: List[Any]) -> None:
        by_loop: Dict[
            asyncio.AbstractEventLoop, List[Tuple[str, asyncio.Queue, List[Any]]]
        ] = {}
        with self._lock:
            for event in events:
                rid = event.request_id
                queue = self._queues.get(rid)
                if queue is None or rid in self._terminal:
                    continue
                subscriber = self._async_subscribers.get(rid)
                if subscriber is not None:
                    loop, async_queue = subscriber
                    deliveries = by_loop.setdefault(loop, [])
                    if deliveries and deliveries[-1][1] is async_queue:
                        deliveries[-1][2].append(event)
                    else:
                        deliveries.append((rid, async_queue, [event]))
                else:
                    queue.append(event)
                if isinstance(event, (RequestFinished, RequestError)):
                    self._terminal.add(rid)
                    # Publish the core terminal fact before a draining caller
                    # or async delivery can observe it. This bookkeeping-only
                    # callback must not re-enter the sink.
                    if self._on_terminal is not None:
                        self._on_terminal(rid)

        for loop, deliveries in by_loop.items():
            scheduled = [(queue, batch) for _, queue, batch in deliveries]
            try:
                loop.call_soon_threadsafe(_deliver_async_events, scheduled)
            except RuntimeError:
                with self._lock:
                    restored = set()
                    for rid, queue, batch in deliveries:
                        if rid not in restored:
                            if self._async_subscribers.get(rid) != (loop, queue):
                                continue
                            self._async_subscribers.pop(rid, None)
                            restored.add(rid)
                        pending = self._queues.get(rid)
                        if pending is not None:
                            pending.extend(batch)


class RequestTracker:
    """Frontend bookkeeping: request_id -> queue + lifecycle flag.

    Mirrors vLLM's ``RequestTracker``: the engine mints the id, registers
    the queue BEFORE the request reaches the scheduler, and therefore no
    event can ever precede its consumer (the old ``_ResultSink`` replay
    buffer existed only because ids were minted in the core).
    """

    def __init__(self):
        # Terminal events flip the finished flag as they are queued (sink
        # callback), so ``wait`` / ``is_finished`` are true the moment the
        # scheduler has finished a request — independent of any consumer.
        self._sink = EventQueueSink(on_terminal=self.mark_finished)
        self._finished: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    @property
    def sink(self) -> EventQueueSink:
        return self._sink

    def register(
        self, request_id: str, maxlen: Optional[int] = None
    ) -> threading.Event:
        with self._lock:
            done = self._finished[request_id] = threading.Event()
        self._sink.register(request_id, maxlen=maxlen)
        return done

    def unregister(self, request_id: str) -> None:
        self._sink.unregister(request_id)
        with self._lock:
            self._finished.pop(request_id, None)

    def is_finished(self, request_id: str) -> bool:
        with self._lock:
            done = self._finished.get(request_id)
        return done is not None and done.is_set()

    def mark_finished(self, request_id: str) -> None:
        with self._lock:
            done = self._finished.get(request_id)
        if done is not None:
            done.set()

    def subscribe_async(
        self, request_id: str, loop: asyncio.AbstractEventLoop
    ) -> Tuple[asyncio.Queue, List[Any]]:
        return self._sink.subscribe_async(request_id, loop)

    def unsubscribe_async(self, request_id: str) -> None:
        self._sink.unsubscribe_async(request_id)

    def drain(self, request_id: str) -> List[Any]:
        """Pop all pending events for one request (non-blocking)."""
        with self._sink._lock:
            queue = self._sink._queues.get(request_id)
            if queue is None:
                return []
            out = list(queue)
            queue.clear()
        return out

    def wait(self, request_id: str, timeout: Optional[float] = None) -> bool:
        with self._lock:
            done = self._finished.get(request_id)
        if done is None:
            return True
        return done.wait(timeout=timeout)


@dataclass(slots=True, eq=False, repr=False, match_args=False)
class StreamChunk:
    """Incremental text and token IDs, plus terminal facts on the final chunk."""

    text: str
    delta_token_ids: List[int]
    stopped: bool
    finish_reason: Optional[str] = field(default=None, init=False)
    prompt_tokens: int = field(default=0, init=False)
    completion_tokens: int = field(default=0, init=False)
    stop_sequence: Optional[str] = field(default=None, init=False)

    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None


def map_finish_reason(reason: Optional[str]) -> str:
    """Internal event reasons → protocol-neutral finish vocabulary."""
    mapping = {
        FINISH_STOP_TOKEN: "stop",
        FINISH_LENGTH: "length",
        FINISH_CANCELLED: "cancelled",
        FINISH_ABORTED: "aborted",
        FINISH_REJECTED: "rejected",
    }
    return mapping.get(reason or "", "stop")
