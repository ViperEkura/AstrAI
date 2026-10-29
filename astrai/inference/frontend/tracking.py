"""Frontend request tracking: event queues, stream chunks, finish mapping.

Split out of ``engine.py`` so the facade stays a facade: the tracker owns
the bounded per-request queues the scheduler loop appends into, and the
chunk type carries structured output to protocol adapters.
"""

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

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


class RequestTracker(OutputEventSink):
    """Collect core events and notify blocking or asynchronous consumers.

    Register before submitting a request. Queue updates and completion flags
    share one lock; decoding and protocol formatting stay on consumer threads.
    """

    def __init__(self, maxlen: int = 4096):
        self._lock = threading.Lock()
        self._queues: Dict[str, Deque[Any]] = {}
        self._finished: Dict[str, threading.Event] = {}
        self._async_subscribers: Dict[
            str, Tuple[asyncio.AbstractEventLoop, asyncio.Queue]
        ] = {}
        self._maxlen = maxlen

    def register(
        self, request_id: str, maxlen: Optional[int] = None
    ) -> threading.Event:
        with self._lock:
            bound = self._maxlen if maxlen is None else maxlen
            self._queues[request_id] = deque(maxlen=bound)
            done = self._finished[request_id] = threading.Event()
            return done

    def unregister(self, request_id: str) -> None:
        with self._lock:
            self._queues.pop(request_id, None)
            self._async_subscribers.pop(request_id, None)
            self._finished.pop(request_id, None)

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
                if queue is None or self._finished[rid].is_set():
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
                    self._finished[rid].set()

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

    def is_finished(self, request_id: str) -> bool:
        with self._lock:
            done = self._finished.get(request_id)
            return done is not None and done.is_set()

    def drain(self, request_id: str) -> List[Any]:
        """Pop all pending events for one request (non-blocking)."""
        with self._lock:
            queue = self._queues.get(request_id)
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
