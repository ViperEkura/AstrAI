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


@dataclass
class TrackedRequest:
    events: Deque[Any]
    finished: threading.Event = field(default_factory=threading.Event)
    subscriber: Optional[Tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = None


class RequestTracker(OutputEventSink):
    """Collect core events and notify blocking or asynchronous consumers.

    Register before submitting a request. Queue updates and completion flags
    share one lock; decoding and protocol formatting stay on consumer threads.
    """

    def __init__(self, event_capacity: int = 4096):
        self._condition = threading.Condition(threading.Lock())
        self._requests: Dict[str, TrackedRequest] = {}
        self._event_capacity = event_capacity

    def register(
        self, request_id: str, event_capacity: Optional[int] = None
    ) -> threading.Event:
        with self._condition:
            bound = self._event_capacity if event_capacity is None else event_capacity
            state = TrackedRequest(deque(maxlen=bound))
            self._requests[request_id] = state
            return state.finished

    def unregister(self, request_id: str) -> None:
        with self._condition:
            self._requests.pop(request_id, None)
            self._condition.notify_all()

    def subscribe_async(
        self, request_id: str, loop: asyncio.AbstractEventLoop
    ) -> Tuple[asyncio.Queue, List[Any]]:
        queue = asyncio.Queue()
        with self._condition:
            state = self._requests.get(request_id)
            if state is None:
                raise KeyError(f"request {request_id!r} is not registered")
            pending = list(state.events)
            state.events.clear()
            state.subscriber = (loop, queue)
        return queue, pending

    def unsubscribe_async(self, request_id: str) -> None:
        with self._condition:
            state = self._requests.get(request_id)
            if state is not None:
                state.subscriber = None

    def __call__(self, events: List[Any]) -> None:
        by_loop: Dict[
            asyncio.AbstractEventLoop, List[Tuple[str, asyncio.Queue, List[Any]]]
        ] = {}
        with self._condition:
            for event in events:
                rid = event.request_id
                state = self._requests.get(rid)
                if state is None or state.finished.is_set():
                    continue
                subscriber = state.subscriber
                if subscriber is not None:
                    loop, async_queue = subscriber
                    deliveries = by_loop.setdefault(loop, [])
                    if deliveries and deliveries[-1][1] is async_queue:
                        deliveries[-1][2].append(event)
                    else:
                        deliveries.append((rid, async_queue, [event]))
                else:
                    state.events.append(event)
                if isinstance(event, (RequestFinished, RequestError)):
                    state.finished.set()
            self._condition.notify_all()

        for loop, deliveries in by_loop.items():
            scheduled = [(queue, batch) for _, queue, batch in deliveries]
            try:
                loop.call_soon_threadsafe(_deliver_async_events, scheduled)
            except RuntimeError:
                with self._condition:
                    restored = set()
                    for rid, queue, batch in deliveries:
                        state = self._requests.get(rid)
                        if state is None:
                            continue
                        if rid not in restored:
                            if state.subscriber != (loop, queue):
                                continue
                            state.subscriber = None
                            restored.add(rid)
                        state.events.extend(batch)
                    self._condition.notify_all()

    def is_finished(self, request_id: str) -> bool:
        with self._condition:
            state = self._requests.get(request_id)
            return state is not None and state.finished.is_set()

    def drain(self, request_id: str) -> List[Any]:
        """Pop all pending events for one request (non-blocking)."""
        with self._condition:
            state = self._requests.get(request_id)
            if state is None:
                return []
            out = list(state.events)
            state.events.clear()
        return out

    def wait_for_events(
        self, request_ids: List[str], timeout: Optional[float] = None
    ) -> bool:
        """Wait until any requested stream has events ready to drain."""
        with self._condition:
            return self._condition.wait_for(
                lambda: any(
                    rid not in self._requests or self._requests[rid].events
                    for rid in request_ids
                ),
                timeout=timeout,
            )

    def wait(self, request_id: str, timeout: Optional[float] = None) -> bool:
        with self._condition:
            state = self._requests.get(request_id)
        return state is None or state.finished.wait(timeout=timeout)


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
