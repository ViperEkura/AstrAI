"""Unit tests for InferenceEngine.generate() and the frontend event protocol."""

from unittest.mock import MagicMock

from astrai.inference.core.events import (
    FINISH_CANCELLED,
    FINISH_LENGTH,
    RequestError,
    RequestFinished,
    TokenDelta,
)
from astrai.inference.frontend.tracking import RequestTracker


def test_request_tracker_events_only_reach_registered_requests():
    """The frontend mints ids before submission, so events for unregistered
    ids are dropped by the sink (they cannot exist on the happy path);
    once registered, events land in the per-request queue in order."""
    tracker = RequestTracker()
    done = tracker.register("t0")
    tracker.sink([TokenDelta("t0", 11, 1), TokenDelta("t0", 12, 2)])
    events = tracker.drain("t0")
    assert [e.token_id for e in events] == [11, 12]
    tracker.sink([TokenFinished := RequestFinished("t0", FINISH_LENGTH, 3, 2)])
    assert tracker.drain("t0") == [TokenFinished]
    tracker.mark_finished("t0")
    assert done.is_set()
    tracker.unregister("t0")


def test_tracker_accepts_one_terminal_and_reclaims_all_state():
    tracker = RequestTracker()
    done = tracker.register("r", maxlen=4)
    terminal_callback = MagicMock(wraps=tracker.mark_finished)
    tracker.sink._on_terminal = terminal_callback
    terminal = RequestFinished("r", FINISH_CANCELLED, 2, 1)
    tracker.sink([TokenDelta("r", 11, 1), terminal, terminal, TokenDelta("r", 12, 2)])
    tracker.sink([RequestError("r", "late", "ignored")])
    assert done.is_set()
    assert tracker.drain("r") == [TokenDelta("r", 11, 1), terminal]
    terminal_callback.assert_called_once_with("r")
    assert tracker.sink._queues["r"].maxlen == 4
    tracker.unregister("r")
    tracker.sink([terminal])
    assert not tracker._finished
    assert not tracker.sink._queues
    assert not tracker.sink._terminal
    terminal_callback.assert_called_once()
    replacement = tracker.register("r")
    assert not replacement.is_set()
    tracker.sink([terminal])
    assert replacement.is_set()
    assert tracker.drain("r") == [terminal]
    tracker.unregister("r")


def test_tracker_publishes_core_terminal_before_async_delivery():
    tracker = RequestTracker()
    tracker.register("r")
    loop = MagicMock()
    queue, pending = tracker.subscribe_async("r", loop)
    assert pending == []

    def deliver(callback, deliveries):
        assert tracker.is_finished("r")
        callback(deliveries)

    loop.call_soon_threadsafe.side_effect = deliver
    terminal = RequestFinished("r", FINISH_LENGTH)
    tracker.sink([terminal])
    assert queue.get_nowait() == [terminal]
    tracker.unregister("r")


def test_tracker_closed_loop_preserves_interleaved_terminal_backlog():
    tracker = RequestTracker()
    loop = MagicMock()
    loop.call_soon_threadsafe.side_effect = RuntimeError("loop closed")
    for rid in ("a", "b"):
        tracker.register(rid)
        tracker.subscribe_async(rid, loop)
    tracker.sink(
        [
            TokenDelta("a", 11, 1),
            TokenDelta("b", 12, 1),
            RequestFinished("a", FINISH_LENGTH, 2, 1),
            RequestFinished("b", FINISH_CANCELLED, 2, 1),
        ]
    )
    for rid, token, reason in (("a", 11, FINISH_LENGTH), ("b", 12, FINISH_CANCELLED)):
        assert tracker.is_finished(rid)
        assert tracker.drain(rid) == [
            TokenDelta(rid, token, 1),
            RequestFinished(rid, reason, 2, 1),
        ]
        tracker.unregister(rid)
    assert not tracker.sink._async_subscribers
    assert not tracker.sink._terminal
