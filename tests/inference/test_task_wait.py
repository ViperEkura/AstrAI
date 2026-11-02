"""Unit tests for Request and Scheduler."""

import threading
import time

from tests.support.task import make_mock_tokenizer, make_scheduler


def test_scheduler_wake():
    tm = make_scheduler(tokenizer=make_mock_tokenizer())
    called = threading.Event()

    def waiter():
        tm.wait_for_requests(timeout=5.0)
        called.set()

    t = threading.Thread(target=waiter)
    t.start()

    time.sleep(0.05)
    tm.wake()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert called.is_set()
