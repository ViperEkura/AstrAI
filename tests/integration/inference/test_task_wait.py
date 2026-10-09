"""Unit tests for Request and RequestManager."""

import threading
import time

from astrai.inference import (
    RequestManager,
)
from tests.support.task import _make_mock_tokenizer


def test_task_manager_wake():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
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
