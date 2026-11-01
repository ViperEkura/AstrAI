"""Unit tests for request admission and lifecycle."""

from unittest.mock import MagicMock

from astrai.inference.core.scheduler import Scheduler
from tests.support.cache import _make_contiguous_pool


def _make_mock_tokenizer():
    t = MagicMock()
    t.encode.return_value = [1, 2, 3, 4, 5]
    t.stop_ids = [0]
    return t


def _make_scheduler(tokenizer=None, max_seq_len=64, max_batch_size=4, metrics=None):
    cache = _make_contiguous_pool(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size
    )
    scheduler = Scheduler(
        tokenizer or _make_mock_tokenizer(), max_batch_size, max_seq_len, cache
    )
    if metrics is not None:
        scheduler._metrics = metrics
    return scheduler
