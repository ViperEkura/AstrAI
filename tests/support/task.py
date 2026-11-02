"""Unit tests for request admission and lifecycle."""

from unittest.mock import MagicMock

from astrai.inference.core.scheduler import Scheduler
from tests.support.cache import make_contiguous_pool


def make_mock_tokenizer():
    t = MagicMock()
    t.encode.return_value = [1, 2, 3, 4, 5]
    t.eos_token_id = 0
    return t


def make_scheduler(tokenizer=None, max_seq_len=64, max_batch_size=4, metrics=None):
    cache = make_contiguous_pool(max_seq_len=max_seq_len, max_batch_size=max_batch_size)
    scheduler = Scheduler(
        tokenizer or make_mock_tokenizer(),
        max_batch_size,
        max_seq_len,
        cache,
        generation_config={"eos_token_id": 0},
    )
    if metrics is not None:
        scheduler._metrics = metrics
    return scheduler
