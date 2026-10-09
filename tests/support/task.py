"""Unit tests for Request and RequestManager."""

from unittest.mock import MagicMock


def _make_mock_tokenizer():
    t = MagicMock()
    t.encode.return_value = [1, 2, 3, 4, 5]
    t.stop_ids = [0]
    return t
