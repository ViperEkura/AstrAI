"""Tests for scheduler concurrency."""

import threading
from unittest.mock import MagicMock, patch

import pytest
import torch

from astrai.inference import Scheduler
from astrai.model.autoregressive_lm import AutoRegressiveLM
from tests.support.models import make_rollout_config
from tests.support.tokenizers import FakeTokenizer


class _UnwrappingTokenizer:
    """FakeTokenizer.encode returns the batch shape ([[ids]]); add_request's
    contract is the flat single-string shape, so unwrap it."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "stop_ids", inner.stop_ids)

    def encode(self, prompt, **kw):
        out = self._inner.encode(prompt, **kw)
        return out[0] if isinstance(out[0], list) else out

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture
def mock_model_and_tokenizer():
    """Create mock model and tokenizer."""
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_model.config.num_key_value_heads = 8
    mock_model.config.num_attention_heads = 8
    mock_model.config.hidden_size = 128
    mock_model.config.num_hidden_layers = 2
    mock_model.config.max_position_embeddings = 100
    mock_model.parameters.return_value = iter(
        [MagicMock(dtype=torch.float32, device=torch.device("cpu"))]
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "token"
    mock_tokenizer.stop_ids = [0]
    mock_tokenizer.pad_id = None

    return mock_model, mock_tokenizer


def _make_mock_scheduler(mock_model_and_tokenizer):
    """Build a CPU scheduler over mocks, patching scheduler-internal imports."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer
    with (
        patch("astrai.inference.core.scheduler.AutoModel"),
        patch("astrai.inference.core.scheduler.AutoTokenizer"),
    ):
        return Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            max_batch_size=4,
            device="cpu",
        )


def _run_threads(*workers, timeout=10.0):
    threads = [threading.Thread(target=worker) for worker in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    assert all(not t.is_alive() for t in threads)


def _make_contract_scheduler():
    scheduler, _, _ = _make_real_scheduler("cpu")
    kv = scheduler._kv_manager = MagicMock()
    kv.cached_tokens.return_value = 0
    kv.extend_slots_batch.side_effect = lambda ids, positions: [True] * len(ids)
    return scheduler


def _make_real_scheduler(device):
    """Build a scheduler backed by a tiny real model for run_batch tests."""
    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    tokenizer = FakeTokenizer()
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_seq_len=64,
    )
    return scheduler, tokenizer, model


class _MultiPatch:
    def __init__(self, patches):
        self._patches = patches

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False
