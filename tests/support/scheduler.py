"""Tests for scheduler concurrency."""

import threading
from unittest.mock import MagicMock

import pytest
import torch

from astrai.inference import EngineCore
from astrai.model.autoregressive_lm import AutoRegressiveLM
from tests.support.models import make_rollout_config
from tests.support.tokenizers import FakeTokenizer


@pytest.fixture
def mock_model_and_tokenizer():
    """Create mock model and tokenizer."""
    mock_model = MagicMock()
    mock_model.generation_config = {"eos_token_id": 0}
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
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.pad_id = None

    return mock_model, mock_tokenizer


def make_mock_scheduler(mock_model_and_tokenizer):
    """Build a CPU scheduler over mocks, patching scheduler-internal imports."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer
    return EngineCore(
        model=mock_model,
        tokenizer=mock_tokenizer,
        max_batch_size=4,
        device="cpu",
    )


def run_threads(*workers, timeout=10.0):
    errors = []

    def run(worker):
        try:
            worker()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(worker,)) for worker in workers]
    try:
        for thread in threads:
            thread.start()
    finally:
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=timeout)
    assert all(not thread.is_alive() for thread in threads)
    if errors:
        raise errors[0]


def make_contract_scheduler():
    scheduler, _, _ = make_real_scheduler("cpu")
    kv = MagicMock()
    scheduler.scheduler._kv_manager = kv
    scheduler.scheduler._planner.kv_manager = kv
    kv.cached_tokens.return_value = 0
    kv.extend_slots_batch.side_effect = lambda ids, positions: [True] * len(ids)
    return scheduler


def make_real_scheduler(device):
    """Build a scheduler backed by a tiny real model for run_batch tests."""
    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    tokenizer = FakeTokenizer()
    model.generation_config = {"eos_token_id": tokenizer.eos_token_id}
    scheduler = EngineCore(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_seq_len=64,
    )
    return scheduler, tokenizer, model
