"""Shared fixtures for inference tests."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from astrai.inference import get_app


@pytest.fixture
def client():
    """Provide a test client for the FastAPI app."""
    _app = get_app()
    _app.state.server_config = {
        "device": "cpu",
        "dtype": "bfloat16",
        "param_path": None,
        "max_batch_size": 1,
        "_test": True,
    }
    _app.state.engine = None
    return TestClient(_app)


@pytest.fixture
def mock_engine():
    """Create a mock InferenceEngine."""

    async def _async_gen():
        yield "chunk1"
        yield "chunk2"
        yield "[DONE]"

    mock = MagicMock()
    mock.generate.return_value = "mock response"
    mock.generate_async.return_value = _async_gen()
    mock.get_stats.return_value = {
        "total_tasks": 0,
        "total_tokens": 0,
        "active_tasks": 0,
        "waiting_queue": 0,
    }
    mock.tokenizer.encode.return_value = [1, 2, 3]
    mock.tokenizer.decode.return_value = "mock response"
    mock.tokenizer.apply_chat_template.return_value = "mock prompt"
    return mock


@pytest.fixture
def loaded_model(client, mock_engine):
    """Simulate that the engine is loaded."""
    get_app().state.engine = mock_engine
    return mock_engine
