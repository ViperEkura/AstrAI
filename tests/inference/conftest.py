"""Shared fixtures for inference tests."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from astrai.inference import get_app


@pytest.fixture
def _cleanup_app_engine():
    """Reset the lazy FastAPI singleton engine after each inference test."""
    app = get_app()
    engine = getattr(app.state, "engine", None)
    config = getattr(app.state, "server_config", None)
    try:
        yield
    finally:
        app.state.engine = engine
        app.state.server_config = config


@pytest.fixture
def client(_cleanup_app_engine):
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
    engine = MagicMock()
    engine.tokenizer.apply_chat_template.return_value = "mock prompt"
    return engine


@pytest.fixture
def loaded_model(client, mock_engine):
    """Simulate that the engine is loaded."""
    get_app().state.engine = mock_engine
    return mock_engine
