import json
import os

import pytest

from astrai.model.autoregressive_lm import AutoRegressiveLM
from tests.support.models import TINY_CONFIG, make_tiny_config
from tests.support.rollout import RandomTokenDataset
from tests.support.tokenizers import build_test_tokenizer


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "unit: fast unit tests")


@pytest.fixture(scope="session")
def device():
    """CPU device for shared fixtures (``"cuda"`` ``"cpu"``)."""
    return "cpu"


@pytest.fixture(scope="session")
def test_tokenizer():
    """Session-scoped tokenizer, created once for the entire test run."""
    return build_test_tokenizer()


@pytest.fixture
def test_model(device):
    """Function-scoped small AutoRegressiveLM model, isolated per test."""
    config = make_tiny_config()
    model = AutoRegressiveLM(config).to(device=device)
    return {"model": model, "device": device, "config": config}


@pytest.fixture
def temp_dir(tmp_path):
    """Function-scoped temporary directory, cleaned up by pytest."""
    return str(tmp_path)


@pytest.fixture
def base_test_env(test_model, test_tokenizer, temp_dir):
    """Function-scoped test environment with isolated temp directory."""
    config_path = os.path.join(temp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(TINY_CONFIG, f)

    return {
        "device": test_model["device"],
        "test_dir": temp_dir,
        "config_path": config_path,
        "transformer_config": test_model["config"],
        "model": test_model["model"],
        "tokenizer": test_tokenizer,
    }


@pytest.fixture
def random_dataset():
    return RandomTokenDataset(length=None)


@pytest.fixture
def multi_turn_dataset():
    return RandomTokenDataset(length=None, with_loss_mask=True)


@pytest.fixture
def early_stopping_dataset():
    return RandomTokenDataset(length=10, stop_after=5)
