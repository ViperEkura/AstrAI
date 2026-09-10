import json
import os

import pytest
import torch

from astrai.extension import KERNEL_NAMES, is_available
from astrai.model.transformer import AutoRegressiveLM
from tests.helpers import (
    TINY_CONFIG,
    RandomTokenDataset,
    build_test_tokenizer,
    make_tiny_config,
)

CUDA_AVAIL = torch.cuda.is_available()
KERNEL_AVAIL = CUDA_AVAIL and all(is_available(k) for k in KERNEL_NAMES)
FP8_AVAIL = (
    CUDA_AVAIL
    and is_available("quantize")
    and torch.cuda.get_device_capability() >= (8, 9)
)
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAIL, reason="CUDA not available")
skip_lt2_cuda = pytest.mark.skipif(
    not CUDA_AVAIL or torch.cuda.device_count() < 2,
    reason="two-rank spawn tests need two CUDA devices",
)
skip_no_kernel = pytest.mark.skipif(not KERNEL_AVAIL, reason="CUDA kernels not built")
skip_no_fp8 = pytest.mark.skipif(
    not FP8_AVAIL,
    reason="fused FP8 MMA requires a built kernel and compute capability 8.9+",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "unit: fast unit tests")


@pytest.fixture(scope="session")
def device():
    """Session-scoped device string (``"cuda"`` if available, else ``"cpu"``)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_test_tokenizer(vocab_size: int = 1000):
    """Create a simple tokenizer for testing purposes."""
    return build_test_tokenizer(vocab_size)


@pytest.fixture(scope="session")
def test_tokenizer():
    """Session-scoped tokenizer, created once for the entire test run."""
    return create_test_tokenizer()


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
