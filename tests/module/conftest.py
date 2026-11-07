import pytest
import torch


@pytest.fixture(scope="session")
def cuda_device():
    """CUDA device only for explicitly marked kernel tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"
