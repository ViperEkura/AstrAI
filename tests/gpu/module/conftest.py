import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """CUDA device for GPU-tier tests using the shared device fixture."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"
