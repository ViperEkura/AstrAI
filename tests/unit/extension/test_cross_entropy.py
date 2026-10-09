"""The native linear CE wrapper rejects CPU inputs."""

import pytest
import torch

from astrai.extension.kernel.cross_entropy import linear_cross_entropy


def test_cpu_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        linear_cross_entropy(
            torch.ones(2, 3), torch.ones(4, 3), torch.zeros(2, dtype=torch.long)
        )
