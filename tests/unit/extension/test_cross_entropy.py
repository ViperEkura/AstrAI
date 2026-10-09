"""Independent guard: CE availability must not change other kernel tests."""

import pytest
import torch

from astrai.extension.kernel.cross_entropy import (
    cross_entropy,
    is_available,
)

skip_no_ce = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="CE CUDA kernel not built",
)


def test_cpu_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        cross_entropy(torch.ones(2, 3), torch.zeros(2, dtype=torch.long))
