"""Pinned lower-precision storage still performs FP32 GPU updates."""

import pytest
import torch

from tests.support.capabilities import skip_no_cuda
from tests.support.precision import check_precision_storage


@skip_no_cuda
@pytest.mark.parametrize("kind", ["sgd", "adamw", "muon_adamw"])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("master_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_offloaded_storage_precision_and_restart(
    kind, compute_dtype, master_dtype, state_dtype
):
    check_precision_storage(
        kind, compute_dtype, master_dtype, state_dtype, "cuda", True
    )
