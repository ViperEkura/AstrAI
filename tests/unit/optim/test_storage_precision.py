"""Persistent formats must match independently rounded FP32 arithmetic."""

import pytest
import torch

from tests.support.precision import check_precision_storage


@pytest.mark.parametrize("kind", ["sgd", "adamw", "muon_adamw"])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("master_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_resident_storage_precision_and_restart(
    kind, compute_dtype, master_dtype, state_dtype
):
    check_precision_storage(
        kind, compute_dtype, master_dtype, state_dtype, "cpu", False
    )
