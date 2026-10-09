"""Batched symmetric BLAS isolation, layouts and Graph replay."""

import pytest
import torch

from astrai.extension.backend.newton_schulz import symm_out, syrk_out
from astrai.extension.kernel import newton_schulz as cuda
from astrai.extension.policy import newton_schulz as plan

CUDA_AVAILABLE = torch.cuda.is_available() and cuda.is_available()
CUDA_ONLY = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="symmetric CUDA extension unavailable"
)


def _layout(x, column):
    if column:
        return x.transpose(-2, -1).contiguous().transpose(-2, -1)
    return x.contiguous()


def _reference(operation, symmetric, x, addend, alpha, beta):
    results = []
    for index in range(x.size(0)):
        left = x[index] if operation == "syrk" else symmetric[index]
        right = x[index].T if operation == "syrk" else x[index]
        if beta:
            results.append(
                torch.addmm(addend[index], left, right, alpha=alpha, beta=beta)
            )
        else:
            results.append(alpha * (left @ right))
    return torch.stack(results)


@pytest.mark.parametrize("operation", ["syrk", "symm"])
def test_cpu_batch_matches_independent_matrices(operation):
    torch.manual_seed(71)
    x = torch.randn(3, 7, 13, dtype=torch.float64)
    symmetric = torch.randn(3, 7, 7, dtype=x.dtype)
    symmetric = (symmetric + symmetric.transpose(-2, -1)) / 2
    shape = symmetric.shape if operation == "syrk" else x.shape
    output = torch.empty(shape, dtype=x.dtype)
    addend = torch.randn(shape, dtype=x.dtype)
    if operation == "syrk":
        addend = (addend + addend.transpose(-2, -1)) / 2
        syrk_out(x, output, addend=addend, alpha=-0.7, beta=0.3)
    else:
        symm_out(symmetric, x, output, addend=addend, alpha=-0.7, beta=0.3)
    torch.testing.assert_close(
        output, _reference(operation, symmetric, x, addend, -0.7, 0.3)
    )


def test_batch_contract_rejects_broadcast_and_mismatched_addend():
    x = torch.randn(3, 7, 13)
    symmetric = torch.eye(7).unsqueeze(0)
    output = torch.empty_like(x)
    with pytest.raises(ValueError, match="shape"):
        symm_out(symmetric, x, output)
    with pytest.raises(ValueError, match="shape"):
        syrk_out(x, torch.empty(2, 7, 7))
    with pytest.raises(ValueError, match="addend shape"):
        syrk_out(x, torch.empty(3, 7, 7), addend=torch.empty(1, 7, 7), beta=1)
    gram = torch.empty(3, 7, 7)
    with pytest.raises(ValueError, match="alias"):
        syrk_out(gram, gram)


@pytest.mark.parametrize("batch_size", [0, 65536])
def test_invalid_batch_plan_configuration_preserves_previous_table(batch_size):
    before = plan.configure()
    with pytest.raises(ValueError, match="geometry"):
        plan.configure(
            [
                dict(
                    operation="syrk",
                    cc=0,
                    rows=64,
                    cols=128,
                    batch_size=batch_size,
                    backend="torch",
                )
            ]
        )
    assert plan.configure() == before
