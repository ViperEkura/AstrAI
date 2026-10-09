"""Muon parameter and momentum updates using the optional NS kernels."""

import pytest
import torch
from torch import nn

from astrai.extension.kernel.newton_schulz import is_available
from astrai.optim.muon_adamw import MuonAdamW

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="Muon NS CUDA kernels are unavailable",
)


class _Matrix(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        )


@pytest.mark.parametrize(
    "shape", [(256, 1536), (1536, 1536), (1536, 6912), (6912, 1536)]
)
@pytest.mark.parametrize("nesterov", [False, True])
def test_muon_kernel_update_tracks_torch_parameters_and_momentum(shape, nesterov):
    torch.manual_seed(17)
    reference = _Matrix(shape)
    candidate = _Matrix(shape)
    candidate.load_state_dict(reference.state_dict())
    reference_optimizer = MuonAdamW(reference, lr=1e-3, nesterov=nesterov)
    candidate_optimizer = MuonAdamW(
        candidate, lr=1e-3, nesterov=nesterov, use_ns_kernels=True
    )
    for _ in range(3):
        gradient = torch.randn_like(reference.weight)
        reference.weight.grad = gradient.clone()
        candidate.weight.grad = gradient.clone()
        reference_optimizer.step()
        candidate_optimizer.step()
    difference = (reference.weight.float() - candidate.weight.float()).abs()
    assert difference.max().item() <= 0.002
    assert (difference.norm() / reference.weight.float().norm()).item() <= 1e-4
    assert torch.equal(
        reference_optimizer.muon.state[reference.weight]["momentum_buffer"],
        candidate_optimizer.muon.state[candidate.weight]["momentum_buffer"],
    )
    assert (
        reference_optimizer.state_dict().keys()
        == candidate_optimizer.state_dict().keys()
    )
