import pytest
import torch

from astrai.trainer.callbacks.metric_util import GradSNRTracker


def test_grad_snr_is_reported_in_decibels():
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = GradSNRTracker(beta=0.5, eps=1e-8)

    model.weight.grad = torch.tensor([[1.0]])
    tracker.update(model)
    model.weight.grad = torch.tensor([[3.0]])
    tracker.update(model)

    # E[g]^2 / Var(g) = 4 / 1 = 4, which is 6.0206 dB.
    assert tracker.snr == pytest.approx(10.0 * torch.log10(torch.tensor(4.0)).item())
