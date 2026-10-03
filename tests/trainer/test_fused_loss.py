import pytest
import torch
import torch.nn.functional as F

from astrai.extension.kernel.cross_entropy import is_available
from astrai.trainer import strategy as impl
from astrai.trainer.strategy import ForwardResult, SEQStrategy, SFTStrategy


@pytest.mark.parametrize("strategy_class", [SEQStrategy, SFTStrategy])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_fused_loss_cpu_falls_back(strategy_class, smoothing, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU must not enter the CUDA kernel")

    monkeypatch.setattr(impl, "cuda_cross_entropy", forbidden)
    _compare(strategy_class, "cpu", smoothing, fused=True)


def _compare(strategy_class, device, smoothing=0.0, fused=True, empty=False):
    torch.manual_seed(17)
    strategy = strategy_class(
        torch.nn.Linear(1, 1),
        device,
        label_smoothing=smoothing,
        fused_cross_entropy=fused,
    )
    logits = torch.randn(
        2, 7, 1031, device=device, dtype=torch.bfloat16
    ).requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    targets = torch.randint(1031, (2, 7), device=device)
    mask = torch.zeros_like(targets, dtype=torch.bool) if empty else (targets % 3 != 0)
    batch = {"target_ids": targets, "loss_mask": mask}
    effective = (
        targets.masked_fill(~mask, -100) if strategy_class is SFTStrategy else targets
    )
    result = strategy.reduce_loss(ForwardResult(logits), batch)
    expected = F.cross_entropy(
        reference.flatten(0, 1).float(),
        effective.flatten(),
        reduction="sum",
        label_smoothing=smoothing,
    )
    torch.testing.assert_close(result.loss_sum, expected, rtol=1e-6, atol=2e-6)
    assert result.token_count.item() == (effective != -100).sum().item()
    result.mean().backward()
    (expected / (effective != -100).sum().clamp(min=1)).backward()
    torch.testing.assert_close(logits.grad, reference.grad, rtol=0.01, atol=2e-7)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="cross_entropy CUDA kernel unavailable",
)
@pytest.mark.parametrize("strategy_class", [SEQStrategy, SFTStrategy])
@pytest.mark.parametrize("empty", [False, True])
def test_fused_loss_preserves_token_sum_and_count(strategy_class, empty):
    _compare(strategy_class, "cuda", empty=empty)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("reason", ["missing_kernel", "smoothing", "disabled"])
def test_fused_loss_cuda_fallback(reason, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fallback must not enter the CUDA kernel")

    monkeypatch.setattr(impl, "cuda_cross_entropy", forbidden)
    monkeypatch.setattr(impl, "has_cross_entropy", lambda: reason != "missing_kernel")
    _compare(
        SFTStrategy,
        "cuda",
        smoothing=0.1 if reason == "smoothing" else 0.0,
        fused=reason != "disabled",
    )
