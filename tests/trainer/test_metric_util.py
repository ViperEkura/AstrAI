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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grad_snr_squares_after_promotion(dtype):
    model = torch.nn.Linear(1, 1, bias=False, dtype=dtype)
    tracker = GradSNRTracker(beta=0.5)
    values = []
    for value in (1e20, 3e20):
        model.weight.grad = torch.full_like(model.weight, value)
        values.append(model.weight.grad.double().item())
        tracker.update(model)
    mean = sum(values) / 2
    variance = sum(value**2 for value in values) / 2 - mean**2
    assert tracker.snr == pytest.approx(
        10 * torch.log10(torch.tensor(mean**2 / variance)).item(), abs=1e-5
    )
    assert tracker._first.dtype == torch.float32
    assert tracker._second.dtype == torch.float64


def test_grad_snr_pending_state_is_bounded_discarded_and_resumable():
    model = torch.nn.Linear(100, 100, bias=False)
    tracker = GradSNRTracker(beta=0.5, max_samples=37)
    reference = GradSNRTracker(beta=0.5, max_samples=37)
    for value in (1.0, 3.0):
        model.weight.grad = torch.full_like(model.weight, value)
        tracker.update(model)
        reference.update(model)
    saved = tracker.state_dict()
    model.weight.grad.fill_(100)
    tracker.stage(model)
    assert tracker.observations == 2
    torch.testing.assert_close(
        tracker.state_dict()["first"], saved["first"], rtol=0, atol=0
    )
    assert tracker.snr == reference.snr
    tracker.discard()
    assert tracker.state_dict()["last_step"] == 2
    restored = GradSNRTracker(beta=0.5, max_samples=37)
    restored.load_state_dict(saved)
    for value in (5.0, -2.0):
        model.weight.grad.fill_(value)
        reference.update(model)
        tracker.update(model)
        restored.stage(model)
        # The optimizer may clear or overwrite the gradient before commit.
        model.weight.grad.zero_()
        restored.commit(restored.last_step + 1)
    assert restored._first.numel() == 37
    assert restored.snr == tracker.snr == reference.snr
    torch.testing.assert_close(restored._first, reference._first, rtol=0, atol=0)
    torch.testing.assert_close(restored._second, reference._second, rtol=0, atol=0)
    with pytest.raises(ValueError, match="layout changed"):
        restored.stage(torch.nn.Linear(1, 1, bias=False))


@pytest.mark.parametrize(
    "values, observations",
    [
        ((float("nan"), float("inf")), 0),
        ((0.0, 0.0), 2),
        ((1.0, 1.0), 2),
        ((1.0, -1.0), 2),
    ],
)
def test_grad_snr_omits_invalid_or_unresolved_signal(values, observations):
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = GradSNRTracker(beta=0.5)
    for value in values:
        model.weight.grad = torch.full_like(model.weight, value)
        tracker.update(model)
    assert tracker.snr is None
    assert tracker.observations == observations


@pytest.mark.parametrize(
    "overrides",
    [
        {"cp_size": 2},
        {"tp_size": 2},
        {"dp_mode": "fsdp"},
        {"compile_mode": "default"},
        {"strategy": "grpo"},
    ],
)
def test_grad_snr_rejects_unsupported_training_paths(tmp_path, overrides):
    from tests.support.rollout import RandomTokenDataset
    from tests.support.trainer import create_train_config

    with pytest.raises(ValueError, match="grad_snr supports eager offline"):
        create_train_config(
            RandomTokenDataset(),
            str(tmp_path),
            "cpu",
            metrics=["grad_snr"],
            **overrides,
        )


def test_grad_snr_decay_spans_committed_invalid_observations():
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = GradSNRTracker(beta=0.5)
    for value in (1.0, float("inf"), 3.0):
        model.weight.grad = torch.full_like(model.weight, value)
        tracker.update(model)
    assert tracker.last_step == tracker.last_observation_step == 3
    assert tracker.observations == 2
    assert tracker.snr == pytest.approx(
        10 * torch.log10(torch.tensor(6.25 / 0.75)).item()
    )


def test_grad_snr_samples_noncontiguous_main_gradient_in_logical_order():
    model = torch.nn.Linear(3, 2, bias=False)
    model.weight.main_grad = torch.arange(6, dtype=torch.float32).reshape(3, 2).T
    model.weight.grad = torch.full_like(model.weight, 100)
    tracker = GradSNRTracker()
    tracker.update(model)
    torch.testing.assert_close(
        tracker._first, torch.tensor([0.0, 2.0, 4.0, 1.0, 3.0, 5.0])
    )


def test_grad_snr_does_not_report_roundoff_as_variance():
    model = torch.nn.Linear(10, 10, bias=False)
    torch.manual_seed(3407)
    model.weight.grad = torch.randn_like(model.weight)
    tracker = GradSNRTracker()
    for _ in range(20):
        tracker.update(model)
        assert tracker.snr is None
