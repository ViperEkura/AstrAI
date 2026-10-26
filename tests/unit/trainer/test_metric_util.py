import json

import pytest
import torch

from astrai.trainer.callbacks.gradient_noise import (
    GradientNoiseProbe,
    GradientNoiseTracker,
    squared_gradient_norm,
)
from astrai.trainer.callbacks.metrics import MetricCallback


def test_weighted_noise_estimate_and_warmup():
    tracker = GradientNoiseTracker(interval=1, beta=0.5)
    # Independent local means [1,2] (one target) and [3,4] (three targets).
    # Global mean [2.5,3.5], weighted local norm 80, covariance 6, signal 17.
    for step in (1, 2):
        tracker.stage(
            step, weighted_local_norm=80, global_norm=18.5, batch_size=4, ranks=2
        )
        assert tracker.samples == step - 1
        values = tracker.commit(step)
        assert values["grad_noise_signal"] == pytest.approx(17)
        assert values["grad_noise_covariance"] == pytest.approx(6)
        if step == 1:
            assert "grad_noise_scale" not in values
        else:
            assert values["grad_noise_scale"] == pytest.approx(6 / 17)


def test_negative_signal_is_averaged_before_ratio_and_state_roundtrip():
    tracker = GradientNoiseTracker(interval=1, beta=0.5)
    tracker.stage(1, 4, 0, 2, 2)
    assert "grad_noise_scale" not in tracker.commit(1)
    restored = GradientNoiseTracker(interval=1, beta=0.5)
    restored.load_state_dict(json.loads(json.dumps(tracker.state_dict())), 1)
    for item in (tracker, restored):
        item.stage(2, 20, 9, 2, 2)
    assert tracker.commit(2) == restored.commit(2)
    assert tracker.state_dict() == restored.state_dict()
    assert tracker.signal == pytest.approx(3)


def test_pending_probe_is_not_checkpoint_state():
    callback = MetricCallback(metrics=["grad_noise_scale"], grad_noise_interval=1)
    before = callback.state_dict()
    callback.noise.stage(1, 80, 18.5, 4, 2)
    assert callback.state_dict() == before
    callback.on_shutdown(None)
    assert callback.noise.pending is None
    assert callback.state_dict() == before


@pytest.mark.parametrize("ranks", [0, 1])
def test_empty_or_single_active_rank_does_not_claim_noise(ranks):
    tracker = GradientNoiseTracker(interval=1)
    tracker.stage(1, 1, 1, 2, ranks)
    assert tracker.commit(1) == {}
    assert tracker.samples == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 9),
        ("signal", float("nan")),
        ("covariance", -1),
        ("last_probe_step", 3),
    ],
)
def test_invalid_estimator_state_is_refused(field, value):
    tracker = GradientNoiseTracker(interval=1)
    state = tracker.state_dict()
    state[field] = value
    with pytest.raises(ValueError):
        tracker.load_state_dict(state, committed_step=2)


def test_bf16_gradient_is_converted_before_squaring():
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    parameter.grad = torch.tensor([1.1, 1.9], dtype=torch.bfloat16)
    expected = parameter.grad.float().square().double().sum()
    assert squared_gradient_norm(parameter) == expected
    assert squared_gradient_norm(parameter) != parameter.grad.square().double().sum()


def test_probe_keeps_accumulated_gradient_for_unused_final_parameter():
    model = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False) for _ in range(2)])
    model[0](torch.tensor([[2.0]])).sum().backward()
    probe = GradientNoiseProbe()
    probe.start(model)
    model[1](torch.tensor([[3.0]])).sum().backward()
    assert {index: float(value) for index, value in probe.norms.items()} == {
        0: 4.0,
        1: 9.0,
    }
    assert model[0].weight.grad.item() == 2
    assert model[1].weight.grad.item() == 3
    probe.close()
    assert not probe.norms and not probe.handles


def test_metric_state_refuses_a_different_checkpoint_or_sampling_config():
    callback = MetricCallback(metrics=["grad_noise_scale"], grad_noise_interval=2)
    state = callback.state_dict()
    with pytest.raises(ValueError, match="checkpoint"):
        callback.load_state_dict(state, 1)
    changed = MetricCallback(metrics=["grad_noise_scale"], grad_noise_interval=3)
    with pytest.raises(ValueError, match="sampling settings"):
        changed.load_state_dict(state, 0)


def test_metric_state_cannot_publish_an_uncommitted_step(tmp_path):
    callback = MetricCallback(metrics=["loss"])
    with pytest.raises(ValueError, match="committed checkpoint step"):
        callback.write_state(tmp_path / "metric_state.json", 1)
    assert not (tmp_path / "metric_state.json").exists()
