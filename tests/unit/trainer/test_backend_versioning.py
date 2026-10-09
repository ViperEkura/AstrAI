from types import SimpleNamespace

import pytest
import torch
from torch import nn

from astrai.inference.core.versioning import PolicyVersionGuard
from astrai.trainer.backend import ColocatedBackend, P2PCopyPublisher


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_colocated_mode_changes_stay_inside_policy_snapshot(training, fail):
    state = {"locked": False}
    calls = []

    class Model:
        def __init__(self):
            self.training = training

        def train(self, mode=True):
            assert state["locked"]
            self.training = mode
            calls.append(mode)

        def eval(self):
            self.train(False)

    model = Model()

    def snapshot(inspect):
        state["locked"] = True
        try:
            return inspect(7)
        finally:
            state["locked"] = False

    def run_batch(prompts, **kwargs):
        assert state["locked"]
        assert model.training is False
        assert prompts == [[1, 2]]
        assert kwargs == {"max_tokens": 3}
        if fail:
            raise RuntimeError("generation failed")
        return [[4, 5, 6]]

    backend = ColocatedBackend(
        SimpleNamespace(model=model, with_policy_snapshot=snapshot, run_batch=run_batch)
    )
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            backend.generate([[1, 2]], max_tokens=3)
    else:
        assert backend.generate([[1, 2]], max_tokens=3) == [[4, 5, 6]]
    assert calls == [False, training]
    assert model.training is training
    assert not state["locked"]


def make_publisher(*, version=0, ready=True):
    source = nn.Linear(2, 2, bias=False)
    target = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        source.weight.fill_(3)
        target.weight.fill_(1)

    def ensure_ready():
        if not ready:
            raise RuntimeError("generation is active")

    guard = PolicyVersionGuard(version, ensure_ready, lambda: None)
    backend = SimpleNamespace(
        model=target, apply_weight_update=guard.apply_weight_update
    )
    return P2PCopyPublisher(backend), guard, source, target


def test_publisher_copies_under_target_guard_and_reuses_pairs():
    publisher, guard, source, target = make_publisher()
    publisher.publish(1, source)
    assert guard.policy_version == 1
    assert torch.equal(target.weight, source.weight)
    pairs = publisher._pairs
    with torch.no_grad():
        source.weight.fill_(5)
    publisher.publish(2, source)
    assert guard.policy_version == 2
    assert publisher._pairs is pairs
    assert torch.equal(target.weight, source.weight)


@pytest.mark.parametrize("version", [1, 2])
def test_publisher_rejects_stale_version_before_mutating_target(version):
    publisher, guard, source, target = make_publisher(version=2)
    before = target.weight.detach().clone()
    with pytest.raises(ValueError):
        publisher.publish(version, source)
    assert guard.policy_version == 2
    assert torch.equal(target.weight, before)
    assert publisher._pairs is None


def test_publisher_checks_target_readiness_before_mutating_weights():
    publisher, guard, source, target = make_publisher(ready=False)
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="generation is active"):
        publisher.publish(1, source)
    assert guard.policy_version == 0
    assert torch.equal(target.weight, before)


def test_publisher_can_run_inside_same_backends_optimizer_commit():
    publisher, guard, source, target = make_publisher()
    guard.apply_weight_update(None, lambda version: publisher.publish(version, source))
    assert guard.policy_version == 1
    assert torch.equal(target.weight, source.weight)
