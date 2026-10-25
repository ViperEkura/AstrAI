from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from astrai.inference.core.request import GenerationResult
from astrai.inference.core.versioning import PolicyVersionGuard
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.trainer.backend import ColocatedBackend, P2PCopyPublisher, ReplicaBackend
from tests.support.models import make_rollout_config
from tests.support.tokenizers import FakeTokenizer


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

    expected = [GenerationResult([4, 5, 6], [], "length")]

    def generate(prompts, **kwargs):
        assert state["locked"]
        assert model.training is False
        assert prompts == [[1, 2]]
        assert kwargs == {"max_tokens": 3}
        if fail:
            raise RuntimeError("generation failed")
        return expected

    backend = ColocatedBackend(
        SimpleNamespace(model=model, with_policy_snapshot=snapshot, generate=generate)
    )
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            backend.generate([[1, 2]], max_tokens=3)
    else:
        assert backend.generate([[1, 2]], max_tokens=3) is expected
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


@pytest.mark.parametrize("overlap", [None, False], ids=["default-overlap", "sync"])
def test_replica_can_publish_weights_after_rollout_drains(overlap):
    """Both rollout modes retire generation before accepting fresh weights."""
    source = AutoRegressiveLM(
        make_rollout_config(
            vocab_size=32, max_position_embeddings=16, num_hidden_layers=1
        )
    ).to(device="cpu", dtype=torch.float32)
    target = deepcopy(source).requires_grad_(False)
    tokenizer = FakeTokenizer()
    tokenizer.stop_ids = []
    kwargs = {} if overlap is None else {"enable_overlap": overlap}
    backend = ReplicaBackend(
        model=target,
        tokenizer=tokenizer,
        device="cpu",
        max_batch_size=2,
        max_seq_len=16,
        enable_cuda_graph=False,
        **kwargs,
    )
    core = backend.engine_core
    try:
        assert core.enable_overlap is (True if overlap is None else overlap)
        results = backend.generate(
            [[3, 4], [5, 6, 7], [8, 9]],
            max_tokens=4,
            temperature=0.7,
            top_k=0,
            seeds=[11, 22, 33],
            return_logprobs=True,
        )
        assert all(
            len(result.token_ids) == len(result.logprobs) == 4 for result in results
        )
        assert not core._pending
        assert core.model_runner.peek_pending() is None
        assert not core.scheduler.has_outstanding()
        assert core.get_stats()["kv_cache_tasks"] == 0
        with torch.no_grad():
            source.lm_head.weight.add_(0.1)
        P2PCopyPublisher(backend).publish(1, source)
        assert backend.policy_version == 1
        assert torch.equal(target.lm_head.weight, source.lm_head.weight)
        assert (
            len(backend.generate([[3, 4]], max_tokens=2, temperature=0)[0].token_ids)
            == 2
        )
        assert not core.scheduler.has_requests()
        assert core.get_stats()["kv_cache_tasks"] == 0
    finally:
        core.stop()
