"""Unit tests for the online rollout module."""

import threading

import torch

from astrai.inference.core.scheduler import Scheduler
from astrai.trainer.backend import ColocatedBackend, P2PCopyPublisher, ReplicaBackend
from astrai.trainer.rollout import (
    BaseRewardModel,
    RolloutGenerator,
    RolloutRunner,
    SamplingParams,
)
from astrai.trainer.strategy import GRPOStrategy
from tests.support.capabilities import skip_lt2_cuda
from tests.support.models import make_frozen, make_model
from tests.support.rollout import FakeExecutor
from tests.support.tokenizers import FakeTokenizer


class ConstantRewardModel(BaseRewardModel):
    """Returns a constant reward for every response."""

    def __init__(self, value: float = 1.0):
        self.value = value

    def score(self, prompts, responses):
        B = len(prompts)
        G = len(responses[0]) if B else 0
        return torch.full((B, G), float(self.value))


class BadShapeRewardModel(BaseRewardModel):
    def score(self, prompts, responses):
        return torch.zeros(len(prompts))


class NonFiniteRewardModel(BaseRewardModel):
    def score(self, prompts, responses):
        B = len(prompts)
        G = len(responses[0]) if B else 0
        return torch.full((B, G), float("nan"))


def _make_scheduler(model, tokenizer, max_batch_size=8, max_len=128):
    return Scheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_seq_len=max_len,
    )


def _make_instruction_batch(n=2):
    """Build a batch of instruction+input prompts as lists of strings."""
    instructions = [f"Tell me about topic {i}" for i in range(n)]
    inputs = [f"context {i}" for i in range(n)]
    return {"instruction": instructions, "input": inputs}


def _blocking_hook(original, started, release):
    """Wrap a hook so it signals ``started`` then blocks until ``release``."""

    def hook(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    return hook


def _assert_interleaved(first, second, *, started, release, finished):
    """Run ``first`` until it blocks, then assert ``second`` cannot finish
    while ``first`` holds the lock; release, join both, and surface errors."""
    errors = []

    def run_safely(fn):
        def run():
            try:
                fn()
            except BaseException as exc:
                errors.append(exc)

        return run

    first_thread = threading.Thread(target=run_safely(first))
    second_thread = threading.Thread(target=run_safely(second))
    first_thread.start()
    assert started.wait(timeout=5)
    second_thread.start()
    assert not finished.wait(timeout=0.1)

    release.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []


def _make_generator(device, **kw):
    model, _ = make_model(device, max_position_embeddings=128)
    tokenizer = FakeTokenizer(with_chat_template=True)
    scheduler = _make_scheduler(
        model,
        tokenizer,
        max_batch_size=kw.get("max_batch_size", 8),
        max_len=kw.get("max_position_embeddings", 128),
    )
    generator = RolloutGenerator(
        backend=ColocatedBackend(scheduler),
        tokenizer=tokenizer,
        params=SamplingParams(
            max_tokens=kw.get("max_tokens", 8),
            group_size=kw.get("group_size", 2),
            temperature=kw.get("temperature", 1.0),
            top_k=kw.get("top_k", 0),
            top_p=kw.get("top_p", 1.0),
        ),
    )
    return generator, model


def _make_replica_pair():
    """A training model on cuda:0 and a diverged replica on cuda:1."""
    train_model, _ = make_model("cuda:0", max_position_embeddings=128)
    replica_model, _ = make_model("cuda:1", max_position_embeddings=128)
    with torch.no_grad():
        for p in replica_model.parameters():
            p.add_(1.0)
    return train_model, replica_model


@skip_lt2_cuda
def test_replica_backend_generation_and_output_device():
    """A replica on cuda:1 generates under its own scheduler and the
    generator lands rollout tensors on the training device."""
    train_model, replica_model = _make_replica_pair()
    replica_model.load_state_dict(train_model.state_dict())
    tokenizer = FakeTokenizer(with_chat_template=True)
    backend = ReplicaBackend(
        model=replica_model,
        tokenizer=tokenizer,
        device="cuda:1",
        max_batch_size=8,
        max_seq_len=128,
    )
    generator = RolloutGenerator(
        backend=backend,
        tokenizer=tokenizer,
        params=SamplingParams(group_size=2, max_tokens=4),
        output_device=torch.device("cuda", 0),
    )

    rollout = generator.generate(_make_instruction_batch(n=2))

    assert rollout.responses.shape[:2] == (2, 2)
    assert rollout.responses.device == torch.device("cuda", 0)
    assert rollout.logprobs_old.device == torch.device("cuda", 0)
    # The replica never toggles the shared training model: it stays eval.
    assert replica_model.training is False


@skip_lt2_cuda
def test_p2p_publisher_copies_weights_and_advances_version():
    train_model, replica_model = _make_replica_pair()
    tokenizer = FakeTokenizer(with_chat_template=True)
    backend = ReplicaBackend(
        model=replica_model,
        tokenizer=tokenizer,
        device="cuda:1",
        max_batch_size=8,
        max_seq_len=128,
    )
    publisher = P2PCopyPublisher(backend)

    with torch.no_grad():
        for p in train_model.parameters():
            p.mul_(2.0).add_(1.0)
    publisher.publish(3, train_model)

    assert backend.policy_version == 3
    for (name, src), (name2, dst) in zip(
        train_model.state_dict().items(), backend.model.state_dict().items()
    ):
        assert name == name2
        assert torch.equal(dst.to(src.device), src)
    # A second publish reuses the cached pairs and keeps versions monotone.
    publisher.publish(4, train_model)
    assert backend.policy_version == 4


@skip_lt2_cuda
def test_online_optimizer_step_syncs_replica_atomically():
    """strategy.optimizer_step advances the replica's weights and version
    inside one commit — the cross-GPU rollout contract."""
    train_model, replica_model = _make_replica_pair()
    tokenizer = FakeTokenizer(with_chat_template=True)
    backend = ReplicaBackend(
        model=replica_model,
        tokenizer=tokenizer,
        device="cuda:1",
        max_batch_size=8,
        max_seq_len=128,
    )
    runner = RolloutRunner(
        generator=RolloutGenerator(
            backend=backend,
            tokenizer=tokenizer,
            params=SamplingParams(group_size=2, max_tokens=4),
            output_device=torch.device("cuda", 0),
        ),
        reward_model=ConstantRewardModel(),
        rollout_interval=2,
    )
    strategy = GRPOStrategy(
        model=train_model,
        device="cuda:0",
        old_model=None,
        ref_model=make_frozen(train_model, "cuda:0"),
        clip_eps=0.2,
        kl_coef=0.01,
        group_size=2,
        model_fn=None,
        executor=FakeExecutor(),
    )
    strategy.set_rollout_runner(runner)
    strategy.set_weight_publishers([P2PCopyPublisher(backend)])

    for p in train_model.parameters():
        p.grad = torch.ones_like(p)
    optimizer = torch.optim.SGD(train_model.parameters(), lr=0.1)
    before = next(train_model.parameters()).detach().clone()
    strategy.optimizer_step(optimizer)

    assert not torch.equal(next(train_model.parameters()), before)
    assert backend.policy_version == 1
    assert runner.policy_version == 1
    for (name, src), (name2, dst) in zip(
        train_model.state_dict().items(), backend.model.state_dict().items()
    ):
        assert name == name2
        assert torch.equal(dst.to(src.device), src)


def _make_runner(device, **kw):
    generator, model = _make_generator(
        device,
        group_size=kw.get("group_size", 2),
        max_tokens=kw.get("max_tokens", 8),
        max_batch_size=kw.get("max_batch_size", 8),
        max_len=kw.get("max_position_embeddings", 128),
    )
    rm = ConstantRewardModel(1.0)
    return (
        RolloutRunner(
            generator=generator,
            reward_model=rm,
            rollout_interval=kw.get("rollout_interval", 2),
            max_policy_lag=kw.get("max_policy_lag"),
        ),
        model,
    )


def _interleave_before_snapshot(runner, callback):
    """Wrap ``with_policy_snapshot`` so ``callback`` runs just before a
    named snapshot callback enters the generator/scheduler locks."""
    original_snapshot = runner.generator.with_policy_snapshot

    def wrapper(inspect):
        if inspect.__name__ == "reuse":
            callback()
        return original_snapshot(inspect)

    runner.generator.with_policy_snapshot = wrapper
