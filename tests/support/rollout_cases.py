"""Unit tests for the online rollout module."""

import threading

import torch

from astrai.inference.core.scheduler import Scheduler
from astrai.trainer.backend import ColocatedBackend
from astrai.trainer.rollout import (
    BaseRewardModel,
    RolloutGenerator,
    RolloutRunner,
    SamplingParams,
)
from tests.support.models import make_model
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
