"""Unit tests for the online rollout module."""

import threading

import torch

from astrai.inference.core.engine_core import EngineCore
from astrai.trainer.backend import ColocatedBackend
from astrai.trainer.rollout import (
    BaseRewardModel,
    RolloutGenerator,
    RolloutRunner,
    SamplingParams,
)
from tests.support.models import make_model
from tests.support.tokenizers import FakeTokenizer


def make_instruction_batch(n=2):
    """Build a batch of instruction+input prompts as lists of strings."""
    instructions = [f"Tell me about topic {i}" for i in range(n)]
    inputs = [f"context {i}" for i in range(n)]
    return {"instruction": instructions, "input": inputs}


def blocking_hook(original, started, release):
    """Wrap a hook so it signals ``started`` then blocks until ``release``."""

    def hook(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    return hook


def assert_interleaved(first, second, *, started, release, finished):
    """Check lock exclusion and release workers even if an assertion fails."""
    errors = []

    def run(fn):
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(fn,)) for fn in (first, second)]
    threads[0].start()
    try:
        assert started.wait(timeout=5)
        threads[1].start()
        assert not finished.wait(timeout=0.1)
    finally:
        release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    if errors:
        raise errors[0]


def make_generator(
    device, *, max_seq_len=128, max_batch_size=8, max_tokens=8, group_size=2
):
    model, _ = make_model(device, max_position_embeddings=max_seq_len)
    tokenizer = FakeTokenizer(with_chat_template=True)
    engine = EngineCore(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    return (
        RolloutGenerator(
            backend=ColocatedBackend(engine),
            tokenizer=tokenizer,
            params=SamplingParams(
                max_tokens=max_tokens, group_size=group_size, temperature=1.0, top_p=1.0
            ),
        ),
        model,
    )


def make_replica_pair():
    """A training model on cuda:0 and a diverged replica on cuda:1."""
    train_model, _ = make_model("cuda:0", max_position_embeddings=128)
    replica_model, _ = make_model("cuda:1", max_position_embeddings=128)
    with torch.no_grad():
        for p in replica_model.parameters():
            p.add_(1.0)
    return train_model, replica_model


def make_runner(device, *, rollout_interval=2, max_policy_lag=None, **generation):
    generator, model = make_generator(device, **generation)
    return (
        RolloutRunner(
            generator=generator,
            reward_model=ConstantRewardModel(),
            rollout_interval=rollout_interval,
            max_policy_lag=max_policy_lag,
        ),
        model,
    )


def interleave_before_snapshot(runner, callback):
    """Wrap ``with_policy_snapshot`` so ``callback`` runs just before a
    named snapshot callback enters the generator/scheduler locks."""
    original_snapshot = runner.generator.with_policy_snapshot

    def wrapper(inspect):
        if inspect.__name__ == "reuse":
            callback()
        return original_snapshot(inspect)

    runner.generator.with_policy_snapshot = wrapper


class ConstantRewardModel(BaseRewardModel):
    """Returns a constant reward for every response."""

    def __init__(self, value: float = 1.0):
        self.value = value

    def score(self, prompts, responses):
        B = len(prompts)
        G = len(responses[0]) if B else 0
        return torch.full((B, G), float(self.value))
