import dataclasses
import threading
from types import SimpleNamespace

import pytest
import torch

from astrai.trainer.rollout import (
    RolloutRunner,
    RolloutVersionError,
)
from tests.support.rollout_cases import (
    ConstantRewardModel,
    assert_interleaved,
    make_generator,
    make_instruction_batch,
    make_runner,
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


def test_rollout_runner_evaluate_leaves_cache_untouched(cpu_device):
    runner, _ = make_runner(cpu_device, rollout_interval=10)
    batch = make_instruction_batch()
    cached, _ = runner(batch)

    eval_batch = make_instruction_batch(n=1)
    result = runner.evaluate(eval_batch)

    assert result.rewards.shape == result.responses.shape[:2]
    replayed, fresh = runner(batch)
    assert replayed is cached
    assert fresh is False
    assert runner._steps_since_rollout == 0


def test_rollout_runner_tags_generation_version_and_preserves_cached_behavior(
    cpu_device,
):
    runner, _ = make_runner(cpu_device, rollout_interval=100)
    batch = make_instruction_batch(n=1)

    first, first_fresh = runner(batch)
    assert first_fresh is True
    assert first.policy_version == 0

    assert runner.update_weights(1) == 1
    cached, cached_fresh = runner(batch)
    assert cached is first
    assert cached_fresh is False
    assert cached.policy_version == 0

    runner.clear_cache()
    refreshed, refreshed_fresh = runner(batch)
    assert refreshed_fresh is True
    assert refreshed.policy_version == 1


def test_rollout_runner_rejects_future_generation_version(cpu_device):
    runner, _ = make_runner(cpu_device, rollout_interval=2)
    raw = runner.generator.generate(make_instruction_batch(n=1))
    raw.policy_version = runner.policy_version + 1
    runner.generator.generate = lambda _batch: raw

    with pytest.raises(RolloutVersionError, match="future policy version"):
        runner(make_instruction_batch(n=1))


def test_rollout_runner_rejects_result_beyond_max_policy_lag(cpu_device):
    runner, _ = make_runner(cpu_device, rollout_interval=4, max_policy_lag=1)
    batch = make_instruction_batch(n=1)
    result, _ = runner(batch)
    assert result.policy_version == 0

    runner.update_weights(2)
    with pytest.raises(RolloutVersionError, match="exceeds max_policy_lag=1"):
        runner(batch)


def test_rollout_runner_revalidates_version_after_async_scoring(cpu_device):
    runner, _ = make_runner(cpu_device, rollout_interval=4, max_policy_lag=0)
    original_score = runner._score

    def score_while_policy_advances(raw):
        result = original_score(raw)
        runner.update_weights(1)
        return result

    runner._score = score_while_policy_advances

    with pytest.raises(RolloutVersionError, match="exceeds max_policy_lag=0"):
        runner(make_instruction_batch(n=1))
    assert runner._cache is None


def test_rollout_runner_publishes_cache_before_concurrent_policy_update(cpu_device):
    runner, _ = make_runner(cpu_device, rollout_interval=4, max_policy_lag=1)
    final_validation_started = threading.Event()
    allow_final_validation_to_finish = threading.Event()
    update_finished = threading.Event()
    rollout_finished = threading.Event()
    validation_calls = 0
    original_validate = runner._validate_policy_version

    def blocking_validate(result, *, live_version=None):
        nonlocal validation_calls
        validation_calls += 1
        original_validate(result, live_version=live_version)
        if validation_calls == 3:
            final_validation_started.set()
            assert allow_final_validation_to_finish.wait(timeout=5)

    runner._validate_policy_version = blocking_validate

    def produce_rollout():
        runner(make_instruction_batch(n=1))
        rollout_finished.set()

    assert_interleaved(
        produce_rollout,
        lambda: runner.apply_weight_update(1, lambda _version: update_finished.set()),
        started=final_validation_started,
        release=allow_final_validation_to_finish,
        finished=update_finished,
    )

    assert rollout_finished.is_set()
    assert update_finished.is_set()
    assert runner._cache is not None
    assert runner._cache.policy_version == 0
    assert runner.policy_version == 1


def test_rollout_runner_reuse_reads_cache_inside_the_snapshot(cpu_device):
    """The reuse decision must observe the cache under the policy snapshot
    (regression: the cache was read outside the lock, so a concurrent
    commit between the read and the lock silently handed the trainer a
    stale rollout — a lost update)."""
    runner, _ = make_runner(cpu_device, rollout_interval=100)
    batch = make_instruction_batch(n=1)
    first, _ = runner(batch)
    assert first.policy_version == 0

    def concurrent_refresh():
        runner.update_weights(1)
        runner._cache = dataclasses.replace(first, policy_version=1)
        runner._steps_since_rollout = 0

    _interleave_before_snapshot(runner, concurrent_refresh)
    result, fresh = runner(batch)
    assert fresh is False
    assert result is not first
    assert result.policy_version == 1


def test_rollout_runner_recovers_when_cache_cleared_before_reuse_snapshot(cpu_device):
    """A cache clear between the reuse decision and the snapshot must
    trigger a fresh rollout instead of an assertion failure (regression:
    ``assert cached is not None`` fired because the object was captured
    outside the lock)."""
    runner, _ = make_runner(cpu_device, rollout_interval=100)
    batch = make_instruction_batch(n=1)
    first, _ = runner(batch)

    _interleave_before_snapshot(runner, runner.clear_cache)
    result, fresh = runner(batch)
    assert fresh is True
    assert result is not first


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rollout_interval": 0}, "rollout_interval must be positive"),
        ({"max_policy_lag": -1}, "max_policy_lag must be non-negative"),
    ],
)
def test_rollout_runner_rejects_invalid_version_window(cpu_device, kwargs, message):
    generator, _ = make_generator(cpu_device)
    with pytest.raises(ValueError, match=message):
        RolloutRunner(generator, ConstantRewardModel(), **kwargs)


def test_rollout_runner_refreshes_for_different_batch(cpu_device):
    runner, _ = make_runner(cpu_device, rollout_interval=100)
    r1, fresh1 = runner(make_instruction_batch(n=1))
    batch2 = {"instruction": ["Different prompt"], "input": [""]}
    r2, fresh2 = runner(batch2)
    assert fresh1 is True
    assert fresh2 is True
    assert r2 is not r1


@pytest.mark.parametrize(
    "reward_model",
    [
        pytest.param(
            SimpleNamespace(score=lambda prompts, responses: torch.zeros(len(prompts))),
            id="bad-shape",
        ),
        pytest.param(ConstantRewardModel(float("nan")), id="nonfinite"),
    ],
)
def test_rollout_runner_rejects_invalid_rewards(cpu_device, reward_model):
    generator, _ = make_generator(cpu_device, group_size=2, max_tokens=2)
    runner = RolloutRunner(generator, reward_model, rollout_interval=1)
    with pytest.raises(ValueError):
        runner(make_instruction_batch(n=1))


def test_rollout_runner_step_triggers_new_rollout(cpu_device):
    runner, _ = make_runner(cpu_device, rollout_interval=2)
    batch = make_instruction_batch()
    r1, fresh1 = runner(batch)
    assert fresh1 is True
    runner.step()
    # interval=2 means trigger when _steps_since_rollout >= 2; 1 step not enough
    r2, fresh2 = runner(batch)
    assert r2 is r1
    assert fresh2 is False
    runner.step()
    # Now _steps_since_rollout == 2 -> re-rollout
    r3, fresh3 = runner(batch)
    assert r3 is not r1
    assert fresh3 is True
    cached, fresh = runner(batch)
    assert cached is r3
    assert fresh is False
