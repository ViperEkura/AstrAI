"""Unit tests for the online rollout module."""

import dataclasses
import threading

import pytest

from astrai.trainer.rollout import (
    RolloutRunner,
    RolloutVersionError,
)
from tests.support.rollout_cases import (
    BadShapeRewardModel,
    ConstantRewardModel,
    NonFiniteRewardModel,
    _assert_interleaved,
    _interleave_before_snapshot,
    _make_generator,
    _make_instruction_batch,
    _make_runner,
)


def test_rollout_runner_shapes(device):
    runner, _ = _make_runner(device, group_size=3, max_tokens=5)
    batch = _make_instruction_batch(n=2)
    r, is_fresh = runner(batch)
    assert is_fresh
    assert r.responses.shape == (2, 3, 5)
    assert r.response_mask.shape == (2, 3, 5)
    assert r.rewards.shape == (2, 3)
    assert r.logprobs_old.shape == (2, 3, 5)
    assert len(r.prompt_texts) == 2
    assert len(r.response_texts) == 2
    assert len(r.response_texts[0]) == 3


def test_rollout_runner_cache_returns_stale_flag(device):
    runner, _ = _make_runner(device, rollout_interval=10)
    batch = _make_instruction_batch()
    r1, fresh1 = runner(batch)
    r2, fresh2 = runner(batch)
    assert r1 is r2
    assert fresh1 is True
    assert fresh2 is False


def test_rollout_runner_evaluate_leaves_cache_untouched(device):
    runner, _ = _make_runner(device, rollout_interval=10)
    batch = _make_instruction_batch()
    cached, _ = runner(batch)

    eval_batch = _make_instruction_batch(n=1)
    result = runner.evaluate(eval_batch)

    assert result.rewards.shape == result.responses.shape[:2]
    replayed, fresh = runner(batch)
    assert replayed is cached
    assert fresh is False
    assert runner._steps_since_rollout == 0


def test_rollout_runner_tags_generation_version_and_preserves_cached_behavior(device):
    runner, _ = _make_runner(device, rollout_interval=100)
    batch = _make_instruction_batch(n=1)

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


def test_rollout_runner_rejects_future_generation_version(device):
    runner, _ = _make_runner(device, rollout_interval=2)
    raw = runner.generator.generate(_make_instruction_batch(n=1))
    raw.policy_version = runner.policy_version + 1
    runner.generator.generate = lambda _batch: raw

    with pytest.raises(RolloutVersionError, match="future policy version"):
        runner(_make_instruction_batch(n=1))


def test_rollout_runner_rejects_result_beyond_max_policy_lag(device):
    runner, _ = _make_runner(device, rollout_interval=4, max_policy_lag=1)
    batch = _make_instruction_batch(n=1)
    result, _ = runner(batch)
    assert result.policy_version == 0

    runner.update_weights(2)
    with pytest.raises(RolloutVersionError, match="exceeds max_policy_lag=1"):
        runner(batch)


def test_rollout_runner_revalidates_version_after_async_scoring(device):
    runner, _ = _make_runner(device, rollout_interval=4, max_policy_lag=0)
    original_score = runner._score

    def score_while_policy_advances(raw):
        result = original_score(raw)
        runner.update_weights(1)
        return result

    runner._score = score_while_policy_advances

    with pytest.raises(RolloutVersionError, match="exceeds max_policy_lag=0"):
        runner(_make_instruction_batch(n=1))
    assert runner._cache is None


def test_rollout_runner_publishes_cache_before_concurrent_policy_update(device):
    runner, _ = _make_runner(device, rollout_interval=4, max_policy_lag=1)
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
        runner(_make_instruction_batch(n=1))
        rollout_finished.set()

    _assert_interleaved(
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


def test_rollout_runner_derives_default_policy_lag_from_interval(device):
    runner, _ = _make_runner(device, rollout_interval=4)
    assert runner.max_policy_lag == 3


def test_rollout_runner_reuse_reads_cache_inside_the_snapshot(device):
    """The reuse decision must observe the cache under the policy snapshot
    (regression: the cache was read outside the lock, so a concurrent
    commit between the read and the lock silently handed the trainer a
    stale rollout — a lost update)."""
    runner, _ = _make_runner(device, rollout_interval=100)
    batch = _make_instruction_batch(n=1)
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


def test_rollout_runner_recovers_when_cache_cleared_before_reuse_snapshot(device):
    """A cache clear between the reuse decision and the snapshot must
    trigger a fresh rollout instead of an assertion failure (regression:
    ``assert cached is not None`` fired because the object was captured
    outside the lock)."""
    runner, _ = _make_runner(device, rollout_interval=100)
    batch = _make_instruction_batch(n=1)
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
def test_rollout_runner_rejects_invalid_version_window(device, kwargs, message):
    generator, _ = _make_generator(device)
    with pytest.raises(ValueError, match=message):
        RolloutRunner(generator, ConstantRewardModel(), **kwargs)


def test_rollout_runner_refreshes_for_different_batch(device):
    runner, _ = _make_runner(device, rollout_interval=100)
    r1, fresh1 = runner(_make_instruction_batch(n=1))
    batch2 = {"instruction": ["Different prompt"], "input": [""]}
    r2, fresh2 = runner(batch2)
    assert fresh1 is True
    assert fresh2 is True
    assert r2 is not r1


@pytest.mark.parametrize("reward_model", [BadShapeRewardModel, NonFiniteRewardModel])
def test_rollout_runner_rejects_invalid_rewards(device, reward_model):
    generator, _ = _make_generator(device, group_size=2, max_tokens=2)
    runner = RolloutRunner(generator, reward_model(), rollout_interval=1)
    with pytest.raises(ValueError):
        runner(_make_instruction_batch(n=1))


def test_rollout_runner_step_triggers_new_rollout(device):
    runner, _ = _make_runner(device, rollout_interval=2)
    batch = _make_instruction_batch()
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


def test_rollout_runner_clear_cache_forces_rerun(device):
    runner, _ = _make_runner(device, rollout_interval=100)
    batch = _make_instruction_batch()
    r1, _ = runner(batch)
    runner.clear_cache()
    r2, fresh2 = runner(batch)
    assert r2 is not r1
    assert fresh2 is True


def test_rollout_runner_step_resets_counter(device):
    runner, _ = _make_runner(device, rollout_interval=1)
    batch = _make_instruction_batch()
    r1, _ = runner(batch)
    runner.step()
    r2, fresh2 = runner(batch)
    assert r2 is not r1
    assert fresh2 is True
    # Counter reset after rollout; second call w/o step should be cached.
    r3, fresh3 = runner(batch)
    assert r3 is r2
    assert fresh3 is False
