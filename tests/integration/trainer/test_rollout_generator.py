"""Unit tests for the online rollout module."""

import threading

import pytest
import torch

from astrai.inference.core.request import GenerationResult
from astrai.trainer.rollout import (
    RolloutEvaluator,
    SamplingParams,
)
from tests.support.rollout_cases import (
    ConstantRewardModel,
    _assert_interleaved,
    _blocking_hook,
    _make_generator,
    _make_instruction_batch,
    _make_runner,
)


def test_rollout_generator_shapes(device):
    gen, _ = _make_generator(device, group_size=3, max_tokens=5)
    batch = _make_instruction_batch(n=2)
    r = gen.generate(batch)
    assert r.responses.shape == (2, 3, 5)
    assert r.response_mask.shape == (2, 3, 5)
    assert r.logprobs_old.shape == (2, 3, 5)
    assert r.prompt_mask.shape == r.prompts.shape
    assert len(r.prompt_texts) == 2
    assert len(r.response_texts) == 2
    assert len(r.response_texts[0]) == 3
    assert r.policy_version == 0


def test_rollout_generator_uses_eval_and_restores_mode(device):
    gen, model = _make_generator(device, group_size=1, max_tokens=2)
    model.train()
    seen_training = []
    original = gen.backend.scheduler.run_batch

    def recording_run_batch(*args, **kwargs):
        seen_training.append(model.training)
        return original(*args, **kwargs)

    gen.backend.scheduler.run_batch = recording_run_batch
    gen.generate(_make_instruction_batch(n=1))
    assert seen_training == [False]
    assert model.training is True


def test_generate_params_override_training_defaults(device):
    """A per-call SamplingParams overrides group size and token budget."""
    gen, _ = _make_generator(device, group_size=2, max_tokens=8)
    batch = _make_instruction_batch(n=2)

    default = gen.generate(batch)
    assert default.responses.shape[1] == 2

    override = gen.generate(
        batch, SamplingParams(group_size=1, max_tokens=1, temperature=1.0)
    )
    assert override.responses.shape[:2] == (2, 1)
    assert override.responses.shape[2] <= 1
    # The generator's training defaults are untouched by the override.
    assert gen.params.group_size == 2
    assert gen.generate(batch).responses.shape[1] == 2


def test_rollout_evaluator_reports_reward_metrics_and_leaves_cache(device):
    """The val evaluator scores under its own params; the replay cache,
    its cadence counter, and the cache key stay exactly as they were."""
    runner, _ = _make_runner(device, group_size=2, max_tokens=4, rollout_interval=10)
    batch = _make_instruction_batch(n=2)

    result, is_fresh = runner(batch)
    assert is_fresh
    cache_before = runner._cache
    steps_before = runner._steps_since_rollout

    evaluator = RolloutEvaluator(
        generator=runner.generator,
        reward_model=ConstantRewardModel(2.0),
        params=SamplingParams(group_size=1, max_tokens=2, temperature=0.0),
    )
    metrics = evaluator.evaluate(batch)

    assert metrics["reward_mean"] == pytest.approx(2.0)
    assert metrics["reward_std"] == pytest.approx(0.0)
    assert metrics["num_responses"] == 2.0  # B=2 prompts x G=1 override
    assert metrics["response_len_mean"] <= 2.0
    assert runner._cache is cache_before
    assert runner._cache_key == runner._batch_key(batch)
    assert runner._steps_since_rollout == steps_before


def test_rollout_generator_serializes_generation_and_policy_update(device):
    gen, _ = _make_generator(device, group_size=1, max_tokens=2)
    generation_started = threading.Event()
    allow_generation_to_finish = threading.Event()
    update_finished = threading.Event()
    gen._generate_eval = _blocking_hook(
        gen._generate_eval, generation_started, allow_generation_to_finish
    )

    _assert_interleaved(
        lambda: gen.generate(_make_instruction_batch(n=1)),
        lambda: gen.apply_weight_update(1, lambda _version: update_finished.set()),
        started=generation_started,
        release=allow_generation_to_finish,
        finished=update_finished,
    )

    assert update_finished.is_set()
    assert gen.policy_version == 1


def test_apply_weight_update_hands_derived_version_inside_lock(device):
    gen, _ = _make_generator(device, group_size=1, max_tokens=2)
    seen = {}

    def record(policy_version):
        # The target version is derived under the lock and passed in; the
        # live version has not moved yet because the commit follows the
        # update — weight publishers rely on exactly this ordering.
        seen["arg"] = policy_version
        seen["live_during"] = gen.backend.scheduler.policy_version

    gen.apply_weight_update(None, record)

    assert seen == {"arg": 1, "live_during": 0}
    assert gen.policy_version == 1


def test_rollout_generator_serializes_direct_scheduler_update(device):
    gen, _ = _make_generator(device, group_size=1, max_tokens=2)
    generation_started = threading.Event()
    allow_generation_to_finish = threading.Event()
    update_finished = threading.Event()
    gen._generate_eval = _blocking_hook(
        gen._generate_eval, generation_started, allow_generation_to_finish
    )
    rollout = []

    def update_scheduler_directly():
        gen.backend.scheduler.update_weights(1)
        update_finished.set()

    _assert_interleaved(
        lambda: rollout.append(gen.generate(_make_instruction_batch(n=1))),
        update_scheduler_directly,
        started=generation_started,
        release=allow_generation_to_finish,
        finished=update_finished,
    )

    assert rollout[0].policy_version == 0
    assert gen.policy_version == 1


def test_rollout_generator_keeps_generation_start_version(device):
    gen, _ = _make_generator(device, group_size=1, max_tokens=2)
    original_run_batch = gen.backend.scheduler.run_batch

    def update_after_generation(*args, **kwargs):
        result = original_run_batch(*args, **kwargs)
        gen.backend.scheduler.update_weights(1)
        return result

    gen.backend.scheduler.run_batch = update_after_generation

    rollout = gen.generate(_make_instruction_batch(n=1))

    assert rollout.policy_version == 0
    assert gen.policy_version == 1


def test_rollout_generator_mask_matches_responses(device):
    """Positions beyond a response's length are pad (mask False)."""
    gen, _ = _make_generator(device, group_size=2, max_tokens=6)
    batch = _make_instruction_batch(n=2)
    r = gen.generate(batch)
    for i in range(2):
        for g in range(2):
            real = r.response_mask[i, g].sum().item()
            assert r.responses[i, g, real:].sum() == 0
            if real < r.logprobs_old.size(-1):
                assert torch.all(r.logprobs_old[i, g, real:] == 0)


def test_rollout_generator_logprobs_are_nonpositive(device):
    """Behaviour-policy logprobs of sampled tokens should be <= 0."""
    gen, _ = _make_generator(device, group_size=2, max_tokens=4)
    batch = _make_instruction_batch(n=1)
    r = gen.generate(batch)
    for i in range(1):
        for g in range(2):
            mask = r.response_mask[i, g]
            lp = r.logprobs_old[i, g][mask]
            assert torch.all(lp <= 1e-5)


def test_rollout_generator_rejects_failed_requests(device):
    gen, _ = _make_generator(device, group_size=2, max_tokens=4)

    def failed_run_batch(*_args, **kwargs):
        assert kwargs["return_details"] is True
        return [
            GenerationResult([1], [-0.1], "length"),
            GenerationResult([], [], "rejected", "kv_cache_allocation_failed"),
        ]

    gen.backend.scheduler.run_batch = failed_run_batch

    with pytest.raises(
        RuntimeError,
        match="Rollout generation failed: request 1: kv_cache_allocation_failed",
    ):
        gen.generate(_make_instruction_batch(n=1))


def test_rollout_generator_instruction_role_mapping(device):
    """instruction -> system, input -> user, output -> assistant."""
    gen, _ = _make_generator(device, group_size=1, max_tokens=4)
    batch = {
        "instruction": ["Be helpful"],
        "input": ["What is 2+2?"],
        "output": ["Four"],
    }
    r = gen.generate(batch)
    text = r.prompt_texts[0]
    assert "SYSTEM: Be helpful" in text
    assert "USER: What is 2+2?" in text
    assert "ASSISTANT: Four" in text


def test_rollout_generator_messages_format(device):
    """Rollout also accepts pre-built messages."""
    gen, _ = _make_generator(device, group_size=2, max_tokens=4)
    batch = {
        "messages": [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Goodbye"}],
        ]
    }
    r = gen.generate(batch)
    assert r.responses.shape[0] == 2
    assert len(r.prompt_texts) == 2
    assert "Hello" in r.prompt_texts[0] or "USER" in r.prompt_texts[0]


def test_rollout_generator_bad_batch_raises(device):
    """Batch without messages or instruction raises a clear error."""
    gen, _ = _make_generator(device)
    with pytest.raises(
        ValueError, match="must contain either 'messages' or 'instruction'"
    ):
        gen.generate({"input_ids": torch.zeros(2, 4, dtype=torch.long)})
