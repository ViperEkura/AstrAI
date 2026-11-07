import threading

import pytest
import torch

from astrai.inference.core.request import GenerationResult
from astrai.trainer.backend import run_batch
from astrai.trainer.rollout import (
    RolloutEvaluator,
    SamplingParams,
)
from tests.support.rollout_cases import (
    ConstantRewardModel,
    assert_interleaved,
    make_generator,
    make_instruction_batch,
    make_runner,
)


def _blocking_hook(original, started, release):
    """Wrap a hook so it signals ``started`` then blocks until ``release``."""

    def hook(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    return hook


def test_rollout_generator_uses_eval_and_restores_mode(cpu_device, monkeypatch):
    gen, model = make_generator(cpu_device, group_size=1, max_tokens=2)
    model.train()
    seen_training = []
    original = run_batch

    def recording_run_batch(*args, **kwargs):
        seen_training.append(model.training)
        return original(*args, **kwargs)

    monkeypatch.setattr("astrai.trainer.backend.run_batch", recording_run_batch)
    gen.generate(make_instruction_batch(n=1))
    assert seen_training == [False]
    assert model.training is True


def test_generate_params_override_training_defaults(cpu_device):
    """A per-call SamplingParams overrides group size and token budget."""
    gen, _ = make_generator(cpu_device, group_size=2, max_tokens=8)
    batch = make_instruction_batch(n=2)

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


def test_rollout_evaluator_reports_reward_metrics_and_leaves_cache(cpu_device):
    """The val evaluator scores under its own params; the replay cache,
    its cadence counter, and the cache key stay exactly as they were."""
    runner, _ = make_runner(cpu_device, group_size=2, max_tokens=4, rollout_interval=10)
    batch = make_instruction_batch(n=2)

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


def test_rollout_generator_serializes_generation_and_policy_update(cpu_device):
    gen, _ = make_generator(cpu_device, group_size=1, max_tokens=2)
    generation_started = threading.Event()
    allow_generation_to_finish = threading.Event()
    update_finished = threading.Event()
    gen._generate_eval = _blocking_hook(
        gen._generate_eval, generation_started, allow_generation_to_finish
    )

    assert_interleaved(
        lambda: gen.generate(make_instruction_batch(n=1)),
        lambda: gen.apply_weight_update(1, lambda _version: update_finished.set()),
        started=generation_started,
        release=allow_generation_to_finish,
        finished=update_finished,
    )

    assert update_finished.is_set()
    assert gen.policy_version == 1


def test_apply_weight_update_hands_derived_version_inside_lock(cpu_device):
    gen, _ = make_generator(cpu_device, group_size=1, max_tokens=2)
    seen = {}

    def record(policy_version):
        # The target version is derived under the lock and passed in; the
        # live version has not moved yet because the commit follows the
        # update — weight publishers rely on exactly this ordering.
        seen["arg"] = policy_version
        seen["live_during"] = gen.backend.engine_core.policy_version

    gen.apply_weight_update(None, record)

    assert seen == {"arg": 1, "live_during": 0}
    assert gen.policy_version == 1


def test_rollout_generator_serializes_direct_scheduler_update(cpu_device):
    gen, _ = make_generator(cpu_device, group_size=1, max_tokens=2)
    generation_started = threading.Event()
    allow_generation_to_finish = threading.Event()
    update_finished = threading.Event()
    gen._generate_eval = _blocking_hook(
        gen._generate_eval, generation_started, allow_generation_to_finish
    )
    rollout = []

    def update_scheduler_directly():
        gen.backend.engine_core.update_weights(1)
        update_finished.set()

    assert_interleaved(
        lambda: rollout.append(gen.generate(make_instruction_batch(n=1))),
        update_scheduler_directly,
        started=generation_started,
        release=allow_generation_to_finish,
        finished=update_finished,
    )

    assert rollout[0].policy_version == 0
    assert gen.policy_version == 1


def test_rollout_generator_keeps_generation_start_version(cpu_device, monkeypatch):
    gen, _ = make_generator(cpu_device, group_size=1, max_tokens=2)
    original_run_batch = run_batch

    def update_after_generation(*args, **kwargs):
        result = original_run_batch(*args, **kwargs)
        gen.backend.engine_core.update_weights(1)
        return result

    monkeypatch.setattr("astrai.trainer.backend.run_batch", update_after_generation)

    rollout = gen.generate(make_instruction_batch(n=1))

    assert rollout.policy_version == 0
    assert gen.policy_version == 1


def test_rollout_generator_mask_matches_responses(cpu_device):
    """Positions beyond a response's length are pad (mask False)."""
    gen, _ = make_generator(cpu_device, group_size=2, max_tokens=6)
    batch = make_instruction_batch(n=2)
    r = gen.generate(batch)
    for i in range(2):
        for g in range(2):
            real = r.response_mask[i, g].sum().item()
            assert r.responses[i, g, real:].sum() == 0
            if real < r.logprobs_old.size(-1):
                assert torch.all(r.logprobs_old[i, g, real:] == 0)


def test_rollout_generator_rejects_failed_requests(cpu_device, monkeypatch):
    gen, _ = make_generator(cpu_device, group_size=2, max_tokens=4)

    def failed_run_batch(*_args, **kwargs):
        assert kwargs["return_details"] is True
        return [
            GenerationResult([1], [-0.1], "length"),
            GenerationResult([], [], "rejected", "kv_cache_allocation_failed"),
        ]

    monkeypatch.setattr("astrai.trainer.backend.run_batch", failed_run_batch)

    with pytest.raises(
        RuntimeError,
        match="Rollout generation failed: request 1: kv_cache_allocation_failed",
    ):
        gen.generate(make_instruction_batch(n=1))


def test_rollout_generator_instruction_role_mapping(cpu_device):
    """instruction -> system, input -> user, output -> assistant."""
    gen, _ = make_generator(cpu_device, group_size=1, max_tokens=4)
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


def test_rollout_generator_bad_batch_raises(cpu_device):
    """Batch without messages or instruction raises a clear error."""
    gen, _ = make_generator(cpu_device)
    with pytest.raises(
        ValueError, match="must contain either 'messages' or 'instruction'"
    ):
        gen.generate({"input_ids": torch.zeros(2, 4, dtype=torch.long)})
