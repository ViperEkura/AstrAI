"""Unit tests for the online rollout module."""

import threading

import pytest
import torch

from astrai.inference.scheduler import InferenceScheduler
from astrai.inference.task import GenerationResult
from astrai.trainer.rollout import (
    BaseRewardModel,
    DynamicSamplingBudgetError,
    DynamicSamplingConfig,
    DynamicSamplingGroup,
    DynamicSamplingState,
    RawRollout,
    RolloutGenerator,
    RolloutResult,
    RolloutRunner,
    RolloutVersionError,
)
from tests.helpers import FakeTokenizer, make_model


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


class ScriptedRewardModel(BaseRewardModel):
    """Returns one explicitly shaped reward matrix per scoring call."""

    def __init__(self, outputs):
        self.outputs = [torch.tensor(output, dtype=torch.float32) for output in outputs]

    def score(self, prompts, responses):
        output = self.outputs.pop(0)
        assert output.shape == (len(prompts), len(responses[0]))
        return output


def _make_scheduler(model, tokenizer, max_batch_size=8, max_len=128):
    return InferenceScheduler(
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


def test_raw_rollout_fields():
    r = RawRollout(
        prompts=torch.zeros(2, 4, dtype=torch.long),
        prompt_mask=torch.ones(2, 4, dtype=torch.bool),
        responses=torch.zeros(2, 3, 5, dtype=torch.long),
        response_mask=torch.ones(2, 3, 5, dtype=torch.bool),
        logprobs_old=torch.zeros(2, 3, 5),
    )
    assert r.prompts.shape == (2, 4)
    assert r.responses.shape == (2, 3, 5)
    assert r.policy_version == 0
    assert r.prompt_texts == []
    assert r.response_texts == []


def test_rollout_result_inherits_raw_rollout_fields():
    r = RolloutResult(
        prompts=torch.zeros(2, 4, dtype=torch.long),
        prompt_mask=torch.ones(2, 4, dtype=torch.bool),
        responses=torch.zeros(2, 3, 5, dtype=torch.long),
        response_mask=torch.ones(2, 3, 5, dtype=torch.bool),
        logprobs_old=torch.zeros(2, 3, 5),
        rewards=torch.zeros(2, 3),
    )
    assert r.rewards.shape == (2, 3)
    assert r.prompts.shape == (2, 4)
    assert r.responses.shape == (2, 3, 5)
    assert r.prompt_mask.shape == (2, 4)
    # RolloutResult must carry every RawRollout field.
    raw_fields = {f for f in RawRollout.__dataclass_fields__}
    assert raw_fields.issubset(set(RolloutResult.__dataclass_fields__))


def test_base_reward_model_is_abstract():
    with pytest.raises(TypeError):
        BaseRewardModel()


def test_constant_reward_model_shape():
    rm = ConstantRewardModel(0.5)
    out = rm.score(["a", "b"], [["x", "y", "z"], ["p", "q", "r"]])
    assert out.shape == (2, 3)
    assert torch.all(out == 0.5)


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
        scheduler=scheduler,
        tokenizer=tokenizer,
        max_tokens=kw.get("max_tokens", 8),
        group_size=kw.get("group_size", 2),
        temperature=kw.get("temperature", 1.0),
        top_k=kw.get("top_k", 0),
        top_p=kw.get("top_p", 1.0),
    )
    return generator, model


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
    original = gen.scheduler.run_batch

    def recording_run_batch(*args, **kwargs):
        seen_training.append(model.training)
        return original(*args, **kwargs)

    gen.scheduler.run_batch = recording_run_batch
    gen.generate(_make_instruction_batch(n=1))
    assert seen_training == [False]
    assert model.training is True


def test_rollout_generator_serializes_generation_and_policy_update(device):
    gen, _ = _make_generator(device, group_size=1, max_tokens=2)
    generation_started = threading.Event()
    allow_generation_to_finish = threading.Event()
    update_finished = threading.Event()
    thread_errors = []
    original = gen._generate_eval

    def blocking_generate(batch, generation_version):
        generation_started.set()
        assert allow_generation_to_finish.wait(timeout=5)
        return original(batch, generation_version)

    gen._generate_eval = blocking_generate

    def generate():
        try:
            gen.generate(_make_instruction_batch(n=1))
        except BaseException as exc:
            thread_errors.append(exc)

    def apply_update():
        try:
            gen.apply_weight_update(1, update_finished.set)
        except BaseException as exc:
            thread_errors.append(exc)

    generation_thread = threading.Thread(target=generate)
    update_thread = threading.Thread(target=apply_update)
    generation_thread.start()
    assert generation_started.wait(timeout=5)
    update_thread.start()
    assert not update_finished.wait(timeout=0.1)

    allow_generation_to_finish.set()
    generation_thread.join(timeout=5)
    update_thread.join(timeout=5)
    assert not generation_thread.is_alive()
    assert not update_thread.is_alive()
    assert thread_errors == []
    assert update_finished.is_set()
    assert gen.policy_version == 1


def test_rollout_generator_serializes_direct_scheduler_update(device):
    gen, _ = _make_generator(device, group_size=1, max_tokens=2)
    generation_started = threading.Event()
    allow_generation_to_finish = threading.Event()
    update_finished = threading.Event()
    thread_errors = []
    original = gen._generate_eval

    def blocking_generate(batch, generation_version):
        generation_started.set()
        assert allow_generation_to_finish.wait(timeout=5)
        return original(batch, generation_version)

    gen._generate_eval = blocking_generate
    rollout = []

    def generate():
        try:
            rollout.append(gen.generate(_make_instruction_batch(n=1)))
        except BaseException as exc:
            thread_errors.append(exc)

    def update_scheduler_directly():
        try:
            gen.scheduler.update_weights(1)
            update_finished.set()
        except BaseException as exc:
            thread_errors.append(exc)

    generation_thread = threading.Thread(target=generate)
    update_thread = threading.Thread(target=update_scheduler_directly)
    generation_thread.start()
    assert generation_started.wait(timeout=5)
    update_thread.start()
    assert not update_finished.wait(timeout=0.1)

    allow_generation_to_finish.set()
    generation_thread.join(timeout=5)
    update_thread.join(timeout=5)
    assert not generation_thread.is_alive()
    assert not update_thread.is_alive()
    assert thread_errors == []
    assert rollout[0].policy_version == 0
    assert gen.policy_version == 1


def test_rollout_generator_keeps_generation_start_version(device):
    gen, _ = _make_generator(device, group_size=1, max_tokens=2)
    original_run_batch = gen.scheduler.run_batch

    def update_after_generation(*args, **kwargs):
        result = original_run_batch(*args, **kwargs)
        gen.scheduler.update_weights(1)
        return result

    gen.scheduler.run_batch = update_after_generation

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

    gen.scheduler.run_batch = failed_run_batch

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


def _make_runner(device, **kw):
    generator, model = _make_generator(
        device,
        group_size=kw.get("group_size", 2),
        max_tokens=kw.get("max_tokens", 8),
        max_batch_size=kw.get("max_batch_size", 8),
        max_len=kw.get("max_position_embeddings", 128),
    )
    rm = kw.get("reward_model", ConstantRewardModel(1.0))
    return (
        RolloutRunner(
            generator=generator,
            reward_model=rm,
            rollout_interval=kw.get("rollout_interval", 2),
            max_policy_lag=kw.get("max_policy_lag"),
            dynamic_sampling=kw.get("dynamic_sampling"),
        ),
        model,
    )


def test_dynamic_sampling_group_enforces_state_machine():
    group = DynamicSamplingGroup(prompt_uid="prompt:0", attempt_id=1, generation_seed=7)
    with pytest.raises(RuntimeError, match="pending -> accepted"):
        group.transition(DynamicSamplingState.ACCEPTED)
    group.transition(DynamicSamplingState.GENERATING)
    group.transition(DynamicSamplingState.SCORING)
    group.transition(DynamicSamplingState.ACCEPTED)
    assert group.accepted is True
    assert group.completed_at is not None


def test_dynamic_sampling_refills_only_low_variance_groups(device):
    rewards = ScriptedRewardModel(
        [
            [[0.0, 1.0], [1.0, 1.0]],
            [[0.0, 2.0]],
        ]
    )
    config = DynamicSamplingConfig(enabled=True, base_seed=19)
    runner, _ = _make_runner(
        device,
        group_size=2,
        max_tokens=2,
        reward_model=rewards,
        dynamic_sampling=config,
    )
    batch_sizes = []
    seeds = []
    original_generate = runner.generator.generate

    def record_generate(batch, *, generation_seed=None):
        batch_sizes.append(len(batch["instruction"]))
        seeds.append(generation_seed)
        return original_generate(batch, generation_seed=generation_seed)

    runner.generator.generate = record_generate
    result, is_fresh = runner(_make_instruction_batch(n=2))

    assert is_fresh is True
    assert batch_sizes == [2, 1]
    assert len(set(seeds)) == 2
    assert result.rewards.tolist() == [[0.0, 1.0], [0.0, 2.0]]
    assert [group.refill_round for group in result.sampling_groups] == [0, 1]
    assert {group.behavior_policy_version for group in result.sampling_groups} == {0}
    assert all(
        group.state is DynamicSamplingState.ACCEPTED for group in result.sampling_groups
    )
    assert runner.last_sampling_metrics["groups_accepted"] == 2.0
    assert runner.last_sampling_metrics["zero_variance_groups"] == 1.0
    assert runner.last_sampling_metrics["refill_rounds"] == 1.0
    assert runner.last_sampling_metrics["rollout_waste_ratio"] > 0.0


def test_dynamic_sampling_restarts_whole_batch_after_version_change(device):
    rewards = ScriptedRewardModel(
        [
            [[0.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [0.0, 2.0]],
        ]
    )
    runner, _ = _make_runner(
        device,
        group_size=2,
        max_tokens=2,
        reward_model=rewards,
        dynamic_sampling=DynamicSamplingConfig(enabled=True),
    )
    batch_sizes = []
    original_generate = runner.generator.generate

    def update_before_refill(batch, *, generation_seed=None):
        batch_sizes.append(len(batch["instruction"]))
        if len(batch_sizes) == 2:
            runner.generator.update_weights(1)
        return original_generate(batch, generation_seed=generation_seed)

    runner.generator.generate = update_before_refill
    result, _ = runner(_make_instruction_batch(n=2))

    assert batch_sizes == [2, 1, 2]
    assert result.policy_version == 1
    assert {group.behavior_policy_version for group in result.sampling_groups} == {1}
    invalidated = [
        group
        for group in runner.last_sampling_history
        if group.state is DynamicSamplingState.INVALIDATED
    ]
    assert len(invalidated) == 2
    assert runner.last_sampling_metrics["version_invalidated_groups"] == 2.0


def test_dynamic_sampling_budget_exhaustion_refuses_partial_batch(device):
    runner, _ = _make_runner(
        device,
        group_size=2,
        max_tokens=2,
        dynamic_sampling=DynamicSamplingConfig(
            enabled=True,
            max_refill_rounds=0,
        ),
    )

    with pytest.raises(DynamicSamplingBudgetError, match="partial"):
        runner(_make_instruction_batch(n=2))

    assert runner.last_sampling_metrics["groups_accepted"] == 0.0
    assert runner.last_sampling_metrics["dropped_groups"] == 2.0
    assert runner.last_sampling_metrics["budget_exhausted_groups"] == 2.0
    assert runner._cache is None


def test_dynamic_sampling_pending_group_budget_fails_before_generation(device):
    runner, _ = _make_runner(
        device,
        dynamic_sampling=DynamicSamplingConfig(enabled=True, max_pending_groups=1),
    )
    with pytest.raises(DynamicSamplingBudgetError, match="max_pending_groups=1"):
        runner(_make_instruction_batch(n=2))


def test_dynamic_sampling_generation_budget_is_reserved_before_attempt(device):
    runner, _ = _make_runner(
        device,
        group_size=2,
        max_tokens=4,
        dynamic_sampling=DynamicSamplingConfig(
            enabled=True,
            max_generated_tokens_per_group=7,
        ),
    )
    called = False

    def should_not_generate(*_args, **_kwargs):
        nonlocal called
        called = True

    runner.generator.generate = should_not_generate
    with pytest.raises(DynamicSamplingBudgetError, match="cannot start"):
        runner(_make_instruction_batch(n=1))
    assert called is False
    assert runner.last_sampling_history[0].discard_reason == (
        "max_generated_tokens_per_group"
    )


def test_dynamic_sampling_scoring_failure_drops_attempt(device):
    runner, _ = _make_runner(
        device,
        group_size=2,
        max_tokens=2,
        reward_model=BadShapeRewardModel(),
        dynamic_sampling=DynamicSamplingConfig(enabled=True),
    )
    with pytest.raises(ValueError, match="Reward model returned shape"):
        runner(_make_instruction_batch(n=1))
    assert runner.last_sampling_history[0].state is DynamicSamplingState.DROPPED
    assert runner.last_sampling_history[0].discard_reason == "scoring_failed"


def test_seeded_rollout_generation_restores_torch_rng(device):
    generator, _ = _make_generator(device, group_size=2, max_tokens=2)
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    generator.generate(_make_instruction_batch(n=1), generation_seed=99)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


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
    thread_errors = []
    validation_calls = 0
    original_validate = runner._validate_policy_version

    def blocking_validate(result, *, live_version=None):
        nonlocal validation_calls
        validation_calls += 1
        original_validate(result, live_version=live_version)
        if validation_calls == 2:
            final_validation_started.set()
            assert allow_final_validation_to_finish.wait(timeout=5)

    runner._validate_policy_version = blocking_validate

    def produce_rollout():
        try:
            runner(_make_instruction_batch(n=1))
            rollout_finished.set()
        except BaseException as exc:
            thread_errors.append(exc)

    def apply_update():
        try:
            runner.apply_weight_update(1, update_finished.set)
        except BaseException as exc:
            thread_errors.append(exc)

    rollout_thread = threading.Thread(target=produce_rollout)
    update_thread = threading.Thread(target=apply_update)
    rollout_thread.start()
    assert final_validation_started.wait(timeout=5)
    update_thread.start()
    assert not update_finished.wait(timeout=0.1)

    allow_final_validation_to_finish.set()
    rollout_thread.join(timeout=5)
    update_thread.join(timeout=5)
    assert not rollout_thread.is_alive()
    assert not update_thread.is_alive()
    assert thread_errors == []
    assert rollout_finished.is_set()
    assert update_finished.is_set()
    assert runner._cache is not None
    assert runner._cache.policy_version == 0
    assert runner.policy_version == 1


def test_rollout_runner_derives_default_policy_lag_from_interval(device):
    runner, _ = _make_runner(device, rollout_interval=4)
    assert runner.max_policy_lag == 3


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
