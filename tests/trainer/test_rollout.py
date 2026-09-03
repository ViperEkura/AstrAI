"""Unit tests for the online rollout module."""

import pytest
import torch

from astrai.inference.scheduler import InferenceScheduler
from astrai.inference.task import GenerationResult
from astrai.trainer.rollout import (
    BaseRewardModel,
    RawRollout,
    RolloutGenerator,
    RolloutResult,
    RolloutRunner,
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
    rm = ConstantRewardModel(1.0)
    return (
        RolloutRunner(
            generator=generator,
            reward_model=rm,
            rollout_interval=kw.get("rollout_interval", 2),
        ),
        model,
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


def test_rollout_runner_release_clears_cache_and_resumes(device):
    runner, _ = _make_runner(device, rollout_interval=100)
    batch = _make_instruction_batch(n=1)
    first, _ = runner(batch)
    assert runner._cache is first

    assert runner.release() is True
    assert runner._cache is None
    assert runner._cache_key is None
    assert runner.generator.scheduler.runtime_released is True
    with pytest.raises(RuntimeError, match="call resume"):
        runner(batch)

    assert runner.resume() is True
    refreshed, is_fresh = runner(batch)
    assert is_fresh is True
    assert refreshed is not first


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
