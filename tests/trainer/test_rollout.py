"""Unit tests for the online rollout module.

Covers :class:`RolloutResult` / :class:`RawRollout`, :class:`BaseRewardModel`,
:class:`RolloutGenerator` (KV-cache-backed via :class:`InferenceScheduler.run_batch`)
and :class:`RolloutRunner` including its internal cache and rollout-interval
trigger logic.
"""

import pytest
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.inference.core.scheduler import InferenceScheduler
from astrai.model.transformer import AutoRegressiveLM
from astrai.trainer.rollout import (
    BaseRewardModel,
    RawRollout,
    RolloutGenerator,
    RolloutResult,
    RolloutRunner,
)


class FakeTokenizer:
    """Minimal char-level tokenizer stub for rollout tests.

    Vocab: 0 = pad, 1..255 = byte values.  ``stop_ids = [2]`` (a fake
    EOS) so tests can verify early-stopping behaviour.
    """

    stop_ids = [2]

    def encode(self, texts, out_ids=True, **_):
        if isinstance(texts, str):
            texts = [texts]
        return [[b for b in t.encode("utf-8")] for t in texts]

    def decode(self, ids, skip_special_tokens=True):
        out = bytes(b for b in ids if b > 2 or not skip_special_tokens).decode(
            "utf-8", errors="ignore"
        )
        return out


class ConstantRewardModel(BaseRewardModel):
    """Returns a constant reward for every response."""

    def __init__(self, value: float = 1.0):
        self.value = value

    def score(self, prompts, responses):
        B = len(prompts)
        G = len(responses[0]) if B else 0
        return torch.full((B, G), float(self.value))


def _make_config(vocab_size=200, max_len=128):
    return AutoRegressiveLMConfig(
        vocab_size=vocab_size,
        dim=16,
        n_heads=2,
        n_kv_heads=1,
        dim_ffn=32,
        max_len=max_len,
        n_layers=2,
        norm_eps=1e-5,
    )


def _make_model(device):
    cfg = _make_config()
    m = AutoRegressiveLM(cfg).to(device=device)
    m.eval()
    return m, cfg


def _make_scheduler(model, tokenizer, max_batch_size=8, max_len=128):
    return InferenceScheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_seq_len=max_len,
        max_prompt_len=max_len,
    )


def _make_prompt_batch(batch_size=2, prompt_len=6, device="cpu"):
    ids = torch.randint(3, 200, (batch_size, prompt_len), device=device)
    mask = torch.ones(batch_size, prompt_len, dtype=torch.bool, device=device)
    return {"input_ids": ids, "attention_mask": mask}


def test_raw_rollout_fields():
    r = RawRollout(
        prompts=torch.zeros(2, 4, dtype=torch.long),
        responses=torch.zeros(2, 3, 5, dtype=torch.long),
        response_mask=torch.ones(2, 3, 5, dtype=torch.bool),
        logprobs_old=torch.zeros(2, 3, 5),
    )
    assert r.prompts.shape == (2, 4)
    assert r.responses.shape == (2, 3, 5)
    assert r.prompt_texts == []
    assert r.response_texts == []


def test_rollout_result_inherits_raw_rollout_fields():
    r = RolloutResult(
        prompts=torch.zeros(2, 4, dtype=torch.long),
        responses=torch.zeros(2, 3, 5, dtype=torch.long),
        response_mask=torch.ones(2, 3, 5, dtype=torch.bool),
        logprobs_old=torch.zeros(2, 3, 5),
        rewards=torch.zeros(2, 3),
    )
    assert r.rewards.shape == (2, 3)
    assert r.prompts.shape == (2, 4)
    assert r.responses.shape == (2, 3, 5)
    assert r.prompt_texts == []
    assert r.response_texts == []


def test_base_reward_model_is_abstract():
    with pytest.raises(TypeError):
        BaseRewardModel()


def test_constant_reward_model_shape():
    rm = ConstantRewardModel(0.5)
    out = rm.score(["a", "b"], [["x", "y", "z"], ["p", "q", "r"]])
    assert out.shape == (2, 3)
    assert torch.all(out == 0.5)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_generator(device, **kw):
    model, _ = _make_model(device)
    tokenizer = FakeTokenizer()
    scheduler = _make_scheduler(
        model,
        tokenizer,
        max_batch_size=kw.get("max_batch_size", 8),
        max_len=kw.get("max_len", 128),
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
    batch = _make_prompt_batch(batch_size=2, prompt_len=4, device=device)
    r = gen.generate(batch)
    assert r.prompts.shape == (2, 4)
    assert r.responses.shape == (2, 3, 5)
    assert r.response_mask.shape == (2, 3, 5)
    assert r.logprobs_old.shape == (2, 3, 5)
    assert len(r.prompt_texts) == 2
    assert len(r.response_texts) == 2
    assert len(r.response_texts[0]) == 3


def test_rollout_generator_mask_matches_responses(device):
    """Positions beyond a response's length are pad (mask False)."""
    gen, _ = _make_generator(device, group_size=2, max_tokens=6)
    batch = _make_prompt_batch(batch_size=2, prompt_len=4, device=device)
    r = gen.generate(batch)
    for i in range(2):
        for g in range(2):
            real = r.response_mask[i, g].sum().item()
            # Pad positions should be 0
            assert r.responses[i, g, real:].sum() == 0
            # logprobs after the real tokens are 0 (padding)
            if real < r.logprobs_old.size(-1):
                assert torch.all(r.logprobs_old[i, g, real:] == 0)


def test_rollout_generator_logprobs_are_nonpositive(device):
    """Behaviour-policy logprobs of sampled tokens should be ≤ 0."""
    gen, _ = _make_generator(device, group_size=2, max_tokens=4)
    batch = _make_prompt_batch(batch_size=1, prompt_len=3, device=device)
    r = gen.generate(batch)
    for i in range(1):
        for g in range(2):
            mask = r.response_mask[i, g]
            lp = r.logprobs_old[i, g][mask]
            assert torch.all(lp <= 1e-5)


def _make_runner(device, **kw):
    generator, model = _make_generator(
        device,
        group_size=kw.get("group_size", 2),
        max_tokens=kw.get("max_tokens", 8),
        max_batch_size=kw.get("max_batch_size", 8),
        max_len=kw.get("max_len", 128),
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
    batch = _make_prompt_batch(batch_size=2, prompt_len=4, device=device)
    r, is_fresh = runner(batch)
    assert is_fresh
    assert r.prompts.shape == (2, 4)
    assert r.responses.shape == (2, 3, 5)
    assert r.response_mask.shape == (2, 3, 5)
    assert r.rewards.shape == (2, 3)
    assert r.logprobs_old.shape == (2, 3, 5)
    assert len(r.prompt_texts) == 2
    assert len(r.response_texts) == 2
    assert len(r.response_texts[0]) == 3


def test_rollout_runner_cache_returns_stale_flag(device):
    runner, _ = _make_runner(device, rollout_interval=10)
    batch = _make_prompt_batch(device=device)
    r1, fresh1 = runner(batch)
    r2, fresh2 = runner(batch)
    assert r1 is r2
    assert fresh1 is True
    assert fresh2 is False


def test_rollout_runner_step_triggers_new_rollout(device):
    runner, _ = _make_runner(device, rollout_interval=2)
    batch = _make_prompt_batch(device=device)
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
    batch = _make_prompt_batch(device=device)
    r1, _ = runner(batch)
    runner.clear_cache()
    r2, fresh2 = runner(batch)
    assert r2 is not r1
    assert fresh2 is True


def test_rollout_runner_step_resets_counter(device):
    runner, _ = _make_runner(device, rollout_interval=1)
    batch = _make_prompt_batch(device=device)
    r1, _ = runner(batch)
    runner.step()
    r2, fresh2 = runner(batch)
    assert r2 is not r1
    assert fresh2 is True
    # Counter reset after rollout; second call w/o step should be cached.
    r3, fresh3 = runner(batch)
    assert r3 is r2
    assert fresh3 is False
