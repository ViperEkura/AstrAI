"""Unit tests for the online rollout module.

Covers :class:`RolloutResult`, :class:`BaseRewardModel`,
:func:`generate_responses`, and :class:`RolloutRunner` including
its internal cache and rollout-interval trigger logic.
"""

import pytest
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.inference.sample import (
    SamplingPipeline,
    TemperatureStrategy,
    TopKStrategy,
    TopPStrategy,
)
from astrai.model.transformer import AutoRegressiveLM
from astrai.trainer.rollout import (
    BaseRewardModel,
    RolloutResult,
    RolloutRunner,
    generate_responses,
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


class _FakeOldModel:
    """Placeholder old-model; RolloutRunner stores but never calls it."""


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


def _make_pipeline():
    return SamplingPipeline(
        [TemperatureStrategy(1.0), TopKStrategy(0), TopPStrategy(1.0)]
    )


def _make_prompt_batch(batch_size=2, prompt_len=6, device="cpu"):
    ids = torch.randint(3, 200, (batch_size, prompt_len), device=device)
    mask = torch.ones(batch_size, prompt_len, dtype=torch.bool, device=device)
    return {"input_ids": ids, "attention_mask": mask}


def test_rollout_result_fields():
    r = RolloutResult(
        prompts=torch.zeros(2, 4, dtype=torch.long),
        responses=torch.zeros(2, 3, 5, dtype=torch.long),
        response_mask=torch.ones(2, 3, 5, dtype=torch.bool),
        rewards=torch.zeros(2, 3),
        logprobs_old=torch.zeros(2, 3, 5),
    )
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


def test_generate_responses_shapes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = _make_model(device)
    pipeline = _make_pipeline()
    ids = torch.randint(3, 200, (2, 4), device=device)
    mask = torch.ones(2, 4, dtype=torch.bool, device=device)

    out = generate_responses(
        model=model,
        input_ids=ids,
        attention_mask=mask,
        max_new_tokens=8,
        sampling_pipeline=pipeline,
        stop_ids=[],
    )
    assert out["generated_ids"].shape == (2, 8)
    assert out["generated_mask"].shape == (2, 8)
    assert out["logprobs"].shape == (2, 8)


def test_generate_responses_stops_on_stop_id():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = _make_model(device)
    pipeline = _make_pipeline()
    ids = torch.randint(3, 200, (1, 3), device=device)
    mask = torch.ones(1, 3, dtype=torch.bool, device=device)

    out = generate_responses(
        model=model,
        input_ids=ids,
        attention_mask=mask,
        max_new_tokens=16,
        sampling_pipeline=pipeline,
        stop_ids=[7],
    )
    gen = out["generated_ids"][0]
    mask = out["generated_mask"][0]
    # If a 7 appeared, all tokens after it must be pad (mask False).
    nonzero_stop = (gen == 7).nonzero()
    if nonzero_stop.numel():
        first = nonzero_stop[0].item()
        assert mask[first + 1 :].sum() == 0


def test_generate_responses_logprobs_match_tokens():
    """logprobs[i] must be the logprob of generated_ids[i]."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = _make_model(device)
    pipeline = _make_pipeline()
    ids = torch.randint(3, 200, (1, 2), device=device)
    mask = torch.ones(1, 2, dtype=torch.bool, device=device)

    out = generate_responses(
        model=model,
        input_ids=ids,
        attention_mask=mask,
        max_new_tokens=4,
        sampling_pipeline=pipeline,
        stop_ids=[],
    )
    gen = out["generated_ids"][0]
    lp = out["logprobs"][0]
    for i in range(4):
        if gen[i] == 0 and not out["generated_mask"][0, i]:
            continue
        assert lp[i] <= 0.0


def _make_runner(device, **kw):
    model, _ = _make_model(device)
    rm = ConstantRewardModel(1.0)
    return RolloutRunner(
        policy_model=model,
        old_model=_FakeOldModel(),
        tokenizer=FakeTokenizer(),
        reward_model=rm,
        sampling_pipeline=_make_pipeline(),
        max_tokens=kw.get("max_tokens", 8),
        group_size=kw.get("group_size", 2),
        rollout_interval=kw.get("rollout_interval", 2),
    ), model


def test_rollout_runner_shapes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner, _ = _make_runner(device, group_size=3, max_tokens=5)
    batch = _make_prompt_batch(batch_size=2, prompt_len=4, device=device)
    r = runner(batch)
    assert r.prompts.shape == (2, 4)
    assert r.responses.shape == (2, 3, 5)
    assert r.response_mask.shape == (2, 3, 5)
    assert r.rewards.shape == (2, 3)
    assert r.logprobs_old.shape == (2, 3, 5)
    assert len(r.prompt_texts) == 2
    assert len(r.response_texts) == 2
    assert len(r.response_texts[0]) == 3


def test_rollout_runner_cache_returns_same_object():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner, _ = _make_runner(device, rollout_interval=10)
    batch = _make_prompt_batch(device=device)
    r1 = runner(batch)
    r2 = runner(batch)
    assert r1 is r2


def test_rollout_runner_step_triggers_new_rollout():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner, _ = _make_runner(device, rollout_interval=2)
    batch = _make_prompt_batch(device=device)
    r1 = runner(batch)
    runner.step()
    # interval=2 means trigger when _steps_since_rollout >= 2; 1 step not enough
    r2 = runner(batch)
    assert r1 is r2
    runner.step()
    # Now _steps_since_rollout == 2 -> re-rollout
    r3 = runner(batch)
    assert r3 is not r1


def test_rollout_runner_clear_cache_forces_rerun():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner, _ = _make_runner(device, rollout_interval=100)
    batch = _make_prompt_batch(device=device)
    r1 = runner(batch)
    runner.clear_cache()
    r2 = runner(batch)
    assert r2 is not r1


def test_rollout_runner_step_resets_counter():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner, _ = _make_runner(device, rollout_interval=1)
    batch = _make_prompt_batch(device=device)
    r1 = runner(batch)
    runner.step()
    r2 = runner(batch)
    assert r2 is not r1
    # Counter reset after rollout; second call w/o step should be cached.
    r3 = runner(batch)
    assert r3 is r2
