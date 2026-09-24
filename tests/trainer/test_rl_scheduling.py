"""RL scheduling (minibatch × update epochs), chunked-grad logprobs,
reference-restore contract, and rollout finish-reason plumbing."""

import os
import random
import shutil
from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import astrai.trainer.strategy as strategy_mod
from astrai.config import TrainConfig
from astrai.serialization import Checkpoint
from astrai.trainer.optional_extras import (
    checkpoint_extras,
    restore_checkpoint_extras,
)
from astrai.trainer.rollout import BaseRewardModel, RolloutResult
from astrai.trainer.schedule import SchedulerFactory
from astrai.trainer.strategy import (
    GRPOStrategy,
    PPOStrategy,
    _truncation_metric,
    get_logprobs,
    move_to_device,
)
from astrai.trainer.trainer import Trainer
from tests.helpers import CHAT_TEMPLATE


class StubLM(nn.Module):
    """Tiny LM stub honoring the get_logprobs contract (skip_lm_head kwarg,
    dict outputs with logits/hidden_states/aux_loss/router_stats)."""

    def __init__(self, vocab: int = 11, hidden: int = 6):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, ids, mask=None, skip_lm_head=False):
        hidden = self.embed(ids)
        logits = None if skip_lm_head else self.lm_head(hidden)
        return {
            "logits": logits,
            "hidden_states": hidden if logits is None else None,
            "aux_loss": None,
            "router_stats": None,
        }


class StubCritic(nn.Module):
    """Value-head stub for PPO: mean embedding as V(s)."""

    def __init__(self, vocab: int = 11, hidden: int = 6):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)

    def forward(self, ids, input_mask=None):
        return {"values": self.embed(ids).mean(-1, keepdim=True)}


class FakeRunner:
    """RolloutRunner stand-in: hands back one fixed RolloutResult."""

    def __init__(self, result: RolloutResult):
        self.result = result
        self.calls = 0
        self.steps = 0

    def __call__(self, batch):
        self.calls += 1
        return self.result, True

    def step(self):
        self.steps += 1

    def apply_weight_update(self, policy_version, update):
        return update(0)

    @property
    def policy_version(self):
        return 0


def _rollout_result(
    b: int = 4,
    g: int = 2,
    r: int = 3,
    p: int = 3,
    vocab: int = 11,
    finish_reasons=None,
) -> RolloutResult:
    torch.manual_seed(0)
    return RolloutResult(
        prompts=torch.randint(1, vocab, (b, p)),
        prompt_mask=torch.ones(b, p, dtype=torch.bool),
        responses=torch.randint(1, vocab, (b, g, r)),
        response_mask=torch.ones(b, g, r, dtype=torch.bool),
        logprobs_old=torch.randn(b, g, r) * 0.1,
        rewards=torch.randn(b, g),
        finish_reasons=finish_reasons or [],
    )


def _grpo_strategy(result, **kwargs):
    torch.manual_seed(1)
    strategy = GRPOStrategy(
        StubLM(),
        "cpu",
        old_model=None,
        ref_model=StubLM(),
        group_size=2,
        **kwargs,
    )
    strategy._rollout_runner = FakeRunner(result)
    return strategy


# --------------- Feature 1: rollout -> minibatch x epochs ---------------


def test_offline_training_steps_yield_once():
    """No runner: the historical one-call-per-batch contract holds."""
    torch.manual_seed(1)
    strategy = GRPOStrategy(StubLM(), "cpu", old_model=None, ref_model=StubLM())
    batch = {
        "prompts": torch.randint(1, 11, (2, 3)),
        "prompt_mask": torch.ones(2, 3, dtype=torch.bool),
        "responses": torch.randint(1, 11, (2, 2, 3)),
        "masks": torch.ones(2, 2, 3, dtype=torch.bool),
        "rewards": torch.randn(2, 2),
        "logprobs_old": torch.randn(2, 2, 3) * 0.1,
    }
    outputs = list(strategy.training_steps(batch))
    assert len(outputs) == 1
    assert torch.isfinite(outputs[0]["loss"])


def test_online_minibatch_and_epoch_step_counts():
    """One rollout round fans out into minibatches × update epochs, with a
    single generation call and one runner step per learner update."""
    result = _rollout_result()
    strategy = _grpo_strategy(result, rl_update_epochs=2, rl_minibatch_prompts=2)
    outputs = list(strategy.training_steps({"instruction": ["x"] * 4}))

    assert len(outputs) == 4  # 2 prompts/minibatch × 2 epochs on B=4
    assert strategy._rollout_runner.calls == 1
    assert all(torch.isfinite(out["loss"]) for out in outputs)
    assert all("truncation_rate" not in out["metrics"] for out in outputs)


def test_minibatch_slices_keep_prompt_groups_intact():
    """Slices are row ranges of the prepared batch: whole groups, shared
    non-tensor payloads, and rewards in the original order."""
    result = _rollout_result(finish_reasons=[["stop", "length"]] * 4)
    strategy = _grpo_strategy(result, rl_minibatch_prompts=2)

    captured = []
    original_prepare = strategy.prepare_from_rollout
    prepared = original_prepare(result)
    strategy.compute_loss_output = lambda batch: (
        captured.append(batch)
        or {
            "loss": torch.tensor(0.5),
            "metrics": {},
        }
    )
    list(strategy.training_steps({"instruction": ["x"] * 4}))

    assert len(captured) == 2
    for i, chunk in enumerate(captured):
        begin = i * 2
        assert torch.equal(chunk["rewards"], prepared["rewards"][begin : begin + 2])
        assert torch.equal(chunk["responses"], prepared["responses"][begin : begin + 2])
        assert chunk["finish_reasons"] == prepared["finish_reasons"]


def test_rollout_truncation_metric_reaches_grpo_metrics():
    result = _rollout_result(finish_reasons=[["stop", "length"]] * 4)
    strategy = _grpo_strategy(result)
    (output,) = list(strategy.training_steps({"instruction": ["x"] * 4}))
    assert output["metrics"]["truncation_rate"] == pytest.approx(0.5)
    assert output["metrics"]["stop_rate"] == pytest.approx(0.5)


def test_ppo_advantages_pinned_once_across_minibatches_and_epochs():
    """GAE is computed once per rollout round on the full batch; every
    minibatch/epoch update consumes the same pinned targets."""
    torch.manual_seed(1)
    strategy = PPOStrategy(
        StubLM(),
        "cpu",
        critic=StubCritic(),
        critic_optimizer=None,
        ref_model=None,
        clip_eps=0.2,
        kl_coef=0.0,
        rl_update_epochs=2,
        rl_minibatch_prompts=2,
    )
    strategy._rollout_runner = FakeRunner(_rollout_result())

    calls = []
    original = strategy._compute_advantages

    def counting(*args, **kwargs):
        calls.append(len(calls))
        return original(*args, **kwargs)

    strategy._compute_advantages = counting
    outputs = list(strategy.training_steps({"instruction": ["x"] * 4}))

    assert len(outputs) == 4
    assert len(calls) == 1


def test_rl_scheduling_fields_validated():
    with pytest.raises(ValueError, match="rl_update_epochs"):
        _grpo_strategy(_rollout_result(), rl_update_epochs=0)
    with pytest.raises(ValueError, match="rl_minibatch_prompts"):
        _grpo_strategy(_rollout_result(), rl_minibatch_prompts=0)
    with pytest.raises(ValueError, match="rl_minibatch_prompts"):
        _grpo_strategy(_rollout_result(), rl_minibatch_prompts=True)


# --------------- Feature 2: gradient-path chunked logprobs ---------------


def test_grad_chunked_logprobs_match_full_path(monkeypatch):
    """Values, hidden grads, and lm_head grads agree with the full-tensor
    path across chunk boundaries (fp32, tolerance = fp32 noise)."""
    monkeypatch.setattr(strategy_mod, "_CHUNK_LOGIT_BYTES", 128)  # ~2 rows/chunk
    torch.manual_seed(3)
    model = StubLM()
    ids = torch.randint(0, 11, (3, 7))
    attn = torch.ones(3, 7, dtype=torch.bool)
    loss_mask = torch.ones(3, 7, dtype=torch.bool)
    loss_mask[:, -2:] = False

    def run(grad_chunked):
        model.zero_grad(set_to_none=True)
        output = get_logprobs(
            model, ids, attn, loss_mask, "none", grad_chunked=grad_chunked
        )
        loss = output["logprobs"].sum()
        loss.backward()
        return (
            output["logprobs"].detach().clone(),
            model.embed.weight.grad.detach().clone(),
            model.lm_head.weight.grad.detach().clone(),
        )

    full = run(False)
    chunked = run(True)
    for ref, got in zip(full, chunked):
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_no_grad_logprobs_unchanged_by_flag():
    """The flag must not disturb the pre-existing no-grad chunked path."""
    torch.manual_seed(3)
    model = StubLM()
    ids = torch.randint(0, 11, (2, 5))
    attn = torch.ones(2, 5, dtype=torch.bool)
    loss_mask = torch.ones(2, 5, dtype=torch.bool)
    with torch.no_grad():
        base = get_logprobs(model, ids, attn, loss_mask, "none")["logprobs"]
        flagged = get_logprobs(model, ids, attn, loss_mask, "none", grad_chunked=True)[
            "logprobs"
        ]
    torch.testing.assert_close(flagged, base)


def test_truncation_metric():
    metric = _truncation_metric([["stop", "length"], ["length"], []])
    assert metric["truncation_rate"].item() == pytest.approx(2 / 3)
    assert metric["stop_rate"].item() == pytest.approx(1 / 3)
    assert _truncation_metric([]) == {}


def test_move_to_device_passes_non_tensors_through():
    tensor = torch.zeros(1)
    batch = move_to_device({"t": tensor, "reasons": [["stop"]]}, "cpu")
    assert batch["t"].device.type == "cpu"
    assert batch["reasons"] == [["stop"]]


# --------------- M0: RNG extras + reference restore contract ---------------


def test_rng_state_round_trip():
    random.seed(1)
    torch.manual_seed(1)
    first = (torch.rand(3), random.random())

    state = checkpoint_extras()["rng_state"]
    diverged = (torch.rand(3), random.random())
    restore_checkpoint_extras({"rng_state": state})
    replayed = (torch.rand(3), random.random())

    # Restore rewinds to the snapshot, so the post-restore draw replays the
    # draw made right after the snapshot — and neither matches the earlier one.
    assert torch.equal(diverged[0], replayed[0])
    assert diverged[1] == replayed[1]
    assert not torch.equal(first[0], replayed[0])
    assert first[1] != replayed[1]


class _RefOnlyStrategy:
    def __init__(self):
        torch.manual_seed(5)
        self.ref_model = StubLM()


def _resume_builder(tmp_path, allow_reanchor=False):
    from astrai.trainer.train_context import TrainContext, TrainContextBuilder

    config = TrainConfig(
        strategy="sft",
        model_fn=lambda: nn.Linear(2, 2),
        dataset=torch.utils.data.TensorDataset(torch.zeros(1, 2)),
        optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.0),
        scheduler_fn=lambda o: None,
        allow_reference_reanchor=allow_reanchor,
    )
    builder = TrainContextBuilder(config)
    builder._resume = True
    context = TrainContext(config=config)
    context.strategy = _RefOnlyStrategy()
    return builder, context


def test_reference_restored_from_checkpoint_extra(tmp_path):
    torch.manual_seed(7)
    saved = {k: v.clone() for k, v in StubLM().state_dict().items()}
    builder, context = _resume_builder(tmp_path)
    context.checkpoint = Checkpoint(extra={"reference_model": saved})

    builder._restore_reference_model(context)

    for key, value in context.strategy.ref_model.state_dict().items():
        torch.testing.assert_close(value, saved[key])


def test_resume_without_reference_extra_rejected(tmp_path):
    builder, context = _resume_builder(tmp_path)
    context.checkpoint = Checkpoint(extra={})

    with pytest.raises(ValueError, match="reference_model"):
        builder._restore_reference_model(context)


def test_reference_reanchor_needs_explicit_opt_in(tmp_path):
    builder, context = _resume_builder(tmp_path, allow_reanchor=True)
    context.checkpoint = Checkpoint(extra={})

    builder._restore_reference_model(context)  # warns, does not raise


def test_fresh_run_keeps_actor_derived_reference(tmp_path):
    builder, context = _resume_builder(tmp_path)
    builder._resume = False
    context.checkpoint = None
    torch.manual_seed(5)
    expected = {
        k: v.clone() for k, v in context.strategy.ref_model.state_dict().items()
    }

    builder._restore_reference_model(context)

    for key, value in context.strategy.ref_model.state_dict().items():
        torch.testing.assert_close(value, expected[key])


# --------------- end-to-end (GPU, real rollout scheduler) ---------------


class InstructionDataset(Dataset):
    _SAMPLES = [
        {"instruction": "Hello", "input": ""},
        {"instruction": "Tell me a story", "input": "about dragons"},
        {"instruction": "Summarize", "input": "the article"},
        {"instruction": "Translate", "input": "to French: hi"},
    ]

    def __len__(self):
        return len(self._SAMPLES)

    def __getitem__(self, idx):
        return dict(self._SAMPLES[idx])


class LengthRewardModel(BaseRewardModel):
    def score(self, prompts, responses):
        b = len(prompts)
        g = len(responses[0]) if b else 0
        rewards = torch.zeros(b, g)
        for i in range(b):
            for j in range(g):
                rewards[i, j] = float(len(responses[i][j]))
        return rewards


def instruction_collate_fn(batch):
    return {
        "instruction": [b["instruction"] for b in batch],
        "input": [b.get("input", "") for b in batch],
    }


def _model_fn(model_config):
    from astrai.model.transformer import AutoRegressiveLM

    return AutoRegressiveLM(model_config).to(dtype=torch.float32)


def _optimizer_fn(m):
    return torch.optim.AdamW(m.parameters(), lr=1e-4)


def _scheduler_fn(optim):
    return SchedulerFactory.create(
        "cosine", optim, warmup_steps=1, lr_decay_steps=4, min_rate=0.05
    )


def _online_config(base_test_env, **overrides):
    model_config = base_test_env["transformer_config"]
    defaults = dict(
        strategy="online_grpo",
        model_fn=partial(_model_fn, model_config),
        dataset=InstructionDataset(),
        optimizer_fn=_optimizer_fn,
        scheduler_fn=_scheduler_fn,
        ckpt_dir=os.path.join(base_test_env["test_dir"], "ckpt"),
        n_epoch=1,
        batch_per_device=2,
        ckpt_interval=100,
        grad_accum_steps=1,
        random_seed=42,
        device_type=base_test_env["device"],
        dp_mode="none",
        strategy_kwargs={"clip_eps": 0.2, "kl_coef": 0.01, "group_size": 2},
        rollout_interval=1,
        rollout_max_policy_lag=0,
        rollout_temperature=1.0,
        rollout_top_k=0,
        rollout_top_p=1.0,
        rollout_max_tokens=4,
        reward_model_fn=LengthRewardModel,
        collate_fn=instruction_collate_fn,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


@pytest.mark.integration
def test_online_minibatch_round_publication_and_reference_resume(base_test_env):
    """Full loop: minibatch×epoch scheduling publishes one policy version per
    learner update, checkpoints carry the frozen reference + RNG extras, and
    resume restores the anchor — refusing when it is missing."""
    test_dir = base_test_env["test_dir"]
    tokenizer = base_test_env["tokenizer"]
    tokenizer.set_chat_template(CHAT_TEMPLATE)
    tokenizer.save_pretrained(test_dir)

    config = _online_config(base_test_env, rl_update_epochs=2, rl_minibatch_prompts=1)
    Trainer(config).train(param_path=test_dir)

    checkpoint_dir = os.path.join(test_dir, "ckpt", "epoch_0_step_2")
    checkpoint = Checkpoint.load(checkpoint_dir)
    # 2 batches × (2 minibatches × 2 epochs) publications.
    assert checkpoint.meta["policy_version"] == 8
    assert "reference_model" in checkpoint.extra
    assert "rng_state" in checkpoint.extra
    ref_keys = set(checkpoint.extra["reference_model"])
    assert ref_keys and ref_keys.issubset(checkpoint.state_dict.keys())

    # Happy resume: the saved anchor wins over the rebuilt-from-actor one.
    Trainer(config).train(param_path=checkpoint_dir, resume=True)

    # Resume from a checkpoint stripped of the anchor must fail loudly.
    stripped_dir = os.path.join(test_dir, "ckpt_stripped")
    stripped = Checkpoint.load(checkpoint_dir)
    stripped.extra.pop("reference_model")
    stripped.save(stripped_dir)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = os.path.join(checkpoint_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(stripped_dir, name))
    with pytest.raises(ValueError, match="reference_model"):
        Trainer(config).train(param_path=stripped_dir, resume=True)

    # The explicit escape hatch proceeds (with a warning).
    reanchor_config = _online_config(
        base_test_env,
        rl_update_epochs=2,
        rl_minibatch_prompts=1,
        allow_reference_reanchor=True,
    )
    Trainer(reanchor_config).train(param_path=stripped_dir, resume=True)
