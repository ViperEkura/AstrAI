"""RL scheduling (minibatch × update epochs), chunked-grad logprobs,
reference-restore contract, and rollout finish-reason plumbing."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from astrai.config import ComponentSpec, TrainConfig
from astrai.dataset import GlobalBatchSampler
from astrai.parallel.executor import BaseExecutor
from astrai.serialization import Checkpoint
from astrai.trainer.context import TrainContext, TrainContextBuilder
from astrai.trainer.rollout import RolloutResult, RolloutRunner
from astrai.trainer.strategy import (
    GRPOStrategy,
    PPOStrategy,
    _truncation_metric,
    get_logprobs,
)
from astrai.trainer.trainer import Trainer


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
        self.policy_version = 0

    def __call__(self, batch):
        self.calls += 1
        return self.result, True

    def step(self):
        self.steps += 1

    def publish_weights(self, version, model):
        pass

    def apply_weight_update(self, policy_version, update):
        result = update(self.policy_version + 1)
        self.policy_version += 1
        return result


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
    return strategy


def _run_replay(strategy, result, epochs=1, minibatch=None):
    """Exercise Trainer's real global-window scheduling with a fixed rollout."""
    total = result.prompts.shape[0]
    config = SimpleNamespace(
        rollout_mode="sync",
        batch_per_device=total,
        grad_accum_steps=1,
        rl_update_epochs=epochs,
        rl_minibatch_prompts=minibatch,
        epoch_tail_policy="flush",
        cp_size=1,
    )
    dataset = list(range(total))
    runner = FakeRunner(result)
    context = TrainContext(
        model=strategy.model,
        strategy=strategy,
        config=config,
        dataset=dataset,
        dataloader=DataLoader(
            dataset,
            batch_sampler=GlobalBatchSampler(dataset, 1, 0, total, 1, 42, "flush"),
        ),
        executor=BaseExecutor(),
        rollout=runner,
        optimizer=torch.optim.SGD(strategy.model.parameters(), lr=0.01),
    )
    context.scheduler = torch.optim.lr_scheduler.StepLR(context.optimizer, 1, gamma=0.9)
    trainer = Trainer.__new__(Trainer)
    trainer.callbacks = []
    trainer._train_global_epoch(context, 0)
    return context


@pytest.mark.parametrize("aggregation", ["token", "sequence"])
def test_async_grpo_microbatches_share_one_update_and_full_batch_gradient(
    aggregation,
):
    result = _rollout_result(b=3)
    result.response_mask[0, :, -1] = False
    result.response_mask[1, :, :] = False
    full = _grpo_strategy(result, loss_aggregation=aggregation)
    micro = _grpo_strategy(result, loss_aggregation=aggregation)

    full_loss = full.compute_loss_output(full.prepare_from_rollout(result))["loss"]
    full_loss.backward()
    outputs = list(
        micro.loss_outputs(micro.prepare_from_rollout(result), microbatch_prompts=1)
    )
    assert len(outputs) == 2
    micro_loss = sum(output["loss"] for output in outputs)
    micro_loss.backward()

    torch.testing.assert_close(micro_loss, full_loss)
    for full_param, micro_param in zip(
        full.model.parameters(), micro.model.parameters()
    ):
        torch.testing.assert_close(micro_param.grad, full_param.grad)


def test_ppo_replay_updates_actor_and_critic_once_per_minibatch():
    result = _rollout_result()
    critic = StubCritic()
    strategy = PPOStrategy(
        StubLM(),
        "cpu",
        critic=critic,
        critic_optimizer=torch.optim.SGD(critic.parameters(), lr=0.01),
        ref_model=None,
    )
    updates = []
    strategy.critic_optimizer.register_step_post_hook(lambda *args: updates.append(1))
    context = _run_replay(strategy, result, epochs=2, minibatch=2)
    assert len(updates) == context.optimizer_step == context.rollout.policy_version == 4
    assert context.rollout.calls == 1
    assert context.scheduler.last_epoch == 4
    assert context.consumed_samples == context.epoch_sample_offset == 4
    assert all(p.grad is None for p in critic.parameters())


def test_shared_train_batch_commits_async_round_once():
    result = _rollout_result(b=3)
    strategy = _grpo_strategy(result)
    coordinator = FakeRunner(result)
    optimizer = torch.optim.SGD(strategy.model.parameters(), lr=0.01)
    before = strategy.model.embed.weight.detach().clone()
    context = TrainContext(
        model=strategy.model,
        strategy=strategy,
        executor=BaseExecutor(),
        config=SimpleNamespace(async_train_microbatch_prompts=1),
        rollout=RolloutRunner(coordinator=coordinator),
        optimizer=optimizer,
        scheduler=None,
        consumed_samples=0,
        epoch_sample_offset=0,
        optimizer_steps_completed=0,
        loss=0.0,
        metrics={},
    )
    trainer = Trainer.__new__(Trainer)
    trainer.callbacks = []
    trainer._train_batch(context, result)

    assert coordinator.policy_version == 1
    assert context.optimizer_steps_completed == 1
    assert context.consumed_samples == 3
    assert not torch.equal(before, strategy.model.embed.weight)


def test_online_minibatch_and_epoch_step_counts():
    """One rollout round fans out into minibatches × update epochs, with a
    single generation call and one runner step per learner update."""
    result = _rollout_result()
    strategy = _grpo_strategy(result)
    context = _run_replay(strategy, result, epochs=2, minibatch=2)
    assert context.optimizer_step == context.rollout.steps == 4
    assert context.rollout.calls == 1
    assert context.consumed_samples == 4
    assert context.scheduler.last_epoch == 4
    assert context.checkpoint_safe


def test_minibatch_slices_keep_prompt_groups_intact():
    """Slices are row ranges of the prepared batch: whole groups, shared
    non-tensor payloads, and rewards in the original order."""
    result = _rollout_result(finish_reasons=[["stop", "length"]] * 4)
    strategy = _grpo_strategy(result)

    captured = []
    original_prepare = strategy.prepare_from_rollout
    prepared = original_prepare(result)
    strategy.compute_loss_output = lambda batch: (
        captured.append(batch)
        or {
            "loss": sum(p.sum() for p in strategy.model.parameters()) * 0 + 0.5,
            "metrics": {},
        }
    )
    _run_replay(strategy, result, minibatch=2)

    assert len(captured) == 2
    for i, chunk in enumerate(captured):
        begin = i * 2
        assert torch.equal(chunk["rewards"], prepared["rewards"][begin : begin + 2])
        assert torch.equal(chunk["responses"], prepared["responses"][begin : begin + 2])
        assert chunk["finish_reasons"] == prepared["finish_reasons"][begin : begin + 2]


def test_rollout_truncation_metric_reaches_grpo_metrics():
    result = _rollout_result(finish_reasons=[["stop", "length"]] * 4)
    strategy = _grpo_strategy(result)
    (output,) = list(strategy.loss_outputs(strategy.prepare_from_rollout(result)))
    assert output["metrics"]["truncation_rate"] == pytest.approx(0.5)
    assert output["metrics"]["stop_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize("minibatch", [None, 2])
def test_replay_accepts_scalar_metadata_before_prompt_tensors(minibatch):
    result = _rollout_result()
    strategy = _grpo_strategy(result)
    prepare = strategy.prepare_from_rollout
    strategy.prepare_from_rollout = lambda result: {
        "temperature": torch.tensor(1.0),
        **prepare(result),
    }
    context = _run_replay(strategy, result, minibatch=minibatch)
    assert context.optimizer_step == (1 if minibatch is None else 2)


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
    )
    result = _rollout_result()

    calls = []
    original = strategy._compute_advantages

    def counting(*args, **kwargs):
        calls.append(len(calls))
        return original(*args, **kwargs)

    strategy._compute_advantages = counting
    context = _run_replay(strategy, result, epochs=2, minibatch=2)

    assert context.optimizer_step == 4
    assert len(calls) == 1


def test_grad_chunked_logprobs_match_full_path(monkeypatch):
    """Values, hidden grads, and lm_head grads agree with the full-tensor
    path across chunk boundaries (fp32, tolerance = fp32 noise)."""
    monkeypatch.setattr(
        "astrai.trainer.strategy.ops._CHUNK_LOGIT_BYTES", 128
    )  # ~2 rows/chunk
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


class _RefOnlyStrategy:
    def __init__(self):
        torch.manual_seed(5)
        self.ref_model = StubLM()


def _resume_builder(tmp_path, allow_reanchor=False):
    config = TrainConfig(
        strategy="sft",
        model=ComponentSpec(
            type="torch.nn:Linear", params={"in_features": 2, "out_features": 2}
        ),
        dataset=ComponentSpec(
            type="tests.support.rollout:RandomTokenDataset", params={"length": 1}
        ),
        optimizer=ComponentSpec(type="torch.optim:SGD", params={"lr": 0.0}),
        scheduler=ComponentSpec(
            type="torch.optim.lr_scheduler:StepLR",
            params={"step_size": 1, "gamma": 1.0},
        ),
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


def test_async_grpo_rejects_round_without_training_targets():
    result = _rollout_result()
    result.response_mask.zero_()
    strategy = _grpo_strategy(result)
    with pytest.raises(RuntimeError, match="no valid response tokens"):
        list(strategy.loss_outputs(strategy.prepare_from_rollout(result), 1))
