"""Unit tests for online rollout integration in :class:`BaseStrategy`.

Covers the shared rollout-trigger logic in ``BaseStrategy.__call__``
(runner injection, cache-driven refresh hook, ``step()`` callback) and
the per-strategy ``prepare_from_rollout`` mappings for both
:class:`GRPOStrategy` and :class:`DPOStrategy`.
"""

import pytest
import torch

from astrai.model.transformer import AutoRegressiveLM
from astrai.trainer.rollout import RolloutResult
from astrai.trainer.strategy import (
    DPOStrategy,
    GRPOStrategy,
    StrategyFactory,
)
from tests.helpers import FakeExecutor, make_frozen, make_model, make_rollout_config


def _make_rollout_result(B=2, G=4, P=6, R=8, device="cpu"):
    return RolloutResult(
        prompts=torch.randint(3, 200, (B, P), device=device),
        prompt_mask=torch.ones(B, P, dtype=torch.bool, device=device),
        responses=torch.randint(3, 200, (B, G, R), device=device),
        response_mask=torch.ones(B, G, R, dtype=torch.bool, device=device),
        rewards=torch.randn(B, G, device=device),
        logprobs_old=torch.zeros(B, G, R, device=device),
    )


class _RecordingRunner:
    """Fake RolloutRunner returning a fixed result with freshness tracking.

    Freshness is ``True`` on the first call after construction or after
    :meth:`swap_result`; ``False`` on subsequent cached calls -- mirroring
    the real ``RolloutRunner`` contract without invoking generation.
    """

    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.step_calls = 0
        self._fresh = True
        self.policy_version = result.policy_version
        self.weight_updates = []

    def __call__(self, batch):
        self.calls += 1
        fresh = self._fresh
        self._fresh = False
        return self.result, fresh

    def step(self):
        self.step_calls += 1

    def update_weights(self, policy_version):
        self.policy_version = policy_version
        self.weight_updates.append(policy_version)
        return policy_version

    def apply_weight_update(self, policy_version, update):
        result = update()
        self.update_weights(policy_version)
        return result

    def swap_result(self, result):
        self.result = result
        self._fresh = True


class _NoOpOptimizer:
    def step(self):
        return None


def _step(strat):
    strat.optimizer_step(_NoOpOptimizer())


def _make_grpo(device, executor=None):
    model, _ = make_model(device)
    ref_model = make_frozen(model, device)
    return GRPOStrategy(
        model=model,
        device=device,
        old_model=None,
        ref_model=ref_model,
        clip_eps=0.2,
        kl_coef=0.01,
        group_size=4,
        model_fn=lambda c=make_rollout_config(): AutoRegressiveLM(c).to(device=device),
        executor=executor or FakeExecutor(),
    )


def _make_dpo(device, executor=None):
    model, _ = make_model(device)
    ref_model = make_frozen(model, device)
    return DPOStrategy(
        model=model,
        device=device,
        ref_model=ref_model,
        beta=0.1,
        reduction="sum",
        model_fn=lambda c=make_rollout_config(): AutoRegressiveLM(c).to(device=device),
        executor=executor or FakeExecutor(),
    )


def test_factory_registers_online_aliases():
    assert StrategyFactory.is_registered("online_grpo")
    assert StrategyFactory.is_registered("online_dpo")
    assert StrategyFactory.get_component_class("online_grpo") is GRPOStrategy
    assert StrategyFactory.get_component_class("online_dpo") is DPOStrategy


@pytest.mark.parametrize("make_fn", ["_make_grpo", "_make_dpo"])
def test_online_strategies_support_online(device, make_fn):
    maker = {"_make_grpo": _make_grpo, "_make_dpo": _make_dpo}[make_fn]
    assert maker(device).supports_online() is True


def test_base_strategy_prepare_from_rollout_raises_by_default(device):
    from astrai.trainer.strategy import BaseStrategy

    class _Offline(BaseStrategy):
        def compute_loss(self, batch):
            return torch.tensor(0.0)

    strat = _Offline(model=torch.nn.Linear(1, 1), device="cpu")
    with pytest.raises(NotImplementedError):
        strat.prepare_from_rollout(_make_rollout_result(device="cpu"))


def test_base_strategy_supports_online_default_false():
    from astrai.trainer.strategy import BaseStrategy

    class _Offline(BaseStrategy):
        def compute_loss(self, batch):
            return torch.tensor(0.0)

    strat = _Offline(model=torch.nn.Linear(1, 1), device="cpu")
    assert strat.supports_online() is False


def test_grpo_prepare_from_rollout_mapping(device):
    strat = _make_grpo(device)
    r = _make_rollout_result(device=device)
    batch = strat.prepare_from_rollout(r)
    assert batch["prompts"] is r.prompts
    assert batch["prompt_mask"] is r.prompt_mask
    assert batch["responses"] is r.responses
    assert batch["masks"] is r.response_mask
    assert batch["rewards"] is r.rewards
    assert batch["logprobs_old"] is r.logprobs_old


def test_dpo_prepare_from_rollout_conditions_responses_on_prompt(device):
    strat = _make_dpo(device)
    r = _make_rollout_result(B=3, G=4, P=6, R=5, device=device)
    r.prompt_mask[0, :2] = False
    r.prompts[0, :2] = 0
    r.response_mask[1, :, -2:] = False
    r.responses[1, :, -2:] = 0
    batch = strat.prepare_from_rollout(r)

    assert batch["chosen"].shape == (3, 11)
    assert batch["rejected"].shape == (3, 11)
    assert batch["chosen_mask"].shape == (3, 11)
    assert batch["rejected_mask"].shape == (3, 11)
    idx = torch.arange(3, device=device)
    expected_best = r.responses[idx, r.rewards.argmax(dim=-1)]
    expected_worst = r.responses[idx, r.rewards.argmin(dim=-1)]
    expected_best_mask = r.response_mask[idx, r.rewards.argmax(dim=-1)]
    expected_worst_mask = r.response_mask[idx, r.rewards.argmin(dim=-1)]

    assert torch.equal(batch["chosen"][:, :6], r.prompts)
    assert torch.equal(batch["rejected"][:, :6], r.prompts)
    assert torch.equal(batch["chosen"][:, 6:], expected_best)
    assert torch.equal(batch["rejected"][:, 6:], expected_worst)
    assert not batch["chosen_mask"][:, :6].any()
    assert not batch["rejected_mask"][:, :6].any()
    assert torch.equal(batch["chosen_mask"][:, 6:], expected_best_mask)
    assert torch.equal(batch["rejected_mask"][:, 6:], expected_worst_mask)
    assert torch.equal(batch["chosen_attention_mask"][:, :6], r.prompt_mask)
    assert torch.equal(batch["rejected_attention_mask"][:, :6], r.prompt_mask)
    assert torch.equal(batch["chosen_attention_mask"][:, 6:], expected_best_mask)
    assert torch.equal(batch["rejected_attention_mask"][:, 6:], expected_worst_mask)


def test_dpo_prepare_from_rollout_same_response_keeps_distinct_prompts():
    strat = _make_dpo("cpu")
    r = _make_rollout_result(B=2, G=2, P=3, R=2, device="cpu")
    r.prompts = torch.tensor([[0, 11, 12], [21, 22, 23]])
    r.prompt_mask = torch.tensor([[False, True, True], [True, True, True]])
    shared_response = torch.tensor([101, 102])
    r.responses[:] = shared_response
    r.response_mask[:] = True
    r.rewards = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    batch = strat.prepare_from_rollout(r)

    assert torch.equal(batch["chosen"][:, 3:], shared_response.expand(2, -1))
    assert torch.equal(batch["rejected"][:, 3:], shared_response.expand(2, -1))
    assert torch.equal(batch["chosen"][:, :3], r.prompts)
    assert torch.equal(batch["rejected"][:, :3], r.prompts)
    assert not torch.equal(batch["chosen"][0], batch["chosen"][1])
    assert not batch["chosen_mask"][:, :3].any()
    assert not batch["rejected_mask"][:, :3].any()


def test_call_without_runner_accepts_behavior_logprobs_grpo(device):
    strat = _make_grpo(device)
    batch = {
        "prompts": torch.randint(3, 200, (2, 4), device=device),
        "responses": torch.randint(3, 200, (2, 4, 6), device=device),
        "masks": torch.ones(2, 4, 6, device=device),
        "rewards": torch.randn(2, 4, device=device),
        "logprobs_old": torch.zeros(2, 4, 6, device=device),
    }
    loss = strat(batch)["loss"]
    assert torch.isfinite(loss).item()


def test_call_with_runner_returns_finite_loss_grpo(device):
    strat = _make_grpo(device)
    strat.set_rollout_runner(_RecordingRunner(_make_rollout_result(device=device)))
    loss = strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})["loss"]
    assert torch.isfinite(loss).item()


def test_call_with_runner_returns_finite_loss_dpo(device):
    strat = _make_dpo(device)
    strat.set_rollout_runner(_RecordingRunner(_make_rollout_result(device=device)))
    loss = strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})["loss"]
    assert torch.isfinite(loss).item()


def test_call_invokes_runner_each_time(device):
    strat = _make_grpo(device)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    assert runner.calls == 2


def test_grpo_reuses_rollout_logprobs_without_old_model(device):
    strat = _make_grpo(device)
    result = _make_rollout_result(device=device)
    result.logprobs_old.normal_().requires_grad_()
    runner = _RecordingRunner(result)
    strat.set_rollout_runner(runner)
    assert strat.old_model is None
    loss = strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})["loss"]
    loss.backward()
    assert result.logprobs_old.grad is None


def test_grpo_reuses_same_cached_result(device):
    strat = _make_grpo(device)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    _step(strat)
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    _step(strat)
    assert runner.calls == 2
    assert runner.step_calls == 2


def test_grpo_accepts_new_rollout_result(device):
    strat = _make_grpo(device)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    _step(strat)
    runner.swap_result(_make_rollout_result(device=device))
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    _step(strat)
    assert runner.calls == 2
    assert runner.step_calls == 2


def test_dpo_no_sync_hook_when_new_rollout_result(device):
    """DPO has no old_model, so ``_on_rollout_refresh`` must be a no-op.

    We verify by ensuring no AttributeError is raised (DPO has no
    old_model) and that step is still called.
    """
    strat = _make_dpo(device)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    _step(strat)
    runner.swap_result(_make_rollout_result(device=device))
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    _step(strat)
    assert runner.step_calls == 2


def test_step_not_called_when_sync_gradients_false(device):
    executor = FakeExecutor(sync_gradients=False)
    strat = _make_grpo(device, executor=executor)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    assert runner.step_calls == 0


def test_step_called_when_sync_gradients_true(device):
    executor = FakeExecutor(sync_gradients=True)
    strat = _make_grpo(device, executor=executor)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    _step(strat)
    assert runner.step_calls == 1
    assert runner.weight_updates == [1]
    assert strat.policy_version == 1


def test_post_hoc_online_optimizer_step_is_rejected(device):
    strat = _make_grpo(device)
    strat.set_rollout_runner(_RecordingRunner(_make_rollout_result(device=device)))

    with pytest.raises(RuntimeError, match="strategy.optimizer_step"):
        strat.on_optimizer_step()


def test_optimizer_step_publishes_version_with_weight_update(device):
    strat = _make_grpo(device)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    parameter = next(strat.model.parameters())
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.SGD(strat.model.parameters(), lr=0.1)
    before = parameter.detach().clone()

    strat.optimizer_step(optimizer)

    assert not torch.equal(parameter, before)
    assert runner.weight_updates == [1]
    assert runner.step_calls == 1


def test_loss_is_differentiable_dpo(device):
    strat = _make_dpo(device)
    strat.set_rollout_runner(_RecordingRunner(_make_rollout_result(device=device)))
    loss = strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})["loss"]
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in strat.model.parameters()
    )
    assert has_grad


def test_ref_model_not_updated_by_backward_dpo(device):
    strat = _make_dpo(device)
    strat.set_rollout_runner(_RecordingRunner(_make_rollout_result(device=device)))
    loss = strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})["loss"]
    loss.backward()
    for p in strat.ref_model.parameters():
        assert p.grad is None
