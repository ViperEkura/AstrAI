from tests.support.rollout import make_rollout_result

"""Rollout-to-batch mappings and policy/reference gradient boundaries."""

import pytest
import torch

from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.trainer.strategy import (
    BaseStrategy,
    DPOStrategy,
    GRPOStrategy,
    StrategyFactory,
)
from tests.support.models import make_frozen, make_model, make_rollout_config
from tests.support.rollout import FakeExecutor


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
def test_online_strategies_support_online(cpu_device, make_fn):
    maker = {"_make_grpo": _make_grpo, "_make_dpo": _make_dpo}[make_fn]
    assert maker(cpu_device).supports_online() is True


def test_base_strategy_prepare_from_rollout_raises_by_default(cpu_device):
    class _Offline(BaseStrategy):
        def compute_loss(self, batch):
            return torch.tensor(0.0)

    strat = _Offline(model=torch.nn.Linear(1, 1), device="cpu")
    with pytest.raises(NotImplementedError):
        strat.prepare_from_rollout(make_rollout_result(device="cpu"))


def test_base_strategy_supports_online_default_false():
    class _Offline(BaseStrategy):
        def compute_loss(self, batch):
            return torch.tensor(0.0)

    strat = _Offline(model=torch.nn.Linear(1, 1), device="cpu")
    assert strat.supports_online() is False


def test_grpo_prepare_from_rollout_mapping(cpu_device):
    strat = _make_grpo(cpu_device)
    r = make_rollout_result(device=cpu_device)
    batch = strat.prepare_from_rollout(r)
    assert batch["prompts"] is r.prompts
    assert batch["prompt_mask"] is r.prompt_mask
    assert batch["responses"] is r.responses
    assert batch["masks"] is r.response_mask
    assert batch["rewards"] is r.rewards
    assert batch["logprobs_old"] is r.logprobs_old


def test_dpo_prepare_from_rollout_conditions_responses_on_prompt(cpu_device):
    strat = _make_dpo(cpu_device)
    r = make_rollout_result(B=3, G=4, P=6, R=5, device=cpu_device)
    r.prompt_mask[0, :2] = False
    r.prompts[0, :2] = 0
    r.response_mask[1, :, -2:] = False
    r.responses[1, :, -2:] = 0
    batch = strat.prepare_from_rollout(r)

    assert batch["chosen"].shape == (3, 11)
    assert batch["rejected"].shape == (3, 11)
    assert batch["chosen_mask"].shape == (3, 11)
    assert batch["rejected_mask"].shape == (3, 11)
    idx = torch.arange(3, device=cpu_device)
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
    r = make_rollout_result(B=2, G=2, P=3, R=2, device="cpu")
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


def test_call_without_runner_accepts_behavior_logprobs_grpo(cpu_device):
    strat = _make_grpo(cpu_device)
    batch = {
        "prompts": torch.randint(3, 200, (2, 4), device=cpu_device),
        "responses": torch.randint(3, 200, (2, 4, 6), device=cpu_device),
        "masks": torch.ones(2, 4, 6, device=cpu_device),
        "rewards": torch.randn(2, 4, device=cpu_device),
        "logprobs_old": torch.zeros(2, 4, 6, device=cpu_device),
    }
    loss = strat(batch)["loss"]
    assert torch.isfinite(loss).item()


def test_prepared_rollout_returns_finite_loss_grpo(cpu_device):
    strat = _make_grpo(cpu_device)
    batch = strat.prepare_from_rollout(make_rollout_result(device=cpu_device))
    loss = strat(batch)["loss"]
    assert torch.isfinite(loss).item()


def test_prepared_rollout_returns_finite_loss_dpo(cpu_device):
    strat = _make_dpo(cpu_device)
    batch = strat.prepare_from_rollout(make_rollout_result(device=cpu_device))
    loss = strat(batch)["loss"]
    assert torch.isfinite(loss).item()


def test_grpo_reuses_rollout_logprobs_without_old_model(cpu_device):
    strat = _make_grpo(cpu_device)
    result = make_rollout_result(device=cpu_device)
    result.logprobs_old.normal_().requires_grad_()
    batch = strat.prepare_from_rollout(result)
    assert strat.old_model is None
    loss = strat(batch)["loss"]
    loss.backward()
    assert result.logprobs_old.grad is None


def test_loss_is_differentiable_dpo(cpu_device):
    strat = _make_dpo(cpu_device)
    batch = strat.prepare_from_rollout(make_rollout_result(device=cpu_device))
    loss = strat(batch)["loss"]
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in strat.model.parameters()
    )
    assert has_grad


def test_ref_model_not_updated_by_backward_dpo(cpu_device):
    strat = _make_dpo(cpu_device)
    batch = strat.prepare_from_rollout(make_rollout_result(device=cpu_device))
    loss = strat(batch)["loss"]
    loss.backward()
    for p in strat.ref_model.parameters():
        assert p.grad is None
