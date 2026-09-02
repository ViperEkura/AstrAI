import pytest
import torch

import astrai.trainer.strategy as strategy_module
from astrai.model.transformer import AutoRegressiveLM
from astrai.trainer.strategy import GRPOStrategy
from tests.helpers import FakeExecutor, make_frozen, make_model


def _make_batch(
    batch_size=2, group_size=4, prompt_len=8, response_len=12, device="cpu"
):
    """Construct a GRPO batch with deterministic shapes.

    Returns dict with prompts [B, P], responses [B, G, R], masks [B, G, R],
    rewards [B, G].
    """
    prompts = torch.randint(0, 200, (batch_size, prompt_len), device=device)
    responses = torch.randint(
        0, 200, (batch_size, group_size, response_len), device=device
    )
    masks = torch.ones(batch_size, group_size, response_len, device=device)
    rewards = torch.randn(batch_size, group_size, device=device)
    return {
        "prompts": prompts,
        "responses": responses,
        "masks": masks,
        "rewards": rewards,
    }


@pytest.fixture
def grpo_strategy(device):
    """Build a GRPOStrategy with a small real model and fake executor."""
    model, config = make_model(device)
    old_model = make_frozen(model, device)
    ref_model = make_frozen(model, device)

    strategy = GRPOStrategy(
        model=model,
        device=device,
        old_model=old_model,
        ref_model=ref_model,
        clip_eps=0.2,
        kl_coef=0.01,
        group_size=4,
        model_fn=lambda c=config: AutoRegressiveLM(c).to(device=device),
        executor=FakeExecutor(),
    )
    return strategy, device


def test_grpo_loss_is_finite(grpo_strategy):
    """compute_loss returns a finite scalar."""
    strategy, device = grpo_strategy
    batch = _make_batch(device=device)
    loss = strategy.compute_loss(batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()


def test_grpo_loss_backward(grpo_strategy):
    """Loss is differentiable w.r.t. policy model parameters."""
    strategy, device = grpo_strategy
    batch = _make_batch(device=device)
    loss = strategy.compute_loss(batch)
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in strategy.model.parameters()
    )
    assert has_grad


def test_grpo_default_advantages_remain_group_standardized(grpo_strategy):
    strategy, device = grpo_strategy
    rewards = torch.tensor([[1.0, 2.0, 5.0]], device=device)
    advantages = strategy._group_advantages(rewards, eps=1e-8)
    expected = (rewards - rewards.mean(dim=-1, keepdim=True)) / rewards.std(
        dim=-1, keepdim=True, unbiased=False
    )
    torch.testing.assert_close(advantages, expected)


def test_dr_grpo_uses_centered_rewards_and_fixed_completion_budget(
    grpo_strategy, monkeypatch
):
    base_strategy, device = grpo_strategy
    strategy = GRPOStrategy(
        model=base_strategy.model,
        device=device,
        old_model=base_strategy.old_model,
        ref_model=base_strategy.ref_model,
        clip_eps=0.2,
        kl_coef=0.0,
        group_size=2,
        loss_variant="dr_grpo",
        max_completion_length=8,
        executor=FakeExecutor(),
    )
    batch = {
        "prompts": torch.tensor([[5]], device=device),
        "responses": torch.tensor([[[6, 7, 8, 9], [10, 0, 0, 0]]], device=device),
        "masks": torch.tensor(
            [[[1, 1, 1, 1], [1, 0, 0, 0]]], device=device, dtype=torch.bool
        ),
        "rewards": torch.tensor([[4.0, 1.0]], device=device),
    }
    zeros = torch.zeros(2, 4, device=device)

    def fake_get_logprobs(*_args, **_kwargs):
        return {"logprobs": zeros, "aux_loss": None, "router_stats": None}

    monkeypatch.setattr(strategy_module, "get_logprobs", fake_get_logprobs)
    output = strategy.compute_loss_output(batch)

    torch.testing.assert_close(
        strategy._group_advantages(batch["rewards"], eps=1e-8),
        torch.tensor([[1.5, -1.5]], device=device),
    )
    # Sum of masked per-token losses is -(4 * 1.5 - 1 * 1.5) = -4.5;
    # Dr.GRPO divides by B * G * fixed_budget = 1 * 2 * 8 = 16.
    assert output["metrics"]["policy_loss"] == pytest.approx(-4.5 / 16, abs=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"loss_variant": "unknown"}, ValueError, "loss_variant"),
        ({"loss_variant": "dr_grpo"}, ValueError, "max_completion_length"),
        (
            {"loss_variant": "dr_grpo", "max_completion_length": 0},
            ValueError,
            "positive",
        ),
        (
            {"loss_variant": "dr_grpo", "max_completion_length": 4.5},
            TypeError,
            "integer",
        ),
    ],
)
def test_grpo_rejects_invalid_loss_variant_config(grpo_strategy, kwargs, error, match):
    strategy, device = grpo_strategy
    with pytest.raises(error, match=match):
        GRPOStrategy(
            model=strategy.model,
            device=device,
            old_model=strategy.old_model,
            ref_model=strategy.ref_model,
            executor=FakeExecutor(),
            **kwargs,
        )


def test_dr_grpo_rejects_response_longer_than_fixed_budget(grpo_strategy):
    base_strategy, device = grpo_strategy
    strategy = GRPOStrategy(
        model=base_strategy.model,
        device=device,
        old_model=base_strategy.old_model,
        ref_model=base_strategy.ref_model,
        loss_variant="dr_grpo",
        max_completion_length=3,
        executor=FakeExecutor(),
    )
    batch = _make_batch(batch_size=1, group_size=2, response_len=4, device=device)
    with pytest.raises(ValueError, match="exceeds max_completion_length"):
        strategy.compute_loss_output(batch)


@pytest.mark.parametrize("model_name", ["ref_model", "old_model"])
def test_grpo_frozen_models_not_updated(grpo_strategy, model_name):
    """Backward should not populate gradients on ref_model or old_model."""
    strategy, device = grpo_strategy
    batch = _make_batch(device=device)
    loss = strategy.compute_loss(batch)
    loss.backward()
    for p in getattr(strategy, model_name).parameters():
        assert p.grad is None


def test_grpo_prompt_tokens_masked(grpo_strategy):
    """When only prompt-equivalent tokens are unmasked (response mask all 0),
    the policy loss should be zero (no valid tokens contribute)."""
    strategy, device = grpo_strategy
    batch = _make_batch(device=device)
    batch["masks"] = torch.zeros_like(batch["masks"])
    loss = strategy.compute_loss(batch)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_grpo_identical_rewards_zero_advantage(grpo_strategy):
    """When all group rewards are identical, advantage is 0 -> policy_loss is 0.
    Only the KL term remains (which is 0 when policy == ref at init)."""
    strategy, device = grpo_strategy
    batch = _make_batch(device=device)
    batch["rewards"] = torch.ones(batch["rewards"].shape, device=device)
    loss = strategy.compute_loss(batch)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_grpo_sync_old_model(grpo_strategy):
    """sync_old_model copies current policy weights into old_model."""
    strategy, device = grpo_strategy
    with torch.no_grad():
        for p in strategy.model.parameters():
            p.add_(0.05)
    policy_sd = strategy.model.state_dict()
    old_sd = strategy.old_model.state_dict()
    differs_before = any(
        not torch.allclose(policy_sd[k], old_sd[k]) for k in policy_sd if k in old_sd
    )
    assert differs_before

    strategy.sync_old_model()

    old_sd_after = strategy.old_model.state_dict()
    matches = all(
        torch.allclose(policy_sd[k], old_sd_after[k])
        for k in policy_sd
        if k in old_sd_after
    )
    assert matches
