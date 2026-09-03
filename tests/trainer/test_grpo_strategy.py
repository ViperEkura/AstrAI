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


def test_grpo_reuses_supplied_behavior_logprobs(grpo_strategy):
    """A rollout batch must not forward the old policy again."""
    strategy, device = grpo_strategy

    class _FailingOldPolicy(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("old policy forward should not run")

    strategy.old_model = _FailingOldPolicy()
    batch = _make_batch(device=device)
    batch["logprobs_old"] = torch.zeros_like(batch["responses"], dtype=torch.float)

    loss = strategy.compute_loss(batch)
    assert torch.isfinite(loss).item()


def test_grpo_requires_behavior_source(grpo_strategy):
    strategy, device = grpo_strategy
    strategy.old_model = None
    with pytest.raises(ValueError, match="must provide logprobs_old"):
        strategy.compute_loss(_make_batch(device=device))


@pytest.mark.parametrize("invalid", ["shape", "nonfinite"])
def test_grpo_rejects_invalid_behavior_logprobs(grpo_strategy, invalid):
    strategy, device = grpo_strategy
    batch = _make_batch(device=device)
    if invalid == "shape":
        batch["logprobs_old"] = torch.zeros(1, device=device)
        match = "shape must match responses"
    else:
        batch["logprobs_old"] = torch.zeros_like(batch["responses"], dtype=torch.float)
        batch["logprobs_old"][0, 0, 0] = float("nan")
        match = "only finite values"
    with pytest.raises(ValueError, match=match):
        strategy.compute_loss(batch)


def test_grpo_symmetric_clip_is_backward_compatible(grpo_strategy):
    strategy, _device = grpo_strategy
    assert strategy.clip_eps == pytest.approx(0.2)
    assert strategy.clip_eps_low == pytest.approx(0.2)
    assert strategy.clip_eps_high == pytest.approx(0.2)
    assert strategy.loss_aggregation == "token"


def test_grpo_dapo_clip_higher_changes_positive_advantage_bound(
    grpo_strategy, monkeypatch
):
    strategy, device = grpo_strategy
    batch = {
        "prompts": torch.tensor([[5]], device=device),
        "responses": torch.tensor([[[6], [7]]], device=device),
        "masks": torch.ones(1, 2, 1, device=device),
        "rewards": torch.tensor([[1.0, -1.0]], device=device),
    }
    policy_logprobs = torch.log(
        torch.tensor([[1.25], [0.75]], device=device, dtype=torch.float32)
    )
    zeros = torch.zeros_like(policy_logprobs)

    def policy_loss(clip_eps_high):
        strategy.clip_eps_high = clip_eps_high
        outputs = iter(
            [
                policy_logprobs,
                zeros,
                policy_logprobs,
            ]
        )

        def fake_get_logprobs(*_args, **_kwargs):
            return {
                "logprobs": next(outputs),
                "aux_loss": None,
                "router_stats": None,
            }

        monkeypatch.setattr(strategy_module, "get_logprobs", fake_get_logprobs)
        return strategy.compute_loss_output(batch)["metrics"]["policy_loss"]

    assert policy_loss(0.2) == pytest.approx(-0.2, abs=1e-6)
    assert policy_loss(0.28) == pytest.approx(-0.225, abs=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"clip_eps_low": 1.0}, "clip_eps_low"),
        ({"clip_eps_high": float("nan")}, "clip_eps_high"),
        (
            {"clip_eps_low": 0.3, "clip_eps_high": 0.2},
            "greater than or equal",
        ),
    ],
)
def test_grpo_rejects_invalid_asymmetric_clip(grpo_strategy, kwargs, match):
    strategy, device = grpo_strategy
    with pytest.raises(ValueError, match=match):
        GRPOStrategy(
            model=strategy.model,
            device=device,
            old_model=strategy.old_model,
            ref_model=strategy.ref_model,
            clip_eps=0.2,
            executor=FakeExecutor(),
            **kwargs,
        )


def test_grpo_token_and_sequence_aggregation_weight_lengths_differently(
    grpo_strategy,
):
    strategy, device = grpo_strategy
    losses = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]]], device=device)
    masks = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]], device=device)

    strategy.loss_aggregation = "token"
    token_loss = strategy._reduce_token_loss(losses, masks)
    strategy.loss_aggregation = "sequence"
    sequence_loss = strategy._reduce_token_loss(losses, masks)

    assert token_loss.item() == pytest.approx(14.0 / 6.0)
    assert sequence_loss.item() == pytest.approx(2.0)


def test_grpo_dapo_soft_overlong_reward_shaping(grpo_strategy):
    strategy, device = grpo_strategy
    strategy.overlong_max_len = 8
    strategy.overlong_buffer_len = 2
    strategy.overlong_penalty_scale = 0.5
    rewards = torch.zeros(1, 4, device=device)
    masks = torch.zeros(1, 4, 8, device=device)
    for index, length in enumerate((5, 6, 7, 8)):
        masks[0, index, :length] = 1

    shaped, penalty = strategy._shape_overlong_rewards(rewards, masks)

    assert penalty is not None
    torch.testing.assert_close(
        penalty, torch.tensor([[0.0, 0.0, -0.5, -1.0]], device=device)
    )
    torch.testing.assert_close(
        shaped, torch.tensor([[0.0, 0.0, -0.25, -0.5]], device=device)
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"loss_aggregation": "batch"}, "loss_aggregation"),
        ({"overlong_buffer_len": 4}, "requires overlong_max_len"),
        (
            {"overlong_max_len": 8, "overlong_buffer_len": 9},
            "no greater than overlong_max_len",
        ),
        (
            {
                "overlong_max_len": 8,
                "overlong_buffer_len": 2,
                "overlong_penalty_scale": -0.1,
            },
            "overlong_penalty_scale",
        ),
    ],
)
def test_grpo_rejects_invalid_dapo_objective_options(grpo_strategy, kwargs, match):
    strategy, device = grpo_strategy
    with pytest.raises((TypeError, ValueError), match=match):
        GRPOStrategy(
            model=strategy.model,
            device=device,
            old_model=strategy.old_model,
            ref_model=strategy.ref_model,
            executor=FakeExecutor(),
            **kwargs,
        )


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
