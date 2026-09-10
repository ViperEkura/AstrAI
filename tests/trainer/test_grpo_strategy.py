import pytest
import torch

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


def test_grpo_optimizer_step_syncs_old_model(grpo_strategy):
    """optimizer_step must refresh old_model after each update."""
    strategy, device = grpo_strategy

    class _SteppedOptimizer:
        def step(self):
            with torch.no_grad():
                for p in strategy.model.parameters():
                    p.add_(0.05)

    strategy.optimizer_step(_SteppedOptimizer())

    policy_sd = strategy.model.state_dict()
    old_sd = strategy.old_model.state_dict()
    assert all(
        torch.allclose(policy_sd[k], old_sd[k]) for k in policy_sd if k in old_sd
    )


def test_online_grpo_optimizer_step_skips_sync(grpo_strategy):
    """old_model=None (online) must not attempt a sync."""
    strategy, device = grpo_strategy
    strategy.old_model = None

    class _SteppedOptimizer:
        def step(self):
            return None

    strategy.optimizer_step(_SteppedOptimizer())
