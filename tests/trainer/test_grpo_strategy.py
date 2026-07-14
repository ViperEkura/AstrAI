import pytest
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.transformer import AutoRegressiveLM
from astrai.trainer.strategy import GRPOStrategy


class _FakeExecutor:
    """Minimal executor stub providing ``unwrap_model`` for ref model creation."""

    def unwrap_model(self, model):
        return model.state_dict()


def _make_config(vocab_size=200, max_len=64):
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
    config = _make_config()
    model = AutoRegressiveLM(config).to(device=device)
    return model, config


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
    # All response tokens valid.
    masks = torch.ones(batch_size, group_size, response_len, device=device)
    # Distinct rewards per group member so std > 0.
    rewards = torch.randn(batch_size, group_size, device=device)
    return {
        "prompts": prompts,
        "responses": responses,
        "masks": masks,
        "rewards": rewards,
    }


def _make_frozen_copy(model, device):
    """Create a frozen copy of ``model`` with independent weights loaded."""
    config = _make_config()
    copy = AutoRegressiveLM(config).to(device=device)
    copy.load_state_dict(model.state_dict())
    copy.requires_grad_(False)
    copy.eval()
    return copy


@pytest.fixture
def grpo_strategy():
    """Build a GRPOStrategy with a small real model and fake executor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = _make_model(device)
    old_model = _make_frozen_copy(model, device)
    ref_model = _make_frozen_copy(model, device)

    strategy = GRPOStrategy(
        model=model,
        device=device,
        old_model=old_model,
        ref_model=ref_model,
        clip_eps=0.2,
        kl_coef=0.01,
        group_size=4,
        model_fn=lambda c=config: AutoRegressiveLM(c).to(device=device),
        executor=_FakeExecutor(),
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
    # At least some parameter should receive a gradient.
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in strategy.model.parameters()
    )
    assert has_grad


def test_grpo_ref_model_not_updated(grpo_strategy):
    """Backward should not populate gradients on ref_model."""
    strategy, device = grpo_strategy
    batch = _make_batch(device=device)
    loss = strategy.compute_loss(batch)
    loss.backward()
    for p in strategy.ref_model.parameters():
        assert p.grad is None


def test_grpo_old_model_not_updated(grpo_strategy):
    """Backward should not populate gradients on old_model."""
    strategy, device = grpo_strategy
    batch = _make_batch(device=device)
    loss = strategy.compute_loss(batch)
    loss.backward()
    for p in strategy.old_model.parameters():
        assert p.grad is None


def test_grpo_prompt_tokens_masked(grpo_strategy):
    """When only prompt-equivalent tokens are unmasked (response mask all 0),
    the policy loss should be zero (no valid tokens contribute)."""
    strategy, device = grpo_strategy
    batch = _make_batch(device=device)
    # Zero out all response masks → no response token contributes.
    batch["masks"] = torch.zeros_like(batch["masks"])
    loss = strategy.compute_loss(batch)
    # With no valid tokens, policy_loss term is 0 and KL term is 0.
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_grpo_identical_rewards_zero_advantage(grpo_strategy):
    """When all group rewards are identical, advantage is 0 → policy_loss is 0.
    Only the KL term remains (which is 0 when policy == ref at init)."""
    strategy, device = grpo_strategy
    batch = _make_batch(device=device)
    batch["rewards"] = torch.ones(batch["rewards"].shape, device=device)
    loss = strategy.compute_loss(batch)
    # At init policy == old == ref, so ratio == 1, KL == 0; advantage == 0.
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_grpo_sync_old_model(grpo_strategy):
    """sync_old_model copies current policy weights into old_model."""
    strategy, device = grpo_strategy
    # Perturb policy model so it differs from old.
    with torch.no_grad():
        for p in strategy.model.parameters():
            p.add_(0.05)
    # old_model should still hold original weights (differ from policy).
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


def test_grpo_partial_mask(grpo_strategy):
    """Only the first half of response tokens are valid."""
    strategy, device = grpo_strategy
    batch = _make_batch(device=device)
    B, G, R = batch["masks"].shape
    half = R // 2
    batch["masks"][:, :, half:] = 0.0
    loss = strategy.compute_loss(batch)
    assert torch.isfinite(loss).item()


def test_grpo_clipping_effect(grpo_strategy):
    """After diverging policy from ref, ratio should be clipped to [1-eps, 1+eps]
    on the surrogate. Verify loss is finite and non-zero for distinct rewards."""
    strategy, device = grpo_strategy
    # Diverge policy from ref.
    with torch.no_grad():
        for p in strategy.model.parameters():
            p.add_(0.3)
    batch = _make_batch(device=device)
    loss = strategy.compute_loss(batch)
    assert torch.isfinite(loss).item()
    # With distinct rewards and diverged policy, loss should be non-trivial.
    assert loss.abs().item() > 1e-4


def test_grpo_no_reduction_param():
    """GRPOStrategy.__init__ must not accept ``reduction`` (removed)."""
    import inspect

    sig = inspect.signature(GRPOStrategy.__init__)
    assert "reduction" not in sig.parameters


def test_grpo_shapes_3d_batch(grpo_strategy):
    """Verify compute_loss handles non-square prompt/response lengths."""
    strategy, device = grpo_strategy
    batch = _make_batch(
        batch_size=3, group_size=4, prompt_len=10, response_len=8, device=device
    )
    loss = strategy.compute_loss(batch)
    assert torch.isfinite(loss).item()
