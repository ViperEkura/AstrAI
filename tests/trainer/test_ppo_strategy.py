"""Unit tests for PPO: GAE numerics, the ValueModel critic, and PPOStrategy."""

import pytest
import torch

import astrai.trainer.strategy as strategy_module
from astrai.model.value import ValueModel
from astrai.trainer.rollout import RolloutResult
from astrai.trainer.strategy import (
    PPOStrategy,
    StrategyFactory,
    compute_gae,
)
from tests.helpers import FakeExecutor, make_frozen, make_model, make_rollout_config


def _make_batch(
    batch_size=2, group_size=4, prompt_len=8, response_len=12, device="cpu"
):
    """Construct a PPO batch with deterministic shapes.

    Returns dict with prompts [B, P], responses [B, G, R], masks [B, G, R],
    rewards [B, G], logprobs_old [B, G, R].
    """
    return {
        "prompts": torch.randint(0, 200, (batch_size, prompt_len), device=device),
        "responses": torch.randint(
            0, 200, (batch_size, group_size, response_len), device=device
        ),
        "masks": torch.ones(batch_size, group_size, response_len, device=device),
        "rewards": torch.randn(batch_size, group_size, device=device),
        "logprobs_old": torch.zeros(
            batch_size, group_size, response_len, device=device
        ),
    }


def _make_value_model(policy_model, device):
    """Build a ValueModel whose backbone warm-starts from the policy."""
    critic = ValueModel(policy_model.config).to(device=device)
    result = critic.load_state_dict(policy_model.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert all(key.startswith("value_head.") for key in result.missing_keys)
    return critic


@pytest.fixture
def ppo_strategy(device):
    model, _ = make_model(device)
    critic = _make_value_model(model, device)
    strategy = PPOStrategy(
        model=model,
        device=device,
        critic=critic,
        critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-3),
        ref_model=make_frozen(model, device),
        clip_eps=0.2,
        kl_coef=0.01,
        gamma=1.0,
        gae_lambda=0.95,
        vf_coef=0.5,
        executor=FakeExecutor(),
    )
    return strategy, device


# ============== compute_gae ==============


def test_gae_monte_carlo_when_values_zero(device):
    """γ=1, λ=1, V=0: advantage and return equal the terminal reward at
    every valid position (Monte-Carlo return)."""
    B, G, R = 2, 3, 4
    rewards = torch.zeros(B, G, R, device=device)
    rewards[..., -1] = 1.0
    values = torch.zeros(B, G, R, device=device)
    mask = torch.ones(B, G, R, dtype=torch.bool, device=device)

    advantages, returns = compute_gae(rewards, values, mask, gamma=1.0, gae_lambda=1.0)

    assert torch.allclose(advantages, torch.full_like(rewards, 1.0))
    assert torch.allclose(returns, torch.full_like(rewards, 1.0))


def test_gae_lambda_zero_is_one_step_td(device):
    """λ=0: advantage degenerates to the TD residual δ_t."""
    torch.manual_seed(0)
    rewards = torch.zeros(1, 1, 3, device=device)
    rewards[0, 0, -1] = 2.0
    values = torch.tensor([[[0.5, 1.0, -0.5]]], device=device)
    mask = torch.ones(1, 1, 3, dtype=torch.bool, device=device)

    advantages, returns = compute_gae(rewards, values, mask, gamma=0.9, gae_lambda=0.0)

    # δ_2 = r + 0 - V_2 = 2.5;  δ_1 = 0 + 0.9·V_2 - V_1 = -1.45;
    # δ_0 = 0 + 0.9·V_1 - V_0 = 0.4
    expected = torch.tensor([[[0.4, -1.45, 2.5]]], device=device)
    assert torch.allclose(advantages, expected, atol=1e-6)
    assert torch.allclose(returns, advantages + values, atol=1e-6)


def test_gae_hand_computed_discounted_case(device):
    """γ=0.9, λ=0.8 against a hand-rolled backward accumulation."""
    rewards = torch.tensor([[[0.0, 0.0, 1.0]]], device=device)
    values = torch.tensor([[[0.1, 0.2, 0.3]]], device=device)
    mask = torch.ones(1, 1, 3, dtype=torch.bool, device=device)
    gamma, lam = 0.9, 0.8

    advantages, returns = compute_gae(rewards, values, mask, gamma, lam)

    delta2 = 1.0 + 0.0 - 0.3
    gae2 = delta2
    delta1 = 0.0 + gamma * 0.3 - 0.2
    gae1 = delta1 + gamma * lam * gae2
    delta0 = 0.0 + gamma * 0.2 - 0.1
    gae0 = delta0 + gamma * lam * gae1
    expected = torch.tensor([[[gae0, gae1, gae2]]], device=device)
    assert torch.allclose(advantages, expected, atol=1e-6)
    assert torch.allclose(returns, expected + values, atol=1e-6)


def test_gae_padding_does_not_leak(device):
    """Garbage values at padded positions must not change valid outputs."""
    torch.manual_seed(1)
    B, G, R = 2, 2, 5
    rewards = torch.zeros(B, G, R, device=device)
    rewards[0, 0, 2] = 1.0  # terminal at position 2 of a length-3 response
    values = torch.randn(B, G, R, device=device)
    mask = torch.ones(B, G, R, dtype=torch.bool, device=device)
    mask[0, 0, 3:] = False
    mask[1, :, 2:] = False
    rewards[0, 0, 3:] = 100.0  # reward garbage in padding must be ignored

    advantages, returns = compute_gae(rewards, values, mask, gamma=0.9, gae_lambda=0.9)

    assert torch.allclose(advantages[0, 0, 3:], torch.zeros_like(advantages[0, 0, 3:]))
    assert torch.allclose(returns[0, 0, 3:], torch.zeros_like(returns[0, 0, 3:]))
    # The terminal reward at position 2 still drives a finite advantage.
    assert advantages[0, 0, 2] != 0.0


def test_gae_empty_response_is_all_zero(device):
    """A fully padded response yields zero advantages and returns."""
    rewards = torch.zeros(1, 1, 3, device=device)
    values = torch.randn(1, 1, 3, device=device)
    mask = torch.zeros(1, 1, 3, dtype=torch.bool, device=device)

    advantages, returns = compute_gae(rewards, values, mask, 1.0, 0.95)

    assert torch.count_nonzero(advantages) == 0
    assert torch.count_nonzero(returns) == 0


# ============== ValueModel ==============


def test_value_model_trunk_matches_policy_hidden_states(device):
    """ValueModel's trunk reproduces AutoRegressiveLM's hidden states.

    Pins the duplicated trunk pass in ``ValueModel.forward`` to the policy
    forward: a ones-initialized value head must return the row-wise sum of
    the policy's ``hidden_states``.
    """
    model, _ = make_model(device)
    critic = _make_value_model(model, device)
    with torch.no_grad():
        critic.value_head.weight.fill_(1.0)
        critic.value_head.bias.zero_()

    torch.manual_seed(2)
    input_ids = torch.randint(0, 200, (2, 10), device=device)
    input_mask = torch.ones(2, 10, dtype=torch.bool, device=device)
    input_mask[1, :3] = False

    with torch.no_grad():
        policy_hidden = model(input_ids, input_mask=input_mask)["hidden_states"]
        values = critic(input_ids, input_mask=input_mask)["values"]

    assert values.shape == (2, 10)
    assert torch.allclose(values, policy_hidden.sum(dim=-1), atol=1e-5)


def test_value_model_zero_head_outputs_zero(device):
    model, _ = make_model(device)
    critic = _make_value_model(model, device)
    input_ids = torch.randint(0, 200, (2, 8), device=device)
    with torch.no_grad():
        values = critic(input_ids)["values"]
    assert torch.count_nonzero(values) == 0


def test_value_model_rejects_packed_inference_input(device):
    critic = ValueModel(make_rollout_config()).to(device=device)
    with pytest.raises(ValueError, match="critic input_ids"):
        critic(torch.randint(0, 200, (16,), device=device))


# ============== PPOStrategy ==============


def test_factory_registers_online_ppo():
    assert StrategyFactory.is_registered("online_ppo")
    assert StrategyFactory.get_component_class("online_ppo") is PPOStrategy


def test_ppo_supports_online(ppo_strategy):
    strategy, _ = ppo_strategy
    assert strategy.supports_online() is True


def test_ppo_loss_is_finite_and_differentiable(ppo_strategy):
    strategy, device = ppo_strategy
    batch = _make_batch(device=device)
    loss = strategy.compute_loss(batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss).item()
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in strategy.model.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in strategy.critic.parameters()
    )


def test_ppo_requires_behavior_logprobs(ppo_strategy):
    strategy, device = ppo_strategy
    batch = _make_batch(device=device)
    del batch["logprobs_old"]
    with pytest.raises(ValueError, match="logprobs_old"):
        strategy.compute_loss(batch)


@pytest.mark.parametrize("invalid", ["shape", "nonfinite"])
def test_ppo_rejects_invalid_behavior_logprobs(ppo_strategy, invalid):
    strategy, device = ppo_strategy
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


def test_ppo_ref_model_not_updated_by_backward(ppo_strategy):
    strategy, device = ppo_strategy
    loss = strategy.compute_loss(_make_batch(device=device))
    loss.backward()
    for p in strategy.ref_model.parameters():
        assert p.grad is None


def test_ppo_zero_advantage_and_zero_critic_gives_zero_loss(ppo_strategy):
    """A zero-head critic, zero advantages, and zero returns → zero loss."""
    strategy, device = ppo_strategy
    batch = _make_batch(device=device)
    batch["advantages"] = torch.zeros_like(batch["responses"], dtype=torch.float)
    batch["returns"] = torch.zeros_like(batch["responses"], dtype=torch.float)
    loss = strategy.compute_loss(batch)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_ppo_all_masked_response_tokens_zero_loss(ppo_strategy):
    strategy, device = ppo_strategy
    batch = _make_batch(device=device)
    batch["masks"] = torch.zeros_like(batch["masks"])
    loss = strategy.compute_loss(batch)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_ppo_uses_supplied_advantages_without_recomputation(ppo_strategy):
    """Explicit advantages/returns must short-circuit GAE computation."""
    strategy, device = ppo_strategy
    batch = _make_batch(device=device)

    def _fail(*args, **kwargs):
        raise AssertionError("advantages were supplied; GAE must not run")

    strategy._compute_advantages = _fail
    batch["advantages"] = torch.ones_like(batch["responses"], dtype=torch.float)
    batch["returns"] = torch.zeros_like(batch["responses"], dtype=torch.float)
    loss = strategy.compute_loss(batch)
    assert torch.isfinite(loss).item()


def test_ppo_optimizer_step_updates_policy_and_critic(ppo_strategy):
    """optimizer_step steps the policy optimizer and then the critic's."""
    strategy, device = ppo_strategy
    batch = _make_batch(device=device)
    loss = strategy.compute_loss(batch)
    loss.backward()

    policy_optimizer = torch.optim.SGD(strategy.model.parameters(), lr=0.1)
    policy_before = next(strategy.model.parameters()).detach().clone()
    critic_before = next(strategy.critic.parameters()).detach().clone()

    strategy.optimizer_step(policy_optimizer)

    assert not torch.equal(next(strategy.model.parameters()), policy_before)
    assert not torch.equal(next(strategy.critic.parameters()), critic_before)
    # Critic gradients are cleared after its step.
    assert all(p.grad is None for p in strategy.critic.parameters())


def test_ppo_optimizer_step_clips_critic_gradients(ppo_strategy):
    strategy, device = ppo_strategy
    batch = _make_batch(device=device)
    loss = strategy.compute_loss(batch)
    loss.backward()
    strategy.critic_optimizer = torch.optim.SGD(strategy.critic.parameters(), lr=1.0)
    strategy.max_grad_norm = 1e-8
    before = next(strategy.critic.parameters()).detach().clone()

    strategy.optimizer_step(torch.optim.SGD(strategy.model.parameters(), lr=0.0))

    # Clipped-to-zero critic gradients under SGD (no momentum) leave the
    # parameters unchanged.
    assert torch.equal(next(strategy.critic.parameters()), before)


# ============== prepare_from_rollout / GAE integration ==============


def _make_rollout_result(B=2, G=2, P=6, R=5, device="cpu"):
    return RolloutResult(
        prompts=torch.randint(3, 200, (B, P), device=device),
        prompt_mask=torch.ones(B, P, dtype=torch.bool, device=device),
        responses=torch.randint(3, 200, (B, G, R), device=device),
        response_mask=torch.ones(B, G, R, dtype=torch.bool, device=device),
        rewards=torch.randn(B, G, device=device),
        logprobs_old=torch.zeros(B, G, R, device=device),
    )


def test_prepare_from_rollout_computes_and_pins_gae(ppo_strategy, monkeypatch):
    """prepare attaches GAE tensors to the result once, then reuses them."""
    strategy, device = ppo_strategy
    result = _make_rollout_result(device=device)

    batch = strategy.prepare_from_rollout(result)
    assert result.advantages is not None and result.returns is not None
    assert batch["advantages"] is result.advantages
    assert batch["returns"] is result.returns
    assert batch["advantages"].shape == result.responses.shape

    calls = []
    original = strategy._compute_advantages
    monkeypatch.setattr(
        strategy,
        "_compute_advantages",
        lambda *a, **k: calls.append(1) or original(*a, **k),
    )
    strategy.prepare_from_rollout(result)
    assert not calls, "pinned advantages must not be recomputed on replay"


def test_prepare_from_rollout_respects_response_padding(ppo_strategy):
    """Padded response positions get zero advantages and returns."""
    strategy, device = ppo_strategy
    result = _make_rollout_result(device=device)
    result.response_mask[0, 0, 3:] = False

    batch = strategy.prepare_from_rollout(result)

    assert torch.count_nonzero(batch["advantages"][0, 0, 3:]) == 0
    assert torch.count_nonzero(batch["returns"][0, 0, 3:]) == 0
    assert torch.count_nonzero(batch["advantages"][0, 0, :3]) > 0


def test_compute_advantages_matches_hand_computed_gae(ppo_strategy, monkeypatch):
    """_compute_advantages applies terminal rewards and GAE faithfully."""
    strategy, device = ppo_strategy
    result = _make_rollout_result(B=1, G=1, P=4, R=3, device=device)
    result.rewards = torch.tensor([[2.0]], device=device)
    result.logprobs_old = torch.zeros(1, 1, 3, device=device)
    # ref_model == policy at init → zero KL reward shaping only if the
    # policy and ref agree on logprobs; keep ref out of the picture here.
    strategy.ref_model = None

    fixed_values = torch.tensor([[[0.1, 0.2, 0.3]]], device=device)
    monkeypatch.setattr(
        strategy_module,
        "rollout_token_values",
        lambda *args, **kwargs: fixed_values.clone(),
    )

    advantages, returns = strategy._compute_advantages(
        result.prompts,
        result.prompt_mask,
        result.responses,
        result.response_mask,
        result.rewards,
        result.logprobs_old,
    )

    rewards = torch.tensor([[[0.0, 0.0, 2.0]]], device=device)
    expected_adv, expected_ret = compute_gae(
        rewards, fixed_values, result.response_mask, 1.0, 0.95
    )
    assert torch.allclose(advantages, expected_adv, atol=1e-6)
    assert torch.allclose(returns, expected_ret, atol=1e-6)


def test_compute_advantages_folds_kl_penalty_into_rewards(ppo_strategy, monkeypatch):
    """With a ref model, each valid token's reward loses kl_coef·k3."""
    strategy, device = ppo_strategy
    result = _make_rollout_result(B=1, G=1, P=4, R=2, device=device)
    result.rewards = torch.tensor([[1.0]], device=device)
    # behaviour policy disagrees with ref by +1 logprob on every token
    result.logprobs_old = torch.ones(1, 1, 2, device=device)

    fixed_values = torch.zeros(1, 1, 2, device=device)
    fixed_ref_logprobs = torch.zeros(1, 1, 2, device=device)
    monkeypatch.setattr(
        strategy_module,
        "rollout_token_values",
        lambda *args, **kwargs: fixed_values.clone(),
    )
    monkeypatch.setattr(
        strategy_module,
        "rollout_token_logprobs",
        lambda *args, **kwargs: {"logprobs": fixed_ref_logprobs.clone()},
    )

    advantages, _ = strategy._compute_advantages(
        result.prompts,
        result.prompt_mask,
        result.responses,
        result.response_mask,
        result.rewards,
        result.logprobs_old,
    )

    # per-token reward = -kl_coef·(1 - 0) = -0.01; terminal adds 1.0
    expected_rewards = torch.tensor([[-0.01, 0.99]], device=device)
    expected_adv, _ = compute_gae(
        expected_rewards.unsqueeze(0), fixed_values, result.response_mask, 1.0, 0.95
    )
    assert torch.allclose(advantages, expected_adv, atol=1e-6)


def test_online_call_returns_finite_loss(ppo_strategy):
    strategy, device = ppo_strategy

    class _RecordingRunner:
        policy_version = 0

        def __call__(self, batch):
            return _make_rollout_result(device=device), True

        def step(self):
            pass

        def apply_weight_update(self, policy_version, update):
            return update()

    strategy.set_rollout_runner(_RecordingRunner())
    out = strategy({"instruction": ["x"]})
    assert torch.isfinite(out["loss"]).item()
    assert "policy_loss" in out["metrics"]
    assert "value_loss" in out["metrics"]
    assert "explained_variance" in out["metrics"]
