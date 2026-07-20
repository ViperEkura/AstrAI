"""Unit tests for online rollout integration in :class:`BaseStrategy`.

Covers the shared rollout-trigger logic in ``BaseStrategy.__call__``
(runner injection, cache-driven refresh hook, ``step()`` callback) and
the per-strategy ``prepare_from_rollout`` mappings for both
:class:`GRPOStrategy` and :class:`DPOStrategy`.
"""

import pytest
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.transformer import AutoRegressiveLM
from astrai.trainer.rollout import RolloutResult
from astrai.trainer.strategy import (
    DPOStrategy,
    GRPOStrategy,
    StrategyFactory,
)


class _FakeExecutor:
    """Executor stub tracking ``sync_gradients`` and providing unwrap_model."""

    def __init__(self, sync_gradients=True):
        self._sync_gradients = sync_gradients

    @property
    def sync_gradients(self):
        return self._sync_gradients

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
    cfg = _make_config()
    return AutoRegressiveLM(cfg).to(device=device), cfg


def _make_frozen(model, device):
    cfg = _make_config()
    copy = AutoRegressiveLM(cfg).to(device=device)
    copy.load_state_dict(model.state_dict())
    copy.requires_grad_(False)
    copy.eval()
    return copy


def _make_rollout_result(B=2, G=4, P=6, R=8, device="cpu"):
    return RolloutResult(
        prompts=torch.randint(3, 200, (B, P), device=device),
        responses=torch.randint(3, 200, (B, G, R), device=device),
        response_mask=torch.ones(B, G, R, dtype=torch.bool, device=device),
        rewards=torch.randn(B, G, device=device),
        logprobs_old=torch.zeros(B, G, R, device=device),
    )


class _RecordingRunner:
    """Fake RolloutRunner returning a fixed result with freshness tracking.

    Freshness is ``True`` on the first call after construction or after
    :meth:`swap_result`; ``False`` on subsequent cached calls — mirroring
    the real ``RolloutRunner`` contract without invoking generation.
    """

    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.step_calls = 0
        self._fresh = True

    def __call__(self, batch):
        self.calls += 1
        fresh = self._fresh
        self._fresh = False
        return self.result, fresh

    def step(self):
        self.step_calls += 1

    def swap_result(self, result):
        self.result = result
        self._fresh = True


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_grpo(device, executor=None):
    model, _ = _make_model(device)
    old_model = _make_frozen(model, device)
    ref_model = _make_frozen(model, device)
    return GRPOStrategy(
        model=model,
        device=device,
        old_model=old_model,
        ref_model=ref_model,
        clip_eps=0.2,
        kl_coef=0.01,
        group_size=4,
        model_fn=lambda c=_make_config(): AutoRegressiveLM(c).to(device=device),
        executor=executor or _FakeExecutor(),
    )


def _make_dpo(device, executor=None):
    model, _ = _make_model(device)
    ref_model = _make_frozen(model, device)
    return DPOStrategy(
        model=model,
        device=device,
        ref_model=ref_model,
        beta=0.1,
        reduction="sum",
        model_fn=lambda c=_make_config(): AutoRegressiveLM(c).to(device=device),
        executor=executor or _FakeExecutor(),
    )


def test_factory_registers_online_aliases():
    assert StrategyFactory.is_registered("online_grpo")
    assert StrategyFactory.is_registered("online_dpo")
    assert StrategyFactory._entries["online_grpo"] is GRPOStrategy
    assert StrategyFactory._entries["online_dpo"] is DPOStrategy


def test_grpo_supports_online(device):
    assert _make_grpo(device).supports_online() is True


def test_dpo_supports_online(device):
    assert _make_dpo(device).supports_online() is True


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
    assert batch["responses"] is r.responses
    assert batch["masks"] is r.response_mask
    assert batch["rewards"] is r.rewards


def test_dpo_prepare_from_rollout_picks_best_worst(device):
    strat = _make_dpo(device)
    r = _make_rollout_result(B=3, G=4, R=5, device=device)
    batch = strat.prepare_from_rollout(r)
    assert batch["chosen"].shape == (3, 5)
    assert batch["rejected"].shape == (3, 5)
    assert batch["chosen_mask"].shape == (3, 5)
    assert batch["rejected_mask"].shape == (3, 5)
    idx = torch.arange(3, device=device)
    expected_best = r.responses[idx, r.rewards.argmax(dim=-1)]
    expected_worst = r.responses[idx, r.rewards.argmin(dim=-1)]
    assert torch.equal(batch["chosen"], expected_best)
    assert torch.equal(batch["rejected"], expected_worst)


def test_call_without_runner_falls_back_to_compute_loss_grpo(device):
    strat = _make_grpo(device)
    batch = {
        "prompts": torch.randint(3, 200, (2, 4), device=device),
        "responses": torch.randint(3, 200, (2, 4, 6), device=device),
        "masks": torch.ones(2, 4, 6, device=device),
        "rewards": torch.randn(2, 4, device=device),
    }
    loss = strat(batch)
    assert torch.isfinite(loss).item()


def test_call_with_runner_returns_finite_loss_grpo(device):
    strat = _make_grpo(device)
    strat.set_rollout_runner(_RecordingRunner(_make_rollout_result(device=device)))
    loss = strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    assert torch.isfinite(loss).item()


def test_call_with_runner_returns_finite_loss_dpo(device):
    strat = _make_dpo(device)
    strat.set_rollout_runner(_RecordingRunner(_make_rollout_result(device=device)))
    loss = strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    assert torch.isfinite(loss).item()


def test_call_invokes_runner_each_time(device):
    strat = _make_grpo(device)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    assert runner.calls == 2


def test_grpo_syncs_old_model_on_first_rollout(device):
    strat = _make_grpo(device)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    with torch.no_grad():
        for p in strat.model.parameters():
            p.add_(0.1)
    old_before = {k: v.clone() for k, v in strat.old_model.state_dict().items()}
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    old_after = strat.old_model.state_dict()
    synced = any(
        not torch.allclose(old_before[k], old_after[k])
        for k in old_before
        if k in old_after
    )
    assert synced


def test_grpo_no_resync_when_same_cached_result(device):
    strat = _make_grpo(device)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    assert runner.calls == 2
    assert runner.step_calls == 1


def test_grpo_resync_when_new_rollout_result(device):
    strat = _make_grpo(device)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    runner.swap_result(_make_rollout_result(device=device))
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
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
    runner.swap_result(_make_rollout_result(device=device))
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    assert runner.step_calls == 2


def test_step_not_called_when_sync_gradients_false(device):
    executor = _FakeExecutor(sync_gradients=False)
    strat = _make_grpo(device, executor=executor)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    assert runner.step_calls == 0


def test_step_called_when_sync_gradients_true(device):
    executor = _FakeExecutor(sync_gradients=True)
    strat = _make_grpo(device, executor=executor)
    runner = _RecordingRunner(_make_rollout_result(device=device))
    strat.set_rollout_runner(runner)
    strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    assert runner.step_calls == 1


def test_loss_is_differentiable_grpo(device):
    strat = _make_grpo(device)
    strat.set_rollout_runner(_RecordingRunner(_make_rollout_result(device=device)))
    loss = strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in strat.model.parameters()
    )
    assert has_grad


def test_loss_is_differentiable_dpo(device):
    strat = _make_dpo(device)
    strat.set_rollout_runner(_RecordingRunner(_make_rollout_result(device=device)))
    loss = strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in strat.model.parameters()
    )
    assert has_grad


def test_ref_and_old_model_not_updated_by_backward_grpo(device):
    strat = _make_grpo(device)
    strat.set_rollout_runner(_RecordingRunner(_make_rollout_result(device=device)))
    loss = strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    loss.backward()
    for p in strat.ref_model.parameters():
        assert p.grad is None
    for p in strat.old_model.parameters():
        assert p.grad is None


def test_ref_model_not_updated_by_backward_dpo(device):
    strat = _make_dpo(device)
    strat.set_rollout_runner(_RecordingRunner(_make_rollout_result(device=device)))
    loss = strat({"input_ids": torch.randint(3, 200, (2, 4), device=device)})
    loss.backward()
    for p in strat.ref_model.parameters():
        assert p.grad is None
