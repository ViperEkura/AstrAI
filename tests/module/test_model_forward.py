"""Smoke tests for MoE aux loss and diagnostic metrics integration.

Does NOT load real data or weights.  Uses a tiny randomly-initialized
MoE model and verifies that aux loss computation and MoE routing
diagnostics flow end‑to‑end through the strategy layer.
"""

import pytest
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.components.mlp import DeepSeekMoE
from astrai.model.transformer import AutoRegressiveLM
from astrai.trainer.strategy import (
    SEQStrategy,
    SFTStrategy,
    _collect_moe_diagnostics,
    _load_balancing_loss,
    StrategyFactory,
)
from tests.helpers import TINY_CONFIG

# ── helpers ──────────────────────────────────────────────────────────


def _make_tiny_moe_config(**overrides) -> AutoRegressiveLMConfig:
    return AutoRegressiveLMConfig(
        **{
            **TINY_CONFIG,
            "ffn_type": "moe",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
            "topk_method": "greedy",
            **overrides,
        }
    )


def _make_model(config=None) -> AutoRegressiveLM:
    if config is None:
        config = _make_tiny_moe_config()
    return AutoRegressiveLM(config)


# ── _collect_moe_diagnostics unit tests ─────────────────────────────


def test_collect_moe_diagnostics_returns_all_keys():
    """_collect_moe_diagnostics should return the four expected keys."""
    # Simulate two MoE layers with uniform routing probabilities
    probs = torch.ones(128, 4) / 4.0
    diag = _collect_moe_diagnostics([probs, probs], top_k=2)

    assert set(diag.keys()) == {
        "router_entropy",
        "dead_expert_fraction",
        "load_imbalance_mean",
        "load_imbalance_max",
    }
    for v in diag.values():
        assert isinstance(v, float)


def test_collect_moe_diagnostics_empty_list():
    """Empty list returns empty dict."""
    assert _collect_moe_diagnostics([], top_k=2) == {}


def test_collect_moe_diagnostics_uniform_routing():
    """Uniform routing probabilities with top_k=2 → tie-breaking by index.

    torch.topk breaks ties by index, so with equal probabilities
    experts 0 and 1 always win over experts 2 and 3:
      - dead_expert_fraction = 2/4 = 0.5
      - load_ratios = [2, 2, 0, 0] → |ratio-1| = [1, 1, 1, 1] → mean = 1.0
      - load_imbalance_max = 2.0
    """
    probs = torch.ones(128, 4) / 4.0
    diag = _collect_moe_diagnostics([probs], top_k=2)

    assert diag["dead_expert_fraction"] == pytest.approx(0.5, abs=1e-6)
    assert diag["load_imbalance_mean"] == pytest.approx(1.0, abs=1e-6)
    assert diag["load_imbalance_max"] == pytest.approx(2.0, abs=1e-6)


def test_collect_moe_diagnostics_max_entropy():
    """Uniform probabilities should give log(num_experts) entropy."""
    num_experts = 4
    probs = torch.ones(128, num_experts) / num_experts
    diag = _collect_moe_diagnostics([probs], top_k=2)
    expected_entropy = float(torch.log(torch.tensor(num_experts, dtype=torch.float32)))
    assert diag["router_entropy"] == pytest.approx(expected_entropy, abs=1e-5)


# ── _load_balancing_loss unit tests ──────────────────────────────────


def test_load_balancing_loss_shape_and_range():
    """Verify _load_balancing_loss returns a non-negative scalar tensor."""
    probs = torch.randn(64, 8).softmax(dim=-1)
    loss = _load_balancing_loss(probs)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_load_balancing_loss_uniform_minimum():
    """Uniform routing gives the lowest possible load balancing loss."""
    probs = torch.ones(64, 8) / 8.0
    loss = _load_balancing_loss(probs).item()

    # Very skewed routing should give higher loss
    skewed = torch.zeros(64, 8)
    skewed[:, 0] = 1.0
    skewed[:, 1] = 1.0
    skewed = skewed / skewed.sum(dim=-1, keepdim=True)
    skewed_loss = _load_balancing_loss(skewed).item()

    assert loss < skewed_loss


# ── SEQStrategy integration tests ────────────────────────────────────


class TestSEQStrategyMoE:
    """End‑to‑end tests for SEQStrategy with MoE aux loss."""

    @pytest.fixture(autouse=True)
    def setup(self, device):
        self.device = device
        self.config = _make_tiny_moe_config()
        self.model = _make_model(self.config).to(device)
        self.model.train()

    def _make_batch(self, batch_size=2, seq_len=8):
        vocab = self.config.vocab_size
        input_ids = torch.randint(0, vocab, (batch_size, seq_len))
        # target = input shifted right
        target_ids = torch.randint(0, vocab, (batch_size, seq_len))
        return {"input_ids": input_ids, "target_ids": target_ids}

    def test_compute_loss_returns_scalar(self):
        """compute_loss should return a scalar tensor."""
        strategy = SEQStrategy(
            self.model,
            self.device,
            moe_aux_loss_coef=0.01,
        )
        loss = strategy.compute_loss(self._make_batch())
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_compute_loss_output_has_metrics(self):
        """compute_loss_output dict with moe_aux_loss_coef > 0 includes MoE metrics."""
        strategy = SEQStrategy(
            self.model,
            self.device,
            moe_aux_loss_coef=0.01,
        )
        output = strategy.compute_loss_output(self._make_batch())

        assert "loss" in output
        assert "metrics" in output
        assert output["loss"].ndim == 0
        assert output["loss"].requires_grad

        metrics = output["metrics"]
        # MoE metrics should appear when coef > 0 and model has MoE layers
        for key in ("moe_aux_loss", "moe_aux_loss_weighted", "task_loss", "loss"):
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], float)

    def test_moe_metrics_populated_after_forward(self):
        """strategy._moe_metrics populated after compute_loss_output."""
        strategy = SEQStrategy(
            self.model,
            self.device,
            moe_aux_loss_coef=0.01,
        )
        strategy.compute_loss_output(self._make_batch())

        moe_metrics = strategy._moe_metrics
        assert moe_metrics, "_moe_metrics should not be empty for MoE model"
        for key in (
            "aux_loss",
            "router_entropy",
            "dead_expert_fraction",
            "load_imbalance_mean",
            "load_imbalance_max",
        ):
            assert key in moe_metrics, f"Missing _moe_metrics key: {key}"
            assert isinstance(moe_metrics[key], float)

    def test_zero_coef_zeroes_weighted_aux(self):
        """moe_aux_loss_coef=0 → weighted_aux_loss is zero, task_loss == loss."""
        strategy = SEQStrategy(
            self.model,
            self.device,
            moe_aux_loss_coef=0.0,
        )
        output = strategy.compute_loss_output(self._make_batch())
        metrics = output["metrics"]

        # task_loss and loss should be equal (aux weighted by zero)
        assert "task_loss" in metrics
        assert "loss" in metrics
        assert metrics["loss"] == pytest.approx(metrics["task_loss"], abs=1e-6)

        # weighted aux loss is zero
        assert metrics.get("moe_aux_loss_weighted") == pytest.approx(0.0, abs=1e-6)

        # MoE diagnostics are still collected (monitoring purposes)
        assert strategy._moe_metrics
        assert "router_entropy" in strategy._moe_metrics

    def test_aux_loss_added_to_total_loss(self):
        """Total loss > task_loss when moe_aux_loss_coef > 0."""
        strategy = SEQStrategy(
            self.model,
            self.device,
            moe_aux_loss_coef=0.01,
        )
        output = strategy.compute_loss_output(self._make_batch())
        assert output["metrics"]["loss"] > output["metrics"]["task_loss"] + 1e-12

    def test_factory_creates_strategy_with_coef(self):
        """StrategyFactory.create passes moe_aux_loss_coef to strategy."""
        strategy = StrategyFactory.create(
            "seq",
            model=self.model,
            device=self.device,
            moe_aux_loss_coef=0.02,
        )
        assert strategy.moe_aux_loss_coef == 0.02

    def test_no_aux_loss_for_mlp_model(self):
        """Pure MLP model: model outputs no aux_loss → no MoE metrics."""
        from astrai.config.model_config import AutoRegressiveLMConfig

        mlp_config = AutoRegressiveLMConfig(
            **{**TINY_CONFIG, "ffn_type": "mlp"}
        )
        mlp_model = AutoRegressiveLM(mlp_config).to(self.device)
        mlp_model.train()

        strategy = SEQStrategy(
            mlp_model,
            self.device,
            moe_aux_loss_coef=0.01,
        )
        output = strategy.compute_loss_output(self._make_batch())
        metrics = output["metrics"]

        assert "moe_aux_loss" not in metrics
        assert "moe_aux_loss_weighted" not in metrics
        assert metrics["loss"] == pytest.approx(metrics["task_loss"], abs=1e-6)
        assert strategy._moe_metrics == {}


# ── SFTStrategy integration tests ────────────────────────────────────


class TestSFTStrategyMoE:
    """End‑to‑end tests for SFTStrategy with MoE aux loss."""

    @pytest.fixture(autouse=True)
    def setup(self, device):
        self.device = device
        self.config = _make_tiny_moe_config()
        self.model = _make_model(self.config).to(device)
        self.model.train()

    def _make_batch(self, batch_size=2, seq_len=8):
        vocab = self.config.vocab_size
        input_ids = torch.randint(0, vocab, (batch_size, seq_len))
        target_ids = torch.randint(0, vocab, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def test_compute_loss_output_with_aux_loss(self):
        """SFTStrategy produces MoE metrics when coef > 0."""
        strategy = SFTStrategy(
            self.model,
            self.device,
            moe_aux_loss_coef=0.01,
        )
        output = strategy.compute_loss_output(self._make_batch())

        metrics = output["metrics"]
        assert "moe_aux_loss" in metrics
        assert "moe_aux_loss_weighted" in metrics
        assert metrics["loss"] > metrics["task_loss"] + 1e-12

        moe_metrics = strategy._moe_metrics
        assert "router_entropy" in moe_metrics
        assert "dead_expert_fraction" in moe_metrics

    def test_sft_zero_coef_zeroes_weighted_aux(self):
        """SFTStrategy with zero coef: weighted aux is zero, loss == task_loss."""
        strategy = SFTStrategy(
            self.model,
            self.device,
            moe_aux_loss_coef=0.0,
        )
        output = strategy.compute_loss_output(self._make_batch())
        metrics = output["metrics"]

        assert metrics["loss"] == pytest.approx(metrics["task_loss"], abs=1e-6)
        assert metrics.get("moe_aux_loss_weighted") == pytest.approx(0.0, abs=1e-6)
        # Diagnostics still collected
        assert strategy._moe_metrics
        assert "router_entropy" in strategy._moe_metrics
