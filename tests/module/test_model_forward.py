"""Smoke tests for MoE aux loss and diagnostic metrics integration.

Does NOT load real data or weights.  Uses a tiny randomly-initialized
MoE model and verifies that aux loss computation and MoE routing
diagnostics flow end‑to‑end through the strategy layer.
"""

import pytest
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.transformer import AutoRegressiveLM
from astrai.trainer.strategy import (
    SEQStrategy,
    SFTStrategy,
    StrategyFactory,
    _collect_moe_diagnostics,
)
from tests.helpers import TINY_CONFIG


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


def _make_batch(config, batch_size=2, seq_len=8, with_extra=False):
    """Build a random token batch, optionally with position ids and loss mask."""
    vocab = config.vocab_size
    batch = {
        "input_ids": torch.randint(0, vocab, (batch_size, seq_len)),
        "target_ids": torch.randint(0, vocab, (batch_size, seq_len)),
    }
    if with_extra:
        batch["position_ids"] = (
            torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        )
        batch["loss_mask"] = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return batch


def _make_seq_moe_fixture(device):
    config = _make_tiny_moe_config()
    model = _make_model(config).to(device)
    model.train()
    return config, model


def _make_sft_moe_fixture(device):
    config = _make_tiny_moe_config()
    model = _make_model(config).to(device)
    model.train()
    return config, model


def test_model_forward_contract_uses_dense_training_and_packed_inference():
    from astrai.inference.cache import PagePool, TaskCacheManager
    from astrai.inference.workspace import InferenceWorkspace

    config = AutoRegressiveLMConfig(**TINY_CONFIG)
    model = AutoRegressiveLM(config).eval()
    dense = model(torch.tensor([[1, 2, 3]]))
    assert dense["logits"].shape == (1, 3, config.vocab_size)

    pool = PagePool(
        n_layers=config.num_hidden_layers,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_batch_size=1,
        max_seq_len=config.max_position_embeddings,
        device="cpu",
        dtype=torch.float32,
    )
    cache = TaskCacheManager(pool)
    workspace = InferenceWorkspace(
        1,
        config.max_position_embeddings,
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads,
        torch.device("cpu"),
        torch.float32,
    )
    assert cache.task_alloc("t", [1, 2, 3])
    packed = model(
        torch.tensor([1, 2, 3]),
        position_ids=torch.arange(3),
        kv_cache=cache.bind(["t"], workspace, start_pos=0),
        fwd="prefill",
    )
    assert packed["logits"].shape == (3, config.vocab_size)

    with pytest.raises(ValueError, match="training input_ids"):
        model(torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError, match="inference input_ids"):
        model(
            torch.tensor([[1, 2, 3]]),
            kv_cache=cache.bind(["t"], workspace, start_pos=0),
            fwd="prefill",
        )


def test_forward_logits_positions_projects_only_requested_rows():
    """logits_positions gathers packed rows before the lm_head projection."""
    from astrai.inference.cache import PagePool, TaskCacheManager
    from astrai.inference.workspace import InferenceWorkspace

    config = AutoRegressiveLMConfig(**TINY_CONFIG)
    model = AutoRegressiveLM(config).eval()
    prompts = [[1, 2, 3], [4, 5]]
    last_rows = torch.tensor([len(prompts[0]) - 1, len(prompts) - 1 + len(prompts[1])])

    pool = PagePool(
        n_layers=config.num_hidden_layers,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_batch_size=2,
        max_seq_len=config.max_position_embeddings,
        device="cpu",
        dtype=torch.float32,
    )
    cache = TaskCacheManager(pool)
    workspace = InferenceWorkspace(
        2,
        config.max_position_embeddings,
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads,
        torch.device("cpu"),
        torch.float32,
    )
    for tid, ids in zip(("t1", "t2"), prompts):
        assert cache.task_alloc(tid, ids)
    input_ids = torch.tensor(sum(prompts, []), dtype=torch.long)
    position_ids = torch.cat([torch.arange(len(p)) for p in prompts])
    with torch.inference_mode():
        kwargs = dict(
            position_ids=position_ids,
            kv_cache=cache.bind(["t1", "t2"], workspace, start_pos=0),
            fwd="prefill",
        )
        full = model(input_ids, **kwargs)
        sliced = model(input_ids, logits_positions=last_rows, **kwargs)

    assert full["logits"].shape == (5, config.vocab_size)
    assert sliced["logits"].shape == (2, config.vocab_size)
    # Projecting M=2 rows vs M=5 rows may pick different GEMM kernels and
    # differ in the last float32 bits — assert_close, not bit equality.
    torch.testing.assert_close(sliced["logits"], full["logits"][last_rows])
    assert torch.equal(sliced["hidden_states"], full["hidden_states"][last_rows])


def _router_stats(probs, topk_indices):
    return {"probs": probs, "topk_indices": topk_indices}


def test_collect_moe_diagnostics_returns_all_keys():
    """_collect_moe_diagnostics should return the four expected keys."""
    # Simulate two MoE layers with uniform routing probabilities
    probs = torch.ones(128, 4) / 4.0
    topk = torch.zeros(128, 2, dtype=torch.long)
    diag = _collect_moe_diagnostics([_router_stats(probs, topk)] * 2)

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
    assert _collect_moe_diagnostics([]) == {}


def test_collect_moe_diagnostics_uniform_routing():
    """Uniform routing with top_k=2 → tie-breaking by index.

    torch.topk breaks ties by index, so with equal probabilities
    experts 0 and 1 always win over experts 2 and 3:
      - dead_expert_fraction = 2/4 = 0.5
      - load_ratios = [2, 2, 0, 0] → |ratio-1| = [1, 1, 1, 1] → mean = 1.0
      - load_imbalance_max = 2.0
    """
    probs = torch.ones(128, 4) / 4.0
    topk = torch.tensor([[0, 1]] * 128)
    diag = _collect_moe_diagnostics([_router_stats(probs, topk)])

    assert diag["dead_expert_fraction"] == pytest.approx(0.5, abs=1e-6)
    assert diag["load_imbalance_mean"] == pytest.approx(1.0, abs=1e-6)
    assert diag["load_imbalance_max"] == pytest.approx(2.0, abs=1e-6)


def test_collect_moe_diagnostics_max_entropy():
    """Uniform probabilities should give log(num_experts) entropy."""
    num_experts = 4
    probs = torch.ones(128, num_experts) / num_experts
    topk = torch.zeros(128, 2, dtype=torch.long)
    diag = _collect_moe_diagnostics([_router_stats(probs, topk)])
    expected_entropy = float(torch.log(torch.tensor(num_experts, dtype=torch.float32)))
    assert diag["router_entropy"] == pytest.approx(expected_entropy, abs=1e-5)


def test_moe_metrics_flow_through_wrapped_model(device):
    """DDP-like wrappers (no .config / get_moe_router_probs) still collect MoE metrics."""
    import torch.nn as nn

    from astrai.trainer.strategy import SEQStrategy

    class ForwardOnlyWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.module = model

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    config = _make_tiny_moe_config()
    model = AutoRegressiveLM(config).to(device)
    wrapped = ForwardOnlyWrapper(model)
    wrapped.train()

    strategy = SEQStrategy(wrapped, device, moe_aux_loss_coef=0.01)
    output = strategy.compute_loss_output(
        {
            "input_ids": torch.randint(0, config.vocab_size, (2, 8)),
            "target_ids": torch.randint(0, config.vocab_size, (2, 8)),
        }
    )

    assert "moe_aux_loss" in output["metrics"]
    assert "router_entropy" in strategy._moe_metrics


def test_seq_compute_loss_returns_scalar(device):
    """compute_loss should return a scalar tensor."""
    config, model = _make_seq_moe_fixture(device)
    strategy = SEQStrategy(
        model,
        device,
        moe_aux_loss_coef=0.01,
    )
    loss = strategy.compute_loss(_make_batch(config))
    assert loss.ndim == 0
    assert loss.requires_grad


def test_seq_compute_loss_output_has_metrics(device):
    """compute_loss_output dict with moe_aux_loss_coef > 0 includes MoE metrics."""
    config, model = _make_seq_moe_fixture(device)
    strategy = SEQStrategy(
        model,
        device,
        moe_aux_loss_coef=0.01,
    )
    output = strategy.compute_loss_output(_make_batch(config))

    assert "loss" in output
    assert "metrics" in output
    assert output["loss"].ndim == 0
    assert output["loss"].requires_grad

    metrics = output["metrics"]
    # MoE metrics should appear when coef > 0 and model has MoE layers
    for key in ("moe_aux_loss", "moe_aux_loss_weighted", "task_loss", "loss"):
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics[key], float)


def test_seq_moe_metrics_populated_after_forward(device):
    """strategy._moe_metrics populated after compute_loss_output."""
    config, model = _make_seq_moe_fixture(device)
    strategy = SEQStrategy(
        model,
        device,
        moe_aux_loss_coef=0.01,
    )
    strategy.compute_loss_output(_make_batch(config))

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


def test_seq_zero_coef_zeroes_weighted_aux(device):
    """moe_aux_loss_coef=0 → weighted_aux_loss is zero, task_loss == loss."""
    config, model = _make_seq_moe_fixture(device)
    strategy = SEQStrategy(
        model,
        device,
        moe_aux_loss_coef=0.0,
    )
    output = strategy.compute_loss_output(_make_batch(config))
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


def test_seq_aux_loss_added_to_total_loss(device):
    """Total loss > task_loss when moe_aux_loss_coef > 0."""
    config, model = _make_seq_moe_fixture(device)
    strategy = SEQStrategy(
        model,
        device,
        moe_aux_loss_coef=0.01,
    )
    output = strategy.compute_loss_output(_make_batch(config))
    assert output["metrics"]["loss"] > output["metrics"]["task_loss"] + 1e-12


def test_seq_factory_creates_strategy_with_coef(device):
    """StrategyFactory.create passes moe_aux_loss_coef to strategy."""
    _, model = _make_seq_moe_fixture(device)
    strategy = StrategyFactory.create(
        "seq",
        model=model,
        device=device,
        moe_aux_loss_coef=0.02,
    )
    assert strategy.moe_aux_loss_coef == 0.02


def test_seq_no_aux_loss_for_mlp_model(device):
    """Pure MLP model: model outputs no aux_loss → no MoE metrics."""
    config, _ = _make_seq_moe_fixture(device)
    mlp_config = AutoRegressiveLMConfig(**{**TINY_CONFIG, "ffn_type": "mlp"})
    mlp_model = AutoRegressiveLM(mlp_config).to(device)
    mlp_model.train()

    strategy = SEQStrategy(
        mlp_model,
        device,
        moe_aux_loss_coef=0.01,
    )
    output = strategy.compute_loss_output(_make_batch(config))
    metrics = output["metrics"]

    assert "moe_aux_loss" not in metrics
    assert "moe_aux_loss_weighted" not in metrics
    assert metrics["loss"] == pytest.approx(metrics["task_loss"], abs=1e-6)
    assert strategy._moe_metrics == {}


def test_sft_compute_loss_output_with_aux_loss(device):
    """SFTStrategy produces MoE metrics when coef > 0."""
    config, model = _make_sft_moe_fixture(device)
    strategy = SFTStrategy(
        model,
        device,
        moe_aux_loss_coef=0.01,
    )
    output = strategy.compute_loss_output(_make_batch(config, with_extra=True))

    metrics = output["metrics"]
    assert "moe_aux_loss" in metrics
    assert "moe_aux_loss_weighted" in metrics
    assert metrics["loss"] > metrics["task_loss"] + 1e-12

    moe_metrics = strategy._moe_metrics
    assert "router_entropy" in moe_metrics
    assert "dead_expert_fraction" in moe_metrics


def test_sft_zero_coef_zeroes_weighted_aux(device):
    """SFTStrategy with zero coef: weighted aux is zero, loss == task_loss."""
    config, model = _make_sft_moe_fixture(device)
    strategy = SFTStrategy(
        model,
        device,
        moe_aux_loss_coef=0.0,
    )
    output = strategy.compute_loss_output(_make_batch(config, with_extra=True))
    metrics = output["metrics"]

    assert metrics["loss"] == pytest.approx(metrics["task_loss"], abs=1e-6)
    assert metrics.get("moe_aux_loss_weighted") == pytest.approx(0.0, abs=1e-6)
    # Diagnostics still collected
    assert strategy._moe_metrics
    assert "router_entropy" in strategy._moe_metrics
