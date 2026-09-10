from types import SimpleNamespace

import pytest
import torch

from astrai.model.transformer import AutoRegressiveLM
from astrai.trainer.strategy import BaseStrategy, SEQStrategy
from astrai.trainer.train_callback import MetricCallback
from tests.helpers import make_tiny_config


def test_seq_strategy_combines_and_reports_moe_aux_loss(device):
    config = make_tiny_config(
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        topk_method="greedy",
    )
    model = AutoRegressiveLM(config).to(device=device)
    strategy = SEQStrategy(model, device, moe_aux_loss_coef=0.25)
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (2, 8), device=device),
        "target_ids": torch.randint(0, config.vocab_size, (2, 8), device=device),
    }

    output = strategy(batch)
    legacy_loss = strategy.compute_loss(batch)

    assert isinstance(legacy_loss, torch.Tensor)
    assert set(output["metrics"]) == {
        "loss",
        "task_loss",
        "moe_aux_loss",
        "moe_aux_loss_weighted",
    }
    assert output["loss"].item() == pytest.approx(
        output["metrics"]["task_loss"] + output["metrics"]["moe_aux_loss_weighted"],
    )
    assert output["metrics"]["moe_aux_loss_weighted"] == pytest.approx(
        0.25 * output["metrics"]["moe_aux_loss"],
    )
    assert output["loss"].requires_grad
    assert all(isinstance(metric, float) for metric in output["metrics"].values())


def test_metric_callback_includes_dynamic_strategy_metrics(tmp_path):
    callback = MetricCallback(
        ckpt_dir=tmp_path,
        save_interval=1,
        metrics=["loss", "lr"],
    )
    context = SimpleNamespace(
        metrics={"task_loss": 2.0, "moe_aux_loss": 1.0},
        loss=2.01,
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-3}]),
        val_loss=None,
        grad_norm=None,
        grad_snr_tracker=None,
        world_size=1,
        dp_size=1,
    )

    metrics = callback._metrics(context, callback.metrics)

    assert metrics == {
        "loss": 2.01,
        "lr": 1e-3,
        "task_loss": 2.0,
        "moe_aux_loss": 1.0,
    }


def test_metric_callback_only_computes_requested_metrics(tmp_path):
    def fail_metric(context):
        _ = context
        raise AssertionError("unrequested metric was computed")

    callback = MetricCallback(
        ckpt_dir=tmp_path,
        save_interval=1,
        metrics=["loss"],
    )
    callback._metric_funcs["grad_snr"] = fail_metric
    context = SimpleNamespace(
        metrics={},
        loss=2.0,
        world_size=1,
        dp_size=1,
    )

    metrics = callback._metrics(context, callback.metrics)

    assert metrics == {"loss": 2.0}


def test_legacy_strategy_tensor_loss_is_normalized():
    class LegacyStrategy(BaseStrategy):
        def compute_loss(self, batch):
            return torch.tensor(2.0, requires_grad=True)

    strategy = LegacyStrategy(torch.nn.Linear(1, 1), "cpu")

    output = strategy({})

    assert output["loss"].item() == 2.0
    assert output["metrics"]["loss"] == 2.0
