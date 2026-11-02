from types import SimpleNamespace

import pytest
import torch

from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.trainer.callbacks import MetricCallback
from astrai.trainer.strategy import SEQStrategy
from astrai.trainer.trainer import Trainer
from tests.support.models import make_tiny_config


def test_seq_strategy_combines_and_reports_moe_aux_loss(cpu_device):
    config = make_tiny_config(
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        topk_method="greedy",
    )
    model = AutoRegressiveLM(config).to(device=cpu_device)
    strategy = SEQStrategy(model, cpu_device, moe_aux_loss_coef=0.25)
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (2, 8), device=cpu_device),
        "target_ids": torch.randint(0, config.vocab_size, (2, 8), device=cpu_device),
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


def test_global_window_combines_losses_with_shared_normalizer():
    strategy = SimpleNamespace(
        loss_outputs=lambda batch: iter(
            {"loss": torch.tensor(loss), "metrics": {"task": loss}, "normalizer": 4}
            for loss in (2.0, 3.0)
        )
    )
    output = Trainer._window_loss_output(strategy, "batch", 4)
    assert output["loss"].item() == 5.0
    assert output["metrics"] == {"task": 5.0}
    assert output["normalizer"] == 4.0


@pytest.mark.parametrize("counts", [[], [2, 3]])
def test_global_window_rejects_empty_or_inconsistent_losses(counts):
    strategy = SimpleNamespace(
        loss_outputs=lambda batch: iter(
            {"loss": torch.tensor(1.0), "metrics": {}, "normalizer": count}
            for count in counts
        )
    )
    with pytest.raises(RuntimeError, match="has no loss|share a normalizer"):
        Trainer._window_loss_output(strategy, "batch", 2)
