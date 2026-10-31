"""Export formats must never change resumable training tensors."""

from dataclasses import replace

import pytest
import torch

from astrai.config import PrecisionConfig
from astrai.serialization import Checkpoint
from astrai.trainer.context import TrainContextBuilder
from astrai.trainer.trainer import Trainer
from tests.integration.trainer.test_unified_resume import _config, _StopAfterOne


@pytest.mark.parametrize("compute", ["float32", "bfloat16"])
def test_export_roundtrip_preserves_training_state(tmp_path, compute):
    policy = PrecisionConfig(compute_dtype=compute, export_dtype="float16")
    baseline_dir, source_dir, resumed_dir = [
        tmp_path / n for n in ("baseline", "source", "resumed")
    ]
    Trainer(_config(baseline_dir, precision=policy)).train()
    Trainer(_config(source_dir, precision=policy), callbacks=[_StopAfterOne()]).train()
    source = source_dir / "epoch_0_step_1"
    saved = Checkpoint.load(str(source))
    assert all(
        t.dtype == torch.float16
        for t in saved.state_dict.values()
        if t.is_floating_point()
    )
    assert all(
        t.dtype == getattr(torch, compute)
        for t in saved.extra["training_model"].values()
        if t.is_floating_point()
    )
    # Changing only export format must preserve the full resume trajectory.
    Trainer(
        _config(resumed_dir, precision=replace(policy, export_dtype="float32"))
    ).train(param_path=str(source), resume=True)
    baseline = Checkpoint.load(str(baseline_dir / "epoch_0_step_4"))
    resumed = Checkpoint.load(str(resumed_dir / "epoch_0_step_4"))
    for name, value in baseline.extra["training_model"].items():
        torch.testing.assert_close(
            value, resumed.extra["training_model"][name], atol=0, rtol=0
        )
    if compute == "bfloat16":
        for a, b in zip(
            baseline.extra["optimizer"]["master_params"],
            resumed.extra["optimizer"]["master_params"],
        ):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize(
    "name", ["compute_dtype", "gradient_dtype", "master_dtype", "optimizer_state_dtype"]
)
def test_resume_rejects_training_precision_change(tmp_path, name):
    policy = PrecisionConfig(compute_dtype="float32")
    source_dir = tmp_path / "source"
    Trainer(_config(source_dir, precision=policy), callbacks=[_StopAfterOne()]).train()
    changed = replace(policy, **{name: "bfloat16"})
    with pytest.raises(ValueError, match="precision." + name):
        TrainContextBuilder(
            _config(tmp_path / "resumed", precision=changed)
        ).with_param_path(str(source_dir / "epoch_0_step_1"), resume=True).build()
