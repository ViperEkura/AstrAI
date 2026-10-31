import json
from dataclasses import asdict

import pytest
import torch
from torch import nn

from astrai.config import ComponentSpec, PrecisionConfig, TrainConfig
from astrai.optimization.optimizer.precision import PrecisionOptimizer
from astrai.serialization.precision import export_model_state


def _config(**kwargs):
    return TrainConfig(
        strategy="sft",
        model=ComponentSpec(
            type="torch.nn:Linear", params={"in_features": 1, "out_features": 1}
        ),
        dataset=ComponentSpec(type="packed"),
        optimizer=ComponentSpec(type="sgd", params={"lr": 0.01}),
        scheduler=ComponentSpec(type="constant"),
        **kwargs,
    )


def test_precision_roundtrip():
    policy = PrecisionConfig(
        compute_dtype="bfloat16",
        gradient_dtype="bfloat16",
        master_dtype="bfloat16",
        optimizer_state_dtype="float32",
        export_dtype="float16",
    )
    config = _config(precision=asdict(policy))
    payload = json.loads(json.dumps(config.to_dict()))
    assert payload["precision"] == asdict(policy)
    assert "model_dtype" not in payload
    assert TrainConfig(**payload).precision == policy


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_dtype": "float32"},
        {"executor_kwargs": {"grad_accum_dtype": "float32"}},
    ],
)
def test_reject_old_precision_configuration(kwargs):
    with pytest.raises(ValueError):
        _config(**kwargs)


@pytest.mark.parametrize(
    "settings",
    [
        {"compute_dtype": "float64"},
        {"master_dtype": None},
        {"optimizer_state_dtype": "int8"},
        {"gradient_dtype": "auto"},
    ],
)
def test_reject_invalid_precision(settings):
    with pytest.raises(ValueError):
        PrecisionConfig(**settings)


def test_export_uses_master_without_mutating_training_state():
    model = nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1)
    optimizer = PrecisionOptimizer(torch.optim.SGD(model.parameters(), lr=1e-5))
    for _ in range(20):
        model.weight.main_grad = torch.ones_like(model.weight, dtype=torch.float32)
        optimizer.step()
        optimizer.zero_grad()
    master = optimizer.master_parameters[0][1]
    assert model.weight.item() == 1
    assert master.item() < 1
    state = model.state_dict()
    fp32 = export_model_state(state, "float32", optimizer.master_parameters)
    bf16 = export_model_state(state, "bfloat16", optimizer.master_parameters)
    torch.testing.assert_close(fp32["weight"], master, atol=0, rtol=0)
    assert bf16["weight"].item() == 1
    assert master.dtype == torch.float32 and model.weight.dtype == torch.bfloat16
    assert export_model_state(state, None, optimizer.master_parameters) is state
