"""FSDP wrapping order controls overlap without changing model topology."""

import pytest
from torch import nn

from astrai.parallel.executor import FSDPExecutor


@pytest.mark.parametrize("layerwise", [True, False])
def test_fsdp_wrapping_order_and_reshard_policy(monkeypatch, layerwise):
    calls = []
    monkeypatch.setattr(FSDPExecutor, "use_distributed", property(lambda self: True))
    monkeypatch.setattr(
        "astrai.parallel.executor.fsdp.fully_shard",
        lambda module, **kwargs: calls.append((module, kwargs)),
    )
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    model.model.norm = nn.LayerNorm(4)
    model.lm_head = nn.Linear(4, 4)
    executor = FSDPExecutor(layerwise=layerwise)
    assert executor._prepare_model(model) is model
    expected = [model.model, model.lm_head]
    if layerwise:
        expected = [*model.model.layers, *expected]
    assert [module for module, _ in calls] == expected
    assert all(kwargs["reshard_after_forward"] is False for _, kwargs in calls)
