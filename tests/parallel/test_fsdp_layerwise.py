"""FSDP wrapping order controls overlap without changing model topology."""

import pytest
from torch import nn

from astrai.parallel.executor import FSDPExecutor


@pytest.mark.parametrize("layerwise", [True, False])
@pytest.mark.parametrize("reshard", [True, False])
def test_fsdp_wrapping_order_and_reshard_policy(monkeypatch, layerwise, reshard):
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
    executor = FSDPExecutor(layerwise=layerwise, reshard_after_forward=reshard)
    assert executor._prepare_model(model) is model
    expected = [model]
    if layerwise:
        expected = [*model.model.layers, *expected]
    assert [module for module, _ in calls] == expected
    assert all(kwargs["reshard_after_forward"] is reshard for _, kwargs in calls[:-1])
    assert calls[-1][1]["reshard_after_forward"] is False


def test_direct_module_list_keeps_distinct_layer_groups(monkeypatch):
    model = nn.Module()
    first, second = nn.Linear(4, 4), nn.Linear(4, 4)
    model.layers = nn.ModuleList([first, second, first])
    model.norm = nn.LayerNorm(4)
    calls = []
    monkeypatch.setattr(FSDPExecutor, "use_distributed", property(lambda self: True))
    monkeypatch.setattr(
        "astrai.parallel.executor.fsdp.fully_shard",
        lambda module, **kwargs: calls.append(module),
    )
    FSDPExecutor()._prepare_model(model)
    assert calls == [first, second, model]
