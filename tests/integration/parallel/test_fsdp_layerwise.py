"""FSDP wrapping order controls overlap without changing model topology."""

import pytest
from torch import nn

from astrai.parallel.executor import FSDPExecutor


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.lm_head = nn.Linear(4, 4)


@pytest.mark.parametrize("reshard", [True, False])
def test_fsdp_layerwise_wraps_blocks_before_parent(monkeypatch, reshard):
    calls = []
    monkeypatch.setattr(FSDPExecutor, "use_distributed", property(lambda self: True))
    monkeypatch.setattr(
        "astrai.parallel.executor.fsdp.fully_shard",
        lambda module, **kwargs: calls.append((module, kwargs)),
    )
    model = _Model()
    executor = FSDPExecutor(reshard_after_forward=reshard)
    assert executor._prepare_model(model) is model
    assert [module for module, _ in calls] == [
        *model.model.layers,
        model,
    ]
    assert all(kwargs["reshard_after_forward"] is reshard for _, kwargs in calls[:-1])
    assert calls[-1][1]["reshard_after_forward"] is False


def test_fsdp_coarse_wrap_remains_available(monkeypatch):
    calls = []
    monkeypatch.setattr(FSDPExecutor, "use_distributed", property(lambda self: True))
    monkeypatch.setattr(
        "astrai.parallel.executor.fsdp.fully_shard",
        lambda module, **kwargs: calls.append(module),
    )
    model = _Model()
    executor = FSDPExecutor(layerwise=False)
    assert executor._prepare_model(model) is model
    assert calls == [model]


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
