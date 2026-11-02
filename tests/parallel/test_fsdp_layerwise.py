"""FSDP wrapping order controls overlap without changing model topology."""

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


def test_fsdp_layerwise_wraps_blocks_before_parent(monkeypatch):
    calls = []
    monkeypatch.setattr(FSDPExecutor, "use_distributed", property(lambda self: True))
    monkeypatch.setattr(
        "astrai.parallel.executor.fsdp.fully_shard",
        lambda module, **kwargs: calls.append((module, kwargs)),
    )
    model = _Model()
    executor = FSDPExecutor()
    assert executor._prepare_model(model) is model
    assert [module for module, _ in calls] == [
        *model.model.layers,
        model.model,
        model.lm_head,
    ]
    assert all(kwargs["reshard_after_forward"] is False for _, kwargs in calls)


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
    assert calls == [model.model, model.lm_head]
