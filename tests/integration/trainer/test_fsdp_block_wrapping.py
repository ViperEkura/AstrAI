"""Wrapping-plan checks; real collectives live in the GPU qualification."""

import pytest
from torch import nn
from torch.distributed.fsdp import FSDPModule

from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel import executor as executor_module
from astrai.parallel.executor import FSDPExecutor
from tests.support.models import make_rollout_config


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("reshard,root_reshard", [(True, False), (False, True)])
def test_decoder_blocks_then_root_own_every_parameter_once(
    monkeypatch, tied, reshard, root_reshard
):
    model = AutoRegressiveLM(
        make_rollout_config(num_hidden_layers=3, tie_word_embeddings=tied)
    )
    parameters = {id(param) for param in model.parameters()}
    mesh, policy = object(), object()
    calls, owned = [], set()

    def record(module, **kwargs):
        new = {id(param) for param in module.parameters()} - owned
        calls.append((module, new, kwargs))
        owned.update(new)

    monkeypatch.setattr(executor_module, "get_world_size", lambda: 2)
    monkeypatch.setattr(executor_module, "fully_shard", record)
    executor = FSDPExecutor(
        mesh=mesh,
        mp_policy=policy,
        reshard_after_forward=reshard,
        root_reshard_after_forward=root_reshard,
    )
    assert executor._prepare_model(model) is model
    assert [call[0] for call in calls] == [*model.model.layers, model]
    assert owned == parameters
    assert sum(len(call[1]) for call in calls) == len(parameters)
    assert calls[-1][1] == {
        id(param)
        for module in (model.model.embed_tokens, model.model.norm, model.lm_head)
        for param in module.parameters()
    }
    assert all(
        call[2] == dict(mesh=mesh, mp_policy=policy, reshard_after_forward=reshard)
        for call in calls[:-1]
    )
    assert calls[-1][2]["reshard_after_forward"] == root_reshard
    assert (model.lm_head.weight is model.model.embed_tokens.weight) == tied


def test_repeated_loop_block_wraps_once(monkeypatch):
    model = AutoRegressiveLM(make_rollout_config(num_hidden_layers=3))
    model.model.layers[2] = model.model.layers[0]
    calls = []
    monkeypatch.setattr(executor_module, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        executor_module, "fully_shard", lambda module, **kwargs: calls.append(module)
    )
    FSDPExecutor()._prepare_model(model)
    assert calls == [model.model.layers[0], model.model.layers[1], model]


def test_cross_block_parameter_alias_fails_before_any_sharding(monkeypatch):
    model = AutoRegressiveLM(make_rollout_config(num_hidden_layers=2))
    model.model.layers[1].input_norm.weight = model.model.layers[0].input_norm.weight
    monkeypatch.setattr(executor_module, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        executor_module,
        "fully_shard",
        lambda *args, **kwargs: pytest.fail("must validate before mutating modules"),
    )
    with pytest.raises(ValueError, match="shared across distinct"):
        FSDPExecutor()._prepare_model(model)


def test_root_supports_fsdp_dynamic_class_layout():
    model = AutoRegressiveLM(make_rollout_config())
    original = type(model)
    model.__class__ = type("FSDPLayoutProbe", (FSDPModule, original), {})
    assert isinstance(model, FSDPModule) and isinstance(model, AutoRegressiveLM)


def test_generic_model_uses_one_root_group(monkeypatch):
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 3))
    calls = []
    monkeypatch.setattr(executor_module, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        executor_module, "fully_shard", lambda module, **kwargs: calls.append(module)
    )
    FSDPExecutor()._prepare_model(model)
    assert calls == [model]
