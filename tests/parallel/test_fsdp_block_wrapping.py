"""FSDP decoder-block wrapping plan and the two-GPU update and checkpoint oracle."""

from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule

from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel.executor import FSDPExecutor
from astrai.parallel.executor import fsdp as executor_module
from astrai.serialization import Checkpoint
from tests.support.models import make_rollout_config


@pytest.mark.parametrize("layerwise", [True, False])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("reshard,root_reshard", [(True, False), (False, True)])
def test_decoder_blocks_then_root_own_every_parameter_once(
    monkeypatch, tied, reshard, root_reshard, layerwise
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

    monkeypatch.setattr(FSDPExecutor, "use_distributed", property(lambda self: True))
    monkeypatch.setattr(executor_module, "fully_shard", record)
    executor = FSDPExecutor(
        mesh=mesh,
        mp_policy=policy,
        layerwise=layerwise,
        reshard_after_forward=reshard,
        root_reshard_after_forward=root_reshard,
    )
    assert executor._prepare_model(model) is model
    blocks = list(model.model.layers) if layerwise else []
    assert [call[0] for call in calls] == [*blocks, model]
    assert owned == parameters
    assert sum(len(call[1]) for call in calls) == len(parameters)
    root_parameters = {
        id(param)
        for module in (model.model.embed_tokens, model.model.norm, model.lm_head)
        for param in module.parameters()
    }
    assert calls[-1][1] == (root_parameters if layerwise else parameters)
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
    monkeypatch.setattr(FSDPExecutor, "use_distributed", property(lambda self: True))
    monkeypatch.setattr(
        executor_module, "fully_shard", lambda module, **kwargs: calls.append(module)
    )
    FSDPExecutor()._prepare_model(model)
    assert calls == [model.model.layers[0], model.model.layers[1], model]


def test_cross_block_parameter_alias_fails_before_any_sharding(monkeypatch):
    model = AutoRegressiveLM(make_rollout_config(num_hidden_layers=2))
    model.model.layers[1].input_norm.weight = model.model.layers[0].input_norm.weight
    monkeypatch.setattr(FSDPExecutor, "use_distributed", property(lambda self: True))
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
    monkeypatch.setattr(FSDPExecutor, "use_distributed", property(lambda self: True))
    monkeypatch.setattr(
        executor_module, "fully_shard", lambda module, **kwargs: calls.append(module)
    )
    FSDPExecutor()._prepare_model(model)
    assert calls == [model]


def _worker(rank, rendezvous, output, tied):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(3407)
        config = make_rollout_config(
            vocab_size=32, num_hidden_layers=2, tie_word_embeddings=tied
        )
        full = AutoRegressiveLM(config).to(device)
        expected = deepcopy(full)
        executor = FSDPExecutor(mesh=init_device_mesh("cuda", (2,)))
        wrapped, optimizer, _ = executor.prepare(
            lambda: full, lambda model: torch.optim.AdamW(model.parameters(), lr=0.001)
        )
        assert sum(isinstance(module, FSDPModule) for module in wrapped.modules()) == 3
        assert (wrapped.lm_head.weight is wrapped.model.embed_tokens.weight) == tied
        wanted_optimizer = torch.optim.AdamW(expected.parameters(), lr=0.001)
        ids = torch.arange(24, device=device).reshape(4, 6) % 32
        labels = (ids + 1) % 32
        for _ in range(2):
            wanted = F.cross_entropy(
                expected(ids)["logits"].reshape(-1, 32), labels.reshape(-1)
            )
            local_ids, local_labels = ids.chunk(2)[rank], labels.chunk(2)[rank]
            got = F.cross_entropy(
                wrapped(local_ids)["logits"].reshape(-1, 32), local_labels.reshape(-1)
            )
            wanted.backward()
            got.backward()
            wanted_optimizer.step()
            optimizer.step()
            wanted_optimizer.zero_grad()
            optimizer.zero_grad()
        with executor.checkpoint_context(
            wrapped, optimizer=optimizer, export_dtype=None
        ) as (state, _):
            if rank == 0:
                for key, value in expected.state_dict().items():
                    torch.testing.assert_close(state[key], value, rtol=3e-4, atol=3e-5)
                Checkpoint(state_dict=state, config=config.to_dict()).save(output)
        dist.barrier()
        checkpoint = Checkpoint.load_any(output)
        restored = AutoRegressiveLM(config).to(device)
        restored.load_state_dict(checkpoint.state_dict)
        torch.testing.assert_close(
            restored(ids)["logits"], expected(ids)["logits"], rtol=3e-4, atol=3e-5
        )
        assert set(checkpoint.state_dict) == set(expected.state_dict())
    finally:
        dist.destroy_process_group()


@pytest.mark.cuda
@pytest.mark.distributed
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two assigned CUDA test GPUs"
)
@pytest.mark.parametrize("tied", [False, True])
def test_fsdp_blocks_two_gpu_update_and_checkpoint(tmp_path, tied):
    mp.spawn(
        _worker,
        args=(
            "file://" + str(tmp_path / "rendezvous"),
            str(tmp_path / "weights"),
            tied,
        ),
        nprocs=2,
        join=True,
    )
