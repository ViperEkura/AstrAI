import os

import pytest
import torch
import torch.distributed as dist

from astrai.parallel import (
    DDPExecutor,
    get_rank,
    only_on_rank,
    resolve_local_device_index,
    setup_parallel,
    spawn_parallel_fn,
)


@only_on_rank(0)
def _test_only_on_rank_helper():
    return True


def only_on_rank():
    result = _test_only_on_rank_helper()
    if get_rank() == 0:
        assert result is True
    else:
        assert result is None


def all_reduce():
    x = torch.tensor([get_rank()], dtype=torch.int)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(dist.get_world_size()))
    assert x.item() == expected_sum


def test_spawn_only_on_rank():
    spawn_parallel_fn(only_on_rank, world_size=2, backend="gloo")


def test_spawn_all_reduce():
    spawn_parallel_fn(all_reduce, world_size=2, backend="gloo")


def test_device_order_maps_accelerators_but_not_cpu(monkeypatch):
    monkeypatch.setenv("ASTRAI_DEVICE_ORDER", "2,0,3,1")
    assert resolve_local_device_index(1, 4, "cuda") == 0
    assert resolve_local_device_index(1, 4, "xpu") == 0
    assert resolve_local_device_index(1, 4, "cpu") == 1


def test_device_order_fails_closed(monkeypatch):
    monkeypatch.setenv("ASTRAI_DEVICE_ORDER", "0,0")
    with pytest.raises(ValueError, match="permutation"):
        resolve_local_device_index(0, 2, "cuda")


def test_device_order_uses_local_world_size_and_validates_rank(monkeypatch):
    monkeypatch.setenv("ASTRAI_DEVICE_ORDER", "2,0,3,1")
    assert resolve_local_device_index(3, 4, "cuda") == 1

    with pytest.raises(ValueError, match="outside local world size"):
        resolve_local_device_index(4, 4, "cuda")


def test_setup_parallel_uses_torchrun_local_world_size(monkeypatch):
    state = {"initialized": False}
    captured = {}

    def fake_init_process_group(**kwargs):
        captured.update(kwargs)
        state["initialized"] = True

    def fake_destroy_process_group():
        state["initialized"] = False

    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("ASTRAI_DEVICE_ORDER", "2,0,3,1")
    monkeypatch.setenv("MASTER_ADDR", "previous")
    monkeypatch.setenv("MASTER_PORT", "previous")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_DEVICE", "cpu")
    monkeypatch.setattr(dist, "is_initialized", lambda: state["initialized"])
    monkeypatch.setattr(dist, "init_process_group", fake_init_process_group)
    monkeypatch.setattr(dist, "destroy_process_group", fake_destroy_process_group)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with setup_parallel(rank=4, world_size=8, local_rank=0, device_type="cuda"):
        assert captured["device_id"] == torch.device("cuda", 2)
        assert captured["rank"] == 4
        assert captured["world_size"] == 8
        assert os.environ["LOCAL_RANK"] == "0"
        assert os.environ["LOCAL_DEVICE"] == "cuda:2"


def test_ddp_uses_mapped_local_device_without_changing_logical_rank(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_ddp(model, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_DEVICE", "cuda:2")
    monkeypatch.setattr("astrai.parallel.executor.DDP", fake_ddp)

    wrapped = DDPExecutor()._prepare_model(torch.nn.Linear(1, 1))

    assert wrapped is sentinel
    assert captured["device_ids"] == [2]
    assert captured["output_device"] == 2
    assert os.environ["LOCAL_RANK"] == "0"
