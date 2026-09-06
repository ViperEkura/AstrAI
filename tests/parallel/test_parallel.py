import pytest
import torch
import torch.distributed as dist

from astrai.parallel import get_rank, only_on_rank, spawn_parallel_fn
from astrai.parallel.setup import resolve_launch_world_size


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


@pytest.fixture
def clean_launch_environment(monkeypatch):
    for name in ("LOCAL_WORLD_SIZE", "RANK", "WORLD_SIZE", "TORCHELASTIC_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)


@pytest.mark.parametrize("nprocs", [1, 2, 4])
def test_resolve_launch_world_size_keeps_cli_value_for_local_launch(
    monkeypatch, clean_launch_environment, nprocs
):
    monkeypatch.setenv("WORLD_SIZE", "8")

    assert resolve_launch_world_size(nprocs) == nprocs


def test_resolve_launch_world_size_uses_torchrun_world_size(
    monkeypatch, clean_launch_environment
):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert resolve_launch_world_size(1) == 4


@pytest.mark.parametrize("world_size", ["not-an-integer", "", "0", "-1", "1.5"])
def test_resolve_launch_world_size_rejects_invalid_torchrun_world_size(
    monkeypatch, clean_launch_environment, world_size
):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", world_size)

    with pytest.raises(ValueError, match="WORLD_SIZE"):
        resolve_launch_world_size(1)


@pytest.mark.parametrize("marker", ["LOCAL_WORLD_SIZE", "TORCHELASTIC_RUN_ID"])
def test_resolve_launch_world_size_rejects_missing_external_world_size(
    monkeypatch, clean_launch_environment, marker
):
    monkeypatch.setenv(marker, "1")

    with pytest.raises(ValueError, match="WORLD_SIZE"):
        resolve_launch_world_size(2)


@pytest.mark.parametrize("marker", ["LOCAL_WORLD_SIZE", "TORCHELASTIC_RUN_ID"])
def test_resolve_launch_world_size_uses_existing_external_detection(
    monkeypatch, clean_launch_environment, marker
):
    monkeypatch.setenv(marker, "1")
    monkeypatch.setenv("WORLD_SIZE", "8")

    assert resolve_launch_world_size(2) == 8
