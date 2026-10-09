import torch
import torch.distributed as dist

from astrai.parallel import get_rank, only_on_rank, spawn_parallel_fn


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
