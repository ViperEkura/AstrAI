import warnings
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from astrai.parallel import get_rank, only_on_rank, spawn_parallel_fn
from astrai.parallel import setup as parallel_setup
from astrai.parallel.executor import (
    DDPExecutor,
    ExecutorFactory,
    FSDPExecutor,
    NoneExecutor,
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


def test_setup_parallel_applies_process_group_timeout(monkeypatch):
    captured = {}

    class FakeDist:
        group = SimpleNamespace(WORLD=None)

        def is_available(self):
            return True

        def is_initialized(self):
            return False

        def init_process_group(self, **kwargs):
            captured.update(kwargs)

        def destroy_process_group(self):
            pass

    monkeypatch.setattr(parallel_setup, "dist", FakeDist())
    # setup_parallel writes launcher variables directly; isolate this fake run.
    monkeypatch.setattr(parallel_setup.os, "environ", parallel_setup.os.environ.copy())
    with parallel_setup.setup_parallel(
        rank=0,
        world_size=2,
        local_rank=0,
        backend="gloo",
        device_type="cpu",
        process_group_timeout=120,
    ):
        pass

    assert captured["timeout"] == timedelta(seconds=120)


def test_every_advertised_executor_is_registered():
    """The factory must resolve every name the docs advertise.

    A registry entry is wired by an import-time decorator, so a split or a
    moved class can silently drop it; this CPU test catches that instead of the
    GPU-only DDP/FSDP suites.
    """

    assert ExecutorFactory.get_component_class("none") is NoneExecutor
    assert ExecutorFactory.get_component_class("ddp") is DDPExecutor
    assert ExecutorFactory.get_component_class("fsdp") is FSDPExecutor


def _ddp_buffer_sync_initialization_and_forward():
    rank = get_rank()
    for init_sync in (True, False):
        for forward_sync_buffers in (True, False):
            model = torch.nn.Linear(2, 2, bias=False)
            model.weight.data.fill_(1 + rank)
            model.register_buffer("counter", torch.tensor(float(rank)))
            executor = DDPExecutor(
                init_sync=init_sync, forward_sync_buffers=forward_sync_buffers
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                wrapped, _, _ = executor.prepare(lambda: model)

            assert model.counter.item() == (0 if init_sync else rank)
            assert torch.all(model.weight == (1 if init_sync else 1 + rank))
            model.counter.fill_(10 + rank)
            wrapped(torch.ones(1, 2))
            expected = 10 if forward_sync_buffers else 10 + rank
            assert model.counter.item() == expected


def test_ddp_buffer_sync_initialization_and_forward():
    spawn_parallel_fn(
        _ddp_buffer_sync_initialization_and_forward, world_size=2, backend="gloo"
    )
