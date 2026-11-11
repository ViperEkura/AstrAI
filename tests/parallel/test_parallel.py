import warnings

import pytest
import torch

from astrai.parallel import get_rank, spawn_parallel_fn
from astrai.parallel.executor import (
    DDPExecutor,
)
from tests.support.capabilities import requires_nccl

pytestmark = [pytest.mark.distributed, pytest.mark.cuda, requires_nccl(2)]


def _ddp_buffer_sync_initialization_and_forward():
    rank = get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    for init_sync in (True, False):
        for forward_sync_buffers in (True, False):
            model = torch.nn.Linear(2, 2, bias=False, device=device)
            model.weight.data.fill_(1 + rank)
            model.register_buffer("counter", torch.tensor(float(rank), device=device))
            executor = DDPExecutor(
                init_sync=init_sync, forward_sync_buffers=forward_sync_buffers
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                wrapped, _, _ = executor.prepare(lambda: model)

            assert model.counter.item() == (0 if init_sync else rank)
            assert torch.all(model.weight == (1 if init_sync else 1 + rank))
            model.counter.fill_(10 + rank)
            wrapped(torch.ones(1, 2, device=device))
            expected = 10 if forward_sync_buffers else 10 + rank
            assert model.counter.item() == expected


def test_ddp_buffer_sync_initialization_and_forward():
    spawn_parallel_fn(
        _ddp_buffer_sync_initialization_and_forward, world_size=2, backend="nccl"
    )
