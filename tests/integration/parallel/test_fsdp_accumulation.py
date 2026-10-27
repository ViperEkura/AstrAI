"""FSDP accumulation preserves gradients when parameters stay unsharded."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.tensor import DTensor

from astrai.parallel.executor import FSDPExecutor
from astrai.parallel.setup import find_free_port
from tests.support.capabilities import skip_no_cuda

pytestmark = skip_no_cuda


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(16, 16)
        self.last = nn.Linear(16, 8)

    def forward(self, x):
        return self.last(self.first(x).relu())


def _compare_accumulation(rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        init_method=f"tcp://127.0.0.1:{port}",
        device_id=torch.device("cuda", rank),
    )
    try:
        models = []
        executors = []
        optimizers = []
        for keep in (False, True):
            torch.manual_seed(7)
            model = _TinyModel().cuda()
            executor = FSDPExecutor(
                grad_accum_steps=3, keep_unsharded_during_accumulation=keep
            )
            model = executor._prepare_model(model)
            models.append(model)
            executors.append(executor)
            optimizers.append(torch.optim.SGD(model.parameters(), lr=0.01))

        torch.manual_seed(123 + rank)
        batches = [
            (torch.randn(4, 16, device="cuda"), torch.randn(4, 8, device="cuda"))
            for _ in range(6)
        ]
        for step in range(2):
            for model, executor, optimizer in zip(models, executors, optimizers):
                optimizer.zero_grad(set_to_none=True)
                for index, (x, target) in enumerate(batches[step * 3 : (step + 1) * 3]):
                    with executor.accumulate_explicit(model, sync=index == 2):
                        ((model(x) - target).square().mean() / 3).backward()
            for expected, actual in zip(models[0].parameters(), models[1].parameters()):
                assert isinstance(expected.grad, DTensor)
                assert isinstance(actual.grad, DTensor)
                torch.testing.assert_close(
                    actual.grad.full_tensor(),
                    expected.grad.full_tensor(),
                    atol=1e-6,
                    rtol=1e-6,
                )
            for optimizer in optimizers:
                optimizer.step()
            for expected, actual in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(
                    actual.full_tensor(), expected.full_tensor(), atol=1e-6, rtol=1e-6
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_fsdp_accumulation_matches_reshard_path():
    mp.spawn(_compare_accumulation, args=(find_free_port(),), nprocs=2)
