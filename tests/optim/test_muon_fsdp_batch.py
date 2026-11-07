"""FSDP Muon batching preserves sharded parameter and momentum updates."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import fully_shard

from astrai.optimization.optimizer.muon import MuonAdamW
from astrai.parallel.setup import find_free_port
from tests.support.capabilities import skip_no_cuda

pytestmark = skip_no_cuda


class _Matrices(nn.Module):
    def __init__(self):
        super().__init__()
        shapes = [(64, 128)] * 5 + [(128, 64)] * 2 + [(64, 64)]
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(shape, device="cuda", dtype=torch.bfloat16))
                for shape in shapes
            ]
        )


def _compare_batch_with_reference(rank, world_size, port, nesterov):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        device_id=torch.device("cuda", rank),
    )
    try:
        torch.manual_seed(123)
        reference = _Matrices()
        torch.manual_seed(123)
        candidate = _Matrices()
        fully_shard(reference, reshard_after_forward=False)
        fully_shard(candidate, reshard_after_forward=False)
        options = {"lr": 1e-3, "nesterov": nesterov, "ns_steps": 3}
        optimizers = [
            MuonAdamW(reference, use_ns_kernels=False, **options),
            MuonAdamW(candidate, use_ns_kernels=True, ns_batch_size=4, **options),
        ]
        for step in range(3):
            for index, (expected, actual) in enumerate(
                zip(reference.parameters(), candidate.parameters())
            ):
                gradient = torch.ones_like(expected) * (
                    0.01 * (index + step + rank + 1)
                )
                expected.grad = gradient
                actual.grad = gradient.clone()
            for optimizer in optimizers:
                optimizer.step()
            for expected, actual in zip(reference.parameters(), candidate.parameters()):
                torch.testing.assert_close(
                    actual.full_tensor(), expected.full_tensor(), atol=0.003, rtol=1e-4
                )
                assert torch.equal(
                    optimizers[0].muon.state[expected]["momentum_buffer"].to_local(),
                    optimizers[1].muon.state[actual]["momentum_buffer"].to_local(),
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.cuda
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("nesterov", [False, True])
def test_fsdp_muon_batch_matches_reference(nesterov, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA devices")
    port = find_free_port()
    if world_size == 1:
        _compare_batch_with_reference(0, world_size, port, nesterov)
    else:
        mp.spawn(
            _compare_batch_with_reference,
            args=(world_size, port, nesterov),
            nprocs=world_size,
        )
