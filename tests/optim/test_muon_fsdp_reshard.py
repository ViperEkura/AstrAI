"""Local Muon resharding matches a source-rank scatter across GPUs."""

import importlib

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
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(shape, device="cuda", dtype=torch.bfloat16))
                for shape in ((65, 127), (65, 127), (128, 64), (128, 64))
            ]
        )


def _compare_reshard(rank, port, use_ns_kernels):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        init_method=f"tcp://127.0.0.1:{port}",
        device_id=torch.device("cuda", rank),
    )
    module = importlib.import_module(
        "astrai.optimization.optimizer.muon.sharded_update"
    )
    distribute = module.distribute_tensor
    try:
        torch.manual_seed(123)
        reference = _Matrices()
        torch.manual_seed(123)
        candidate = _Matrices()
        fully_shard(reference, reshard_after_forward=False)
        fully_shard(candidate, reshard_after_forward=False)
        options = dict(
            lr=0.03, ns_steps=3, use_ns_kernels=use_ns_kernels, ns_batch_size=2
        )
        baseline = MuonAdamW(reference, **options)
        optimized = MuonAdamW(candidate, **options)
        for step in range(3):
            for index, (expected, actual) in enumerate(
                zip(reference.parameters(), candidate.parameters())
            ):
                gradient = torch.ones_like(expected) * (0.01 * (index + step + 1))
                expected.grad = gradient
                actual.grad = gradient.clone()

            def source_rank(tensor, mesh, placements, **kwargs):
                return distribute(tensor, mesh, placements, src_data_rank=0)

            module.distribute_tensor = source_rank
            baseline.step()
            module.distribute_tensor = distribute
            optimized.step()
            for expected, actual in zip(reference.parameters(), candidate.parameters()):
                torch.testing.assert_close(
                    actual.full_tensor(), expected.full_tensor(), atol=0, rtol=0
                )
                torch.testing.assert_close(
                    optimized.muon.state[actual]["momentum_buffer"].to_local(),
                    baseline.muon.state[expected]["momentum_buffer"].to_local(),
                    atol=0,
                    rtol=0,
                )
    finally:
        module.distribute_tensor = distribute
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("use_ns_kernels", [False, True])
def test_fsdp_muon_local_reshard_matches_scatter(use_ns_kernels):
    mp.spawn(_compare_reshard, args=(find_free_port(), use_ns_kernels), nprocs=2)
