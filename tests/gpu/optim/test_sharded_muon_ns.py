"""Assigned-GPU qualification: observe native NS on full DTensor matrices."""

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from astrai.extension.backend import newton_schulz as backend
from astrai.extension.kernel.newton_schulz import is_available
from astrai.extension.policy import newton_schulz as plan
from astrai.optim.muon_adamw import _ShardedMuon
from tests.gpu.optim.test_muon_ns_batch import _small_plans

WORLD = int(os.environ.get("ASTRAI_MUON_TEST_WORLD_SIZE", "2"))


def _worker(rank, world, rendezvous, output, nesterov):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=180),
    )
    try:
        mesh = init_device_mesh("cuda", (world,))
        torch.manual_seed(3407)
        expected = nn.Parameter(
            torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        )
        actual = nn.Parameter(
            distribute_tensor(expected.detach().clone(), mesh, [Shard(0)])
        )
        wanted = torch.optim.Muon([expected], lr=0.001, nesterov=nesterov)
        got = _ShardedMuon([actual], lr=0.001, nesterov=nesterov, use_ns_kernels=True)
        launches = []
        original = backend.cuda.iterate

        def observed(matrix, *args):
            launches.append(
                {
                    "shape": list(matrix.shape),
                    "native_operations": sum(bool(choice[0]) for choice in args[-1]),
                }
            )
            return original(matrix, *args)

        backend.cuda.iterate = observed
        torch.cuda.reset_peak_memory_stats()
        with plan.override(_small_plans()):
            for seed, scale in enumerate((0.0, 1e-9, 1.0, 0.0, 0.2)):
                torch.manual_seed(90 + seed)
                gradient = torch.randn_like(expected) * scale
                expected.grad = gradient.clone()
                actual.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
                wanted.step()
                got.step()
                torch.testing.assert_close(
                    actual.full_tensor(), expected, rtol=0.003, atol=0.004
                )
                torch.testing.assert_close(
                    got.state[actual]["momentum_buffer"].full_tensor(),
                    wanted.state[expected]["momentum_buffer"],
                    rtol=0,
                    atol=0,
                )
                wanted.zero_grad()
                got.zero_grad()
        assert len(launches) == 5
        assert all(
            item["shape"] == [64, 128] and item["native_operations"] > 0
            for item in launches
        )
        torch.cuda.synchronize()
        Path(output, f"ns-dispatch-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "world_size": world,
                    "rank": rank,
                    "nesterov": nesterov,
                    "launches": launches,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
                indent=2,
            )
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < WORLD
    or not torch.cuda.is_available()
    or not is_available(),
    reason="requires assigned CUDA test GPUs and native NS extension",
)
@pytest.mark.parametrize("nesterov", [False, True])
def test_native_full_matrix_dispatch_and_momentum(tmp_path, nesterov):
    mp.spawn(
        _worker,
        args=(WORLD, "file://" + str(tmp_path / "rendezvous"), str(tmp_path), nesterov),
        nprocs=WORLD,
        join=True,
    )
