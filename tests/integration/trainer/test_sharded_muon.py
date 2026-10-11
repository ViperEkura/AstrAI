"""Logical-matrix and multi-step DTensor optimizer state oracles on CPU."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from astrai.optim import muon_adamw as module
from astrai.optim.muon_adamw import _ShardedMuon


def _worker(rank, world, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=180),
    )
    try:
        mesh = init_device_mesh("cpu", (world,))
        original = module.newton_schulz
        calls = []

        def observed(matrix, *args, **kwargs):
            calls.append((tuple(matrix.shape), kwargs["backend"]))
            return original(matrix, *args, **kwargs)

        module.newton_schulz = observed
        shapes = [(3, 7), (9, 4), (4, 9)]
        for dtype in (torch.float32, torch.bfloat16):
            for nesterov in (True, False):
                for mode in ("standard", "buffers", "auto"):
                    for placement in (Shard(0), Shard(1), Replicate()):
                        torch.manual_seed(3407)
                        expected = [
                            nn.Parameter(torch.randn(shape, dtype=dtype))
                            for shape in shapes
                        ]
                        actual = [
                            nn.Parameter(
                                distribute_tensor(
                                    param.detach().clone(), mesh, [placement]
                                )
                            )
                            for param in expected
                        ]
                        options = dict(
                            lr=0.001,
                            weight_decay=0.1,
                            momentum=0.95,
                            nesterov=nesterov,
                            ns_steps=5,
                            adjust_lr_fn="match_rms_adamw",
                        )
                        wanted = torch.optim.Muon(expected, **options)
                        got = _ShardedMuon(
                            actual,
                            reuse_ns_buffers=mode == "buffers",
                            use_ns_kernels=mode == "auto",
                            **options,
                        )
                        calls.clear()
                        for step, scale in enumerate((0.0, 1e-9, 1.0, 0.0, 0.2)):
                            generator = torch.Generator().manual_seed(70 + step)
                            for full, shard in zip(expected, actual):
                                gradient = (
                                    torch.randn(full.shape, generator=generator) * scale
                                ).to(dtype)
                                full.grad = gradient.clone()
                                shard.grad = distribute_tensor(
                                    gradient.clone(), mesh, [placement]
                                )
                            wanted.step()
                            got.step()
                            for full, shard in zip(expected, actual):
                                torch.testing.assert_close(
                                    shard.full_tensor(),
                                    full,
                                    rtol=2e-3,
                                    atol=0.002 if dtype == torch.bfloat16 else 3e-5,
                                )
                                torch.testing.assert_close(
                                    got.state[shard]["momentum_buffer"].full_tensor(),
                                    wanted.state[full]["momentum_buffer"],
                                    rtol=0,
                                    atol=0,
                                )
                            if step == 2:
                                path = output + f"/state-rank-{rank}.pt"
                                torch.save(got.state_dict(), path)
                                resumed = _ShardedMuon(
                                    actual,
                                    reuse_ns_buffers=mode == "buffers",
                                    use_ns_kernels=mode == "auto",
                                    **options,
                                )
                                resumed.load_state_dict(
                                    torch.load(path, weights_only=False)
                                )
                                got = resumed
                            wanted.zero_grad()
                            got.zero_grad()
                        assert all(shape in shapes for shape, _ in calls)
                        if mode == "standard":
                            assert not calls
                        else:
                            assert len(calls) == 15
                            assert {backend for _, backend in calls} == {
                                "auto" if mode == "auto" else "torch"
                            }
        # Unsupported logical rank must fail before momentum/parameter mutation.
        param = nn.Parameter(distribute_tensor(torch.ones(2, 3, 4), mesh, [Shard(0)]))
        param.grad = distribute_tensor(torch.ones(2, 3, 4), mesh, [Shard(0)])
        before = param.full_tensor().clone()
        with pytest.raises(ValueError, match="2D"):
            _ShardedMuon([param])
        torch.testing.assert_close(param.full_tensor(), before, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4, 8])
def test_sharded_muon_parameters_momentum_dispatch_and_resume(tmp_path, world):
    mp.spawn(
        _worker,
        args=(world, "file://" + str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=world,
        join=True,
    )
