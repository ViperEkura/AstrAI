from types import SimpleNamespace

import pytest
import torch
from torch import nn

from astrai.extension.kernel.muon_ns import is_available
from astrai.extension.kernel.muon_sharded import (
    is_available as sharded_is_available,
)
from astrai.extension.kernel.muon_sharded import (
    step_ as sharded_step_,
)
from astrai.extension.kernel.muon_sharded import (
    steps_ as sharded_steps_,
)
from astrai.optim import muon_adamw
from astrai.optim.muon_adamw import MuonAdamW


class _TinyMuonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(8, 4))
        self.vector = nn.Parameter(torch.randn(4))

    def forward(self):
        return self.matrix.square().mean() + self.vector.square().mean()


def _assign_grads(model, grads):
    for param, grad in zip(model.parameters(), grads):
        param.grad = grad.clone()


def test_fused_ns_flag_falls_back_without_cuda_kernel():
    torch.manual_seed(11)
    reference = _TinyMuonModel()
    candidate = _TinyMuonModel()
    candidate.load_state_dict(reference.state_dict())
    grads = [torch.randn_like(param) for param in reference.parameters()]

    ref_optimizer = MuonAdamW(reference, lr=1e-3, fused_ns=False)
    fused_optimizer = MuonAdamW(candidate, lr=1e-3, fused_ns=True)
    for _ in range(3):
        _assign_grads(reference, grads)
        _assign_grads(candidate, grads)
        ref_optimizer.step()
        fused_optimizer.step()

    for ref_param, fused_param in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(fused_param, ref_param, rtol=0, atol=0)


skip_no_muon_ns = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="Muon NS CUDA kernel is not available",
)


@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@skip_no_muon_ns
def test_fused_ns_matches_reused_torch_path_on_cuda(steps, nesterov, weight_decay):
    torch.manual_seed(19)
    reference = _TinyMuonModel().to(device="cuda", dtype=torch.bfloat16)
    candidate = _TinyMuonModel().to(device="cuda", dtype=torch.bfloat16)
    candidate.load_state_dict(reference.state_dict())
    grads = [torch.randn_like(param) for param in reference.parameters()]

    settings = {"lr": 1e-3, "nesterov": nesterov, "weight_decay": weight_decay}
    ref_optimizer = MuonAdamW(reference, reuse_ns_buffers=True, **settings)
    fused_optimizer = MuonAdamW(candidate, fused_ns=True, **settings)
    assert len(ref_optimizer.muon.param_groups[0]["params"]) == 1
    assert len(ref_optimizer.adamw.param_groups[0]["params"]) == 1
    for _ in range(steps):
        _assign_grads(reference, grads)
        _assign_grads(candidate, grads)
        ref_optimizer.step()
        fused_optimizer.step()

    for ref_param, fused_param in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(fused_param, ref_param, rtol=0, atol=0)

    assert fused_optimizer.state_dict().keys() == ref_optimizer.state_dict().keys()


@skip_no_muon_ns
def test_fused_ns_falls_back_when_module_is_unavailable(monkeypatch):
    torch.manual_seed(31)
    reference = _TinyMuonModel().to("cuda")
    candidate = _TinyMuonModel().to("cuda")
    candidate.load_state_dict(reference.state_dict())
    gradients = [torch.randn_like(param) for param in reference.parameters()]
    monkeypatch.setattr(muon_adamw, "is_available", lambda: False)
    monkeypatch.setattr(
        muon_adamw,
        "muon_ns",
        lambda *args, **kwargs: pytest.fail("unavailable kernel was called"),
    )
    ref_optimizer = MuonAdamW(reference, lr=1e-3)
    fused_optimizer = MuonAdamW(candidate, lr=1e-3, fused_ns=True)
    _assign_grads(reference, gradients)
    _assign_grads(candidate, gradients)
    ref_optimizer.step()
    fused_optimizer.step()
    for ref_param, fused_param in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(fused_param, ref_param, rtol=0, atol=0)


def test_sharded_ns_gathers_before_fused_call_and_redistributes(monkeypatch):
    events = []
    full = torch.randn(4, 3)
    ortho = torch.randn_like(full)

    class FakeDTensor:
        device_mesh = object()
        placements = (object(),)

        def full_tensor(self):
            events.append("gather")
            return full

    def fake_muon_ns(tensor, coefficients, steps, eps):
        assert tensor is full
        events.append("fused_ns")
        return ortho

    def fake_distribute(tensor, mesh, placements):
        assert tensor is ortho
        events.append("distribute")
        return SimpleNamespace(value=tensor)

    monkeypatch.setattr(muon_adamw, "muon_ns", fake_muon_ns)
    monkeypatch.setattr(muon_adamw, "distribute_tensor", fake_distribute)
    group = {"ns_coefficients": (3.4445, -4.775, 2.0315), "ns_steps": 5, "eps": 1e-7}
    result = muon_adamw._sharded_orthogonalize(FakeDTensor(), group, fused_ns=True)
    assert result.value is ortho
    assert events == ["gather", "fused_ns", "distribute"]


def _run_two_rank_dtensor_parity(rank, init_file):
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        mesh = DeviceMesh("cuda", [0, 1])
        torch.manual_seed(41)
        full = torch.randn((12, 7), device=f"cuda:{rank}", dtype=torch.bfloat16)
        sharded = distribute_tensor(full.clone(), mesh, [Shard(0)])
        group = {
            "ns_coefficients": (3.4445, -4.775, 2.0315),
            "ns_steps": 5,
            "eps": 1e-7,
        }
        actual = muon_adamw._sharded_orthogonalize(sharded, group, fused_ns=True)
        expected_full = muon_adamw._zeropower_via_newtonschulz(
            full.clone(), group["ns_coefficients"], group["ns_steps"], group["eps"]
        )
        expected = distribute_tensor(expected_full, mesh, [Shard(0)])
        assert torch.equal(actual.to_local(), expected.to_local())
    finally:
        dist.destroy_process_group()


@skip_no_muon_ns
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_fused_ns_two_rank_dtensor_parity(tmp_path):
    torch.multiprocessing.spawn(
        _run_two_rank_dtensor_parity,
        args=(str(tmp_path / "dtensor-init"),),
        nprocs=2,
        join=True,
    )


def _run_two_rank_sharded_fused_parity(rank, init_file):
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        mesh = DeviceMesh("cuda", [0, 1])
        coefficients = (3.4445, -4.775, 2.0315)
        for shape, dtype in (
            ((64, 16), torch.bfloat16),
            ((65, 16), torch.bfloat16),
            ((64, 16), torch.float32),
        ):
            torch.manual_seed(67)
            full_param = torch.randn(shape, device=f"cuda:{rank}", dtype=dtype)
            full_grad = torch.randn_like(full_param)
            full_buffer = torch.randn_like(full_param)
            reference_param = distribute_tensor(full_param.clone(), mesh, [Shard(0)])
            candidate_param = distribute_tensor(full_param.clone(), mesh, [Shard(0)])
            reference_grad = distribute_tensor(full_grad.clone(), mesh, [Shard(0)])
            candidate_grad = distribute_tensor(full_grad.clone(), mesh, [Shard(0)])
            reference_buffer = distribute_tensor(full_buffer.clone(), mesh, [Shard(0)])
            candidate_buffer = distribute_tensor(full_buffer.clone(), mesh, [Shard(0)])
            for _ in range(3):
                reference_buffer.lerp_(reference_grad, 0.05)
                update = reference_grad.lerp(reference_buffer, 0.95)
                full_update = update.full_tensor()
                orthogonal = muon_adamw._zeropower_via_newtonschulz(
                    full_update, coefficients, 5, 1e-7
                )
                reference_param.mul_(1 - 0.01 * 0.1)
                reference_param.add_(
                    distribute_tensor(orthogonal, mesh, [Shard(0)]), alpha=-0.01
                )
                sharded_step_(
                    candidate_param,
                    candidate_grad,
                    candidate_buffer,
                    lr=0.01,
                    adjusted_lr=0.01,
                    weight_decay=0.1,
                    momentum=0.95,
                    nesterov=True,
                    ns_coefficients=coefficients,
                    ns_steps=5,
                    eps=1e-7,
                )
            difference = (
                (
                    reference_param.to_local().float()
                    - candidate_param.to_local().float()
                )
                .abs()
                .max()
                .item()
            )
            assert difference <= 0.02, f"{shape=} {dtype=} {rank=} {difference=}"

        for shape in ((64, 16), (32, 16)):
            torch.manual_seed(79)
            full_param = torch.randn(shape, device=f"cuda:{rank}", dtype=torch.bfloat16)
            full_grad = torch.randn_like(full_param)
            reference_param = distribute_tensor(
                full_param.clone().requires_grad_(), mesh, [Shard(0)]
            )
            candidate_param = distribute_tensor(
                full_param.clone().requires_grad_(), mesh, [Shard(0)]
            )
            reference_param.grad = distribute_tensor(
                full_grad.clone(), mesh, [Shard(0)]
            )
            candidate_param.grad = distribute_tensor(
                full_grad.clone(), mesh, [Shard(0)]
            )
            options = {"lr": 1e-3, "ns_steps": 5, "weight_decay": 0.1}
            reference_optimizer = muon_adamw._ShardedMuon([reference_param], **options)
            candidate_optimizer = muon_adamw._ShardedMuon(
                [candidate_param], fused_ns=True, **options
            )
            for _ in range(3):
                reference_optimizer.step()
                candidate_optimizer.step()
            max_abs = (
                (
                    reference_param.to_local().float()
                    - candidate_param.to_local().float()
                )
                .abs()
                .max()
                .item()
            )
            assert max_abs <= 0.02, f"optimizer integration {shape=} {rank=} {max_abs=}"
            if shape == (32, 16):
                assert max_abs == 0.0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not sharded_is_available(),
    reason="requires two CUDA devices and the sharded Muon extension",
)
def test_sharded_fused_ns_two_rank_parity(tmp_path):
    torch.multiprocessing.spawn(
        _run_two_rank_sharded_fused_parity,
        args=(str(tmp_path / "sharded-muon-init"),),
        nprocs=2,
        join=True,
    )


def _run_two_rank_sharded_bucket(rank, init_file):
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import Shard, distribute_tensor

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        mesh = DeviceMesh("cuda", [0, 1])
        coefficients = (3.4445, -4.775, 2.0315)
        items = []
        references = []
        for seed, shape in ((71, (64, 16)), (73, (128, 16))):
            torch.manual_seed(seed)
            full_param = torch.randn(shape, device=f"cuda:{rank}", dtype=torch.bfloat16)
            full_grad = torch.randn_like(full_param)
            full_buffer = torch.zeros_like(full_param)
            candidate_param = distribute_tensor(full_param.clone(), mesh, [Shard(0)])
            candidate_grad = distribute_tensor(full_grad.clone(), mesh, [Shard(0)])
            candidate_buffer = distribute_tensor(full_buffer.clone(), mesh, [Shard(0)])
            reference_param = distribute_tensor(full_param.clone(), mesh, [Shard(0)])
            reference_grad = distribute_tensor(full_grad.clone(), mesh, [Shard(0)])
            reference_buffer = distribute_tensor(full_buffer.clone(), mesh, [Shard(0)])
            reference_buffer.lerp_(reference_grad, 0.05)
            update = reference_grad.lerp(reference_buffer, 0.95)
            full_update = update.full_tensor()
            orthogonal = muon_adamw._zeropower_via_newtonschulz(
                full_update, coefficients, 3, 1e-7
            )
            reference_param.mul_(1 - 0.01 * 0.1)
            reference_param.add_(
                distribute_tensor(orthogonal, mesh, [Shard(0)]), alpha=-0.01
            )
            references.append(reference_param)
            items.append((candidate_param, candidate_grad, candidate_buffer, 0.01))

        original_all_reduce = dist.all_reduce
        all_reduce_calls = 0

        def counted_all_reduce(*args, **kwargs):
            nonlocal all_reduce_calls
            all_reduce_calls += 1
            return original_all_reduce(*args, **kwargs)

        dist.all_reduce = counted_all_reduce
        try:
            sharded_steps_(
                items,
                lr=0.01,
                weight_decay=0.1,
                momentum=0.95,
                nesterov=True,
                ns_coefficients=coefficients,
                ns_steps=3,
                eps=1e-7,
            )
        finally:
            dist.all_reduce = original_all_reduce
        assert all_reduce_calls == 4
        for reference, (candidate, _, _, _) in zip(references, items):
            max_abs = (
                (reference.to_local().float() - candidate.to_local().float())
                .abs()
                .max()
                .item()
            )
            assert max_abs <= 0.02
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not sharded_is_available(),
    reason="requires two CUDA devices and the sharded Muon extension",
)
def test_sharded_fused_ns_buckets_collectives(tmp_path):
    torch.multiprocessing.spawn(
        _run_two_rank_sharded_bucket,
        args=(str(tmp_path / "sharded-bucket-init"),),
        nprocs=2,
        join=True,
    )
