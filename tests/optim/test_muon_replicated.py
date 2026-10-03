import copy
import os

import pytest
import torch
import torch.distributed as dist
from torch import nn

from astrai.extension.kernel.muon_ns import is_available
from astrai.optim.muon_adamw import MuonAdamW
from astrai.optim.muon_replicated import _make_buckets


def test_replicated_muon_bucket_plan_covers_each_matrix_once():
    params = [
        torch.empty(shape) for shape in [(256, 128), (128, 64), (64, 192), (64, 64)]
    ]
    plans = _make_buckets(params, 3, 80000)
    seen = []
    for entries, count in plans:
        spans = {rank: [] for rank in range(3)}
        for index, rank, offset in entries:
            seen.append(index)
            spans[rank].append((offset, offset + params[index].numel()))
            assert offset + params[index].numel() <= count
        for owned in spans.values():
            owned.sort()
            assert all(a[1] <= b[0] for a, b in zip(owned, owned[1:]))
    assert sorted(seen) == list(range(len(params)))


@pytest.fixture(scope="module")
def process_group():
    if (
        int(os.environ.get("WORLD_SIZE", "1")) < 2
        or not torch.cuda.is_available()
        or not is_available()
    ):
        pytest.skip("run with torchrun on at least two CUDA devices with muon_ns")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("nccl")
    yield dist.group.WORLD
    if created:
        dist.destroy_process_group()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrices = nn.ParameterList(
            [
                nn.Parameter(torch.randn(shape))
                for shape in [(256, 128), (128, 64), (64, 192)]
            ]
        )
        self.bias = nn.Parameter(torch.randn(32))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("nesterov", [True, False])
def test_replicated_muon_matches_local_and_checkpoint(process_group, dtype, nesterov):
    torch.manual_seed(31)
    reference = Model().to(device="cuda", dtype=dtype)
    candidate = copy.deepcopy(reference)
    settings = dict(lr=1e-3, weight_decay=0.1, nesterov=nesterov, fused_ns=True)
    baseline = MuonAdamW(reference, **settings)
    owner = MuonAdamW(candidate, muon_process_group=process_group, **settings)
    for step in range(4):
        torch.manual_seed(100 + step)
        for index, (a, b) in enumerate(
            zip(reference.parameters(), candidate.parameters())
        ):
            grad = torch.randn_like(a)
            a.grad = b.grad = None if step == 1 and index == 1 else grad
        baseline.step()
        owner.step()
        for a, b in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(baseline.muon.state.values(), owner.muon.state.values()):
            torch.testing.assert_close(
                a["momentum_buffer"], b["momentum_buffer"], rtol=0, atol=0
            )
        if step == 2:
            state = copy.deepcopy(owner.state_dict())
            owner = MuonAdamW(candidate, muon_process_group=process_group, **settings)
            owner.load_state_dict(state)


def test_replicated_muon_rejects_rank_dependent_missing_gradients(process_group):
    torch.manual_seed(31)
    model = Model().to(device="cuda", dtype=torch.bfloat16)
    optimizer = MuonAdamW(model, fused_ns=True, muon_process_group=process_group)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    if dist.get_rank(process_group) == 0:
        model.matrices[0].grad = None
    with pytest.raises(RuntimeError, match="gradient presence"):
        optimizer.step()


def test_ddp_model_selects_replica_group_and_keeps_parameters_synchronized(
    process_group,
):
    torch.manual_seed(19)
    model = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32)).cuda()
    ddp = nn.parallel.DistributedDataParallel(
        model, device_ids=[torch.cuda.current_device()], process_group=process_group
    )
    optimizer = MuonAdamW(ddp, fused_ns=True)
    assert optimizer.muon.replicated_muon.process_group is process_group
    for step in range(2):
        torch.manual_seed(40 + 10 * step + dist.get_rank(process_group))
        optimizer.zero_grad()
        ddp(torch.randn(4, 128, device="cuda")).square().mean().backward()
        optimizer.step()
        for param in model.parameters():
            expected = param.detach().clone()
            dist.broadcast(expected, src=0, group=process_group)
            torch.testing.assert_close(param, expected, rtol=0, atol=0)
