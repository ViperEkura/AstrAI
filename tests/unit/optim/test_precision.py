"""Update precision and restart continuity for BF16 compute parameters."""

import copy
import io

import pytest
import torch
from torch import nn

from astrai.config.train_config import ComponentSpec
from astrai.optimization.optimizer.muon.optimizer import MuonAdamW
from astrai.optimization.optimizer.precision import FP32Optimizer
from astrai.trainer.components import (
    build_optimizer_component,
    load_optimizer_state_dict,
)


class PrecisionModel(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.matrix = nn.Parameter(torch.full((8, 8), 0.02, dtype=dtype))
        self.norm = nn.Parameter(torch.ones(8, dtype=dtype))
        self.embed = nn.Embedding(8, 8, dtype=dtype)
        self.lm_head = nn.Linear(8, 8, bias=False, dtype=dtype)


def test_sub_ulp_updates_and_decay_survive_restart():
    model = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = FP32Optimizer(torch.optim.SGD([model], lr=1e-5, weight_decay=0.1))
    reference = nn.Parameter(model.float())
    expected = torch.optim.SGD([reference], lr=1e-5, weight_decay=0.1)
    direct = model.detach().clone()
    for step in range(600):
        model.grad = torch.ones_like(model)
        reference.grad = torch.ones_like(reference)
        optimizer.step()
        expected.step()
        direct.mul_(1 - 1e-6).add_(-1e-5)
        if step == 299:
            payload = io.BytesIO()
            torch.save(optimizer.state_dict(), payload)
            payload.seek(0)
            saved = torch.load(payload, weights_only=True)
            restored = nn.Parameter(model.clone())
            optimizer = FP32Optimizer(
                torch.optim.SGD([restored], lr=1e-5, weight_decay=0.1)
            )
            optimizer.load_state_dict(saved)
            model = restored
    torch.testing.assert_close(
        optimizer.state_dict()["master_params"][0], reference, atol=0, rtol=0
    )
    torch.testing.assert_close(model, reference.bfloat16(), atol=0, rtol=0)
    assert torch.all(direct == 1)
    assert torch.all(model < 1)


@pytest.mark.parametrize("reuse_buffers", [False, True])
@pytest.mark.parametrize(
    "device,cpu_offload",
    [
        ("cpu", False),
        pytest.param(
            "cuda",
            True,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_muon_and_adam_keep_fp32_state_and_resume(reuse_buffers, device, cpu_offload):
    torch.manual_seed(9)
    model = PrecisionModel(torch.bfloat16).to(device)
    model.norm = nn.Parameter(model.norm.float())
    reference = copy.deepcopy(model).float()
    kwargs = dict(lr=2e-5, weight_decay=0.1, reuse_ns_buffers=reuse_buffers)
    optimizer = FP32Optimizer(MuonAdamW(model, **kwargs), cpu_offload=cpu_offload)
    expected = MuonAdamW(reference, **kwargs)
    update_devices = set()

    def observe_update(optimizer, args, kwargs):
        update_devices.update(
            p.device.type
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        )

    for child in optimizer._optimizers:
        child.register_step_pre_hook(observe_update)
    for step in range(120):
        for p, q in zip(model.parameters(), reference.parameters()):
            p.grad = torch.rand_like(p) * 0.25
            q.grad = p.grad.float()
        optimizer.step()
        expected.step()
        optimizer.zero_grad()
        expected.zero_grad()
        if step == 59:
            saved = copy.deepcopy(optimizer.state_dict())
            model = copy.deepcopy(model)
            optimizer = FP32Optimizer(
                MuonAdamW(model, **kwargs), cpu_offload=cpu_offload
            )
            optimizer.load_state_dict(saved)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q.to(p.dtype), atol=0, rtol=0)
    assert torch.all(model.norm < 1)
    assert update_devices == {device}
    for master in optimizer.state_dict()["master_params"]:
        assert master.dtype == torch.float32
        if cpu_offload:
            assert master.device.type == "cpu"
    for child in (optimizer.optimizer.muon, optimizer.optimizer.adamw):
        for state in child.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    assert value.dtype == torch.float32
                    if cpu_offload:
                        assert value.device.type == "cpu"
    actual = optimizer.optimizer.state_dict()
    wanted = expected.state_dict()
    for child in ("muon", "adamw"):
        for key, state in actual[child]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(
                    value.cpu(), wanted[child]["state"][key][name].cpu(), atol=0, rtol=0
                )


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_factory_closure_missing_gradient_and_scheduler_options(cpu_offload):
    model = nn.Linear(2, 1, bias=False).bfloat16()
    optimizer = build_optimizer_component(
        ComponentSpec(
            type="torch.optim:SGD", params={"lr": 1e-5, "cpu_offload": cpu_offload}
        ),
        model,
    )
    assert isinstance(optimizer, FP32Optimizer)
    calls = []

    def closure():
        calls.append(True)
        optimizer.zero_grad()
        loss = model.weight.float().sum()
        loss.backward()
        return loss

    assert optimizer.step(closure) is not None
    before = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad()
    optimizer.step()
    assert len(calls) == 1
    torch.testing.assert_close(
        before["master_params"][0],
        optimizer.state_dict()["master_params"][0],
        atol=0,
        rtol=0,
    )
    optimizer.param_groups[0]["lr"] = 3e-5
    load_optimizer_state_dict(optimizer, before)
    assert optimizer.param_groups[0]["lr"] == 3e-5
    model.weight.grad = torch.ones(
        model.weight.numel(), dtype=model.weight.dtype
    ).view_as(model.weight)
    optimizer.zero_grad(set_to_none=False)
    assert torch.count_nonzero(model.weight.grad) == 0


def test_legacy_or_low_precision_master_state_is_rejected():
    model = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    base = torch.optim.AdamW([model])
    legacy = base.state_dict()
    optimizer = FP32Optimizer(base)
    with pytest.raises(ValueError, match="no FP32 master"):
        optimizer.load_state_dict(legacy)
    saved = optimizer.state_dict()
    saved["master_params"] = [model.detach()]
    with pytest.raises(ValueError, match="do not match"):
        optimizer.load_state_dict(saved)
