"""Independent references for configurable persistent precision."""

import copy

import torch
from torch import nn

from astrai.optimization.optimizer.muon.optimizer import MuonAdamW
from astrai.optimization.optimizer.precision import PrecisionOptimizer


def make_precision_model(dtype, device):
    model = nn.Module()
    model.matrix = nn.Parameter(torch.full((8, 8), 0.02, dtype=dtype, device=device))
    model.norm = nn.Parameter(torch.ones(8, dtype=dtype, device=device))
    return model


def make_precision_optimizer(model, kind):
    if kind == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1
        )
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    return MuonAdamW(model, lr=1e-3, ns_backend="torch", reuse_ns_buffers=True)


def check_precision_storage(
    kind, compute_dtype, master_dtype, stored_dtype, device, offload
):
    torch.manual_seed(42)
    model = make_precision_model(compute_dtype, device)
    reference = copy.deepcopy(model).to(master_dtype).float()
    native = make_precision_optimizer(reference, kind)
    optimizer = PrecisionOptimizer(
        make_precision_optimizer(model, kind),
        cpu_offload=offload,
        master_dtype=master_dtype,
        state_dtype=stored_dtype,
    )
    for parameter, master in optimizer.master_parameters:
        torch.testing.assert_close(
            parameter, master.to(device=device, dtype=compute_dtype), atol=0, rtol=0
        )
    expected_children = (
        (native.muon, native.adamw) if kind == "muon_adamw" else (native,)
    )
    reference_by_name = dict(reference.named_parameters())
    for step in range(12):
        for (name, p), q in zip(model.named_parameters(), reference.parameters()):
            p.grad = None
            gradient = torch.rand_like(p, dtype=torch.float32) * 0.25
            if name == "norm" and step % 4 == 0:
                gradient = None
            p.main_grad = gradient
            q.grad = gradient
        optimizer.step()
        native.step()
        with torch.no_grad():
            # Native FP32 arithmetic, then independently round only the
            # persistent fields selected by the test's storage policy.
            if master_dtype == torch.bfloat16:
                for q in reference.parameters():
                    q.copy_(q.bfloat16().float())
            if stored_dtype == torch.bfloat16:
                for child in expected_children:
                    for state in child.state.values():
                        for name, value in state.items():
                            if (
                                isinstance(value, torch.Tensor)
                                and value.is_floating_point()
                                and name != "step"
                            ):
                                value.copy_(value.bfloat16().float())
        names = {p: name for name, p in model.named_parameters()}
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q.to(compute_dtype), atol=0, rtol=0)
        for p, master in optimizer.master_parameters:
            assert master.dtype == master_dtype
            torch.testing.assert_close(
                master.cpu(),
                reference_by_name[names[p]].detach().to(master_dtype).cpu(),
                atol=0,
                rtol=0,
            )
        for child in optimizer._optimizers:
            for state in child.state.values():
                for name, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        assert value.dtype == (
                            torch.float32 if name == "step" else stored_dtype
                        )
                        if offload:
                            assert value.device.type == "cpu" and value.is_pinned()
        optimizer.zero_grad()
        native.zero_grad()
        if step == 5:
            saved = copy.deepcopy(optimizer.state_dict())
            model = make_precision_model(compute_dtype, device)
            optimizer = PrecisionOptimizer(
                make_precision_optimizer(model, kind),
                cpu_offload=offload,
                master_dtype=master_dtype,
                state_dtype=stored_dtype,
            )
            optimizer.load_state_dict(saved)
