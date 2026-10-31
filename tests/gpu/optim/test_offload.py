"""Pinned-state lifetime, immediate snapshots and restart continuity."""

import copy
import io

import pytest
import torch
from torch import nn

from astrai.optimization.optimizer.muon.optimizer import MuonAdamW
from astrai.optimization.optimizer.precision import PrecisionOptimizer
from tests.support.capabilities import skip_no_cuda


def _model():
    model = nn.Module()
    model.matrix = nn.Parameter(
        torch.full((8, 8), 0.02, device="cuda", dtype=torch.bfloat16)
    )
    model.norm = nn.Parameter(torch.ones(8, device="cuda", dtype=torch.bfloat16))
    model.embed = nn.Embedding(8, 8, device="cuda", dtype=torch.bfloat16)
    model.lm_head = nn.Linear(8, 8, bias=False, device="cuda", dtype=torch.bfloat16)
    return model


def _optimizer(model, kind):
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=2e-5, fused=True)
    if kind == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=2e-5, momentum=0.9, weight_decay=0.1
        )
    return MuonAdamW(model, lr=2e-5, reuse_ns_buffers=True, ns_backend="torch")


@skip_no_cuda
@pytest.mark.parametrize("kind", ["adamw", "sgd", "muon_adamw"])
def test_pipelined_state_is_complete_on_return_and_restores(kind):
    torch.manual_seed(42)
    model = _model()
    reference = copy.deepcopy(model).float()
    reference_by_name = dict(reference.named_parameters())
    optimizer = PrecisionOptimizer(_optimizer(model, kind), cpu_offload=True)
    expected = _optimizer(reference, kind)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for step in range(16):
        with torch.cuda.stream(stream):
            for (name, p), q in zip(model.named_parameters(), reference.parameters()):
                if step % 7 == 0 and name == "norm":
                    p.main_grad = None
                    p.grad = None
                    q.grad = None
                else:
                    gradient = torch.rand_like(p, dtype=torch.float32) * 0.25
                    if step % 2:
                        p.main_grad = None
                        p.grad = gradient.bfloat16()
                        q.grad = p.grad.float()
                    else:
                        p.main_grad = gradient
                        q.grad = gradient
            optimizer.step()
        # No external synchronize: the CPU snapshot must already be readable.
        payload = io.BytesIO()
        torch.save(optimizer.state_dict(), payload)
        payload.seek(0)
        saved = torch.load(payload, weights_only=True)
        expected.step()
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q.bfloat16(), atol=0, rtol=0)
        names = {p: name for name, p in model.named_parameters()}
        for value, master, compute in zip(
            saved["master_params"], optimizer._master_params, optimizer._model_params
        ):
            assert master.device.type == "cpu" and master.is_pinned()
            torch.testing.assert_close(value, master, atol=0, rtol=0)
            torch.testing.assert_close(
                value, reference_by_name[names[compute]].detach().cpu(), atol=0, rtol=0
            )
        optimizer.zero_grad()
        expected.zero_grad()
        if step == 7:
            model = _model()
            optimizer = PrecisionOptimizer(_optimizer(model, kind), cpu_offload=True)
            optimizer.load_state_dict(saved)
        for child in optimizer._optimizers:
            for state in child.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        assert value.device.type == "cpu" and value.is_pinned()
