"""Contracts for per-operation dispatch and scoped selections."""

from concurrent.futures import ThreadPoolExecutor

import torch

from astrai.extension import (
    OpDispatcher,
    OpSpec,
    op_backend,
    op_context,
    snapshot_op_selection,
    swiglu,
    use_op_selection,
)
from astrai.extension.backend.attention import AttentionDispatcher, AttentionSpec
from astrai.extension.backend.newton_schulz import (
    SymmDispatcher,
    SymmSpec,
    SyrkDispatcher,
    SyrkSpec,
)
from astrai.extension.backend.rotary import RotaryDispatcher, RotarySpec
from astrai.extension.backend.swiglu import SwiGLUDispatcher, SwiGLUSpec


def test_each_operation_has_its_own_dispatcher_and_spec():
    for dispatcher, spec in (
        (AttentionDispatcher, AttentionSpec),
        (RotaryDispatcher, RotarySpec),
        (SwiGLUDispatcher, SwiGLUSpec),
        (SyrkDispatcher, SyrkSpec),
        (SymmDispatcher, SymmSpec),
    ):
        assert issubclass(dispatcher, OpDispatcher)
        assert issubclass(spec, OpSpec)


def test_selection_snapshot_moves_to_worker_and_restores():
    with op_backend(attention="torch_native", swiglu="torch"):
        snapshot = snapshot_op_selection()

    def worker():
        before = op_context.snapshot().get("attention")
        with use_op_selection(snapshot):
            inside = (
                op_context.snapshot().get("attention"),
                op_context.snapshot().get("swiglu"),
            )
        return before, inside, op_context.snapshot().get("attention")

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(worker).result() == (
            None,
            ("torch_native", "torch"),
            None,
        )


def test_swiglu_backward_uses_forward_graph_after_scope_exit():
    up = torch.randn(2, 8, requires_grad=True)
    gate = torch.randn(2, 8, requires_grad=True)
    reference_up = up.detach().clone().requires_grad_()
    reference_gate = gate.detach().clone().requires_grad_()

    with op_backend(swiglu="torch"):
        loss = swiglu(up, gate).sum()
    loss.backward()

    reference = (reference_up * torch.nn.functional.silu(reference_gate)).sum()
    reference.backward()
    torch.testing.assert_close(up.grad, reference_up.grad)
    torch.testing.assert_close(gate.grad, reference_gate.grad)


def test_attention_spec_checks_all_projected_inputs_for_backward():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q)
    spec = AttentionDispatcher().build_spec(q, k, v, None, 0, None, False, None)
    assert spec.needs_backward
    assert spec.phase == "train"
