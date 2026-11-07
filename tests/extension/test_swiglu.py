"""Numerical and dispatch checks for the fused SwiGLU CUDA training operator."""

import pytest
import torch
import torch.nn.functional as F

from astrai.extension.backend.swiglu import SwiGLUDispatcher, swiglu
from astrai.extension.kernel.swiglu import swiglu as fused_swiglu
from astrai.extension.runtime.context import op_backend
from astrai.extension.runtime.loader import is_available
from astrai.extension.runtime.operation import ExplicitSelectionError

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not is_available("swiglu"), reason="SwiGLU kernel not built"),
]


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (torch.float16, 3e-3, 3e-3),
        (torch.bfloat16, 2e-2, 2e-2),
        (torch.float32, 2e-5, 2e-5),
    ],
)
@pytest.mark.parametrize("shape", [(0, 128), (1025,), (128, 256), (4, 128, 4096)])
def test_fused_forward_and_backward_match_float_reference(dtype, rtol, atol, shape):
    torch.manual_seed(33)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    with op_backend(swiglu="cuda"):
        output = swiglu(up, gate)
    output.backward(grad)

    ref_up = up.detach().float().requires_grad_()
    ref_gate = gate.detach().float().requires_grad_()
    reference = ref_up * F.silu(ref_gate)
    reference.backward(grad.float())

    torch.testing.assert_close(output.float(), reference, rtol=rtol, atol=atol)
    torch.testing.assert_close(up.grad.float(), ref_up.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(gate.grad.float(), ref_gate.grad, rtol=rtol, atol=atol)


@pytest.mark.cuda
def test_auto_selects_fused_forward_with_registered_backward():
    up = torch.randn(16, 128, device="cuda", requires_grad=True)
    gate = torch.randn_like(up, requires_grad=True)
    assert SwiGLUDispatcher().dispatch(up, gate) is fused_swiglu
    output = swiglu(up, gate)
    assert output.grad_fn is not None
    output.sum().backward()
    assert up.grad is not None and gate.grad is not None


@pytest.mark.cuda
def test_strided_input_uses_torch_and_explicit_cuda_rejects():
    up = torch.randn(8, 16, device="cuda").T
    gate = torch.randn(8, 16, device="cuda").T
    assert SwiGLUDispatcher().dispatch(up, gate) is not fused_swiglu
    torch.testing.assert_close(swiglu(up, gate), up * F.silu(gate))
    with op_backend(swiglu="cuda"):
        with pytest.raises(ExplicitSelectionError):
            swiglu(up, gate)


@pytest.mark.cuda
def test_native_entry_rejects_mismatched_shapes():
    up = torch.randn(8, 16, device="cuda")
    gate = torch.randn(8, 15, device="cuda")
    with pytest.raises(RuntimeError, match="same shape"):
        fused_swiglu(up, gate)


@pytest.mark.cuda
def test_unaligned_contiguous_view_uses_scalar_kernel():
    base_up = torch.randn(2050, device="cuda", dtype=torch.bfloat16)
    base_gate = torch.randn_like(base_up)
    up = base_up[1:2049].requires_grad_()
    gate = base_gate[1:2049].requires_grad_()
    grad = torch.randn_like(up)
    assert up.is_contiguous() and up.storage_offset() == 1

    with op_backend(swiglu="cuda"):
        output = swiglu(up, gate)
    grad_up, grad_gate = torch.autograd.grad(output, (up, gate), grad)

    ref_up = up.detach().float().requires_grad_()
    ref_gate = gate.detach().float().requires_grad_()
    reference = ref_up * F.silu(ref_gate)
    ref_grad_up, ref_grad_gate = torch.autograd.grad(
        reference, (ref_up, ref_gate), grad.float()
    )
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(grad_up.float(), ref_grad_up, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(grad_gate.float(), ref_grad_gate, rtol=2e-2, atol=2e-2)
