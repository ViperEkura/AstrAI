"""Linear dispatch with an opt-in BF16 native forward and backward."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable

from astrai.extension.kernel.gemm import quant_gemm
from astrai.extension.runtime.context import op_context
from astrai.extension.runtime.loader import is_available
from astrai.extension.runtime.operation import ExplicitSelectionError


class _NativeLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        flat = x.reshape(-1, x.shape[-1])
        return quant_gemm(flat, weight, bias=bias).reshape(
            *x.shape[:-1], weight.shape[0]
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad = grad_output.reshape(-1, weight.shape[0]).contiguous()
        dx = dw = db = None
        if ctx.needs_input_grad[0]:
            dx = quant_gemm(grad, weight, trans_b=False).reshape_as(x)
        if ctx.needs_input_grad[1]:
            dw = quant_gemm(
                grad, x.reshape(-1, x.shape[-1]), trans_a=True, trans_b=False
            )
        if ctx.has_bias and ctx.needs_input_grad[2]:
            db = grad.float().sum(0).to(grad.dtype)
        return dx, dw, db


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    choice, origin = op_context.choice("linear")
    if choice in (None, "auto", "torch"):
        return F.linear(x, weight, bias)
    if choice == "cuda":
        if (
            x.is_cuda
            and weight.device == x.device
            and x.dtype == weight.dtype == torch.bfloat16
            and x.ndim >= 1
            and weight.ndim == 2
            and x.shape[-1] == weight.shape[1]
            and (
                bias is None
                or (
                    bias.device == x.device
                    and bias.dtype == x.dtype
                    and bias.shape == (weight.shape[0],)
                )
            )
            and is_available("gemm")
        ):
            if x.numel() and weight.numel():
                return _NativeLinear.apply(x, weight, bias)
        elif origin == "context":
            raise ExplicitSelectionError(
                "linear=cuda requires CUDA BF16 operands and the GEMM extension"
            )
    elif origin == "context":
        raise ValueError(f"Unknown linear implementation: {choice!r}")
    return F.linear(x, weight, bias)
