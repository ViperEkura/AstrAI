from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable

from astrai.extension.kernel.cross_entropy import is_available, linear_cross_entropy
from astrai.extension.kernel.gemm import quant_gemm
from astrai.extension.runtime.context import op_context
from astrai.extension.runtime.loader import is_available as kernel_available
from astrai.extension.runtime.operation import ExplicitSelectionError
from astrai.model.components.initialization import should_initialize


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
            and kernel_available("gemm")
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


class Linear(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, bias: bool = False, init_std: float = 0.02
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.init_std = init_std
        # A module must never hand out the bytes torch.empty left behind;
        # models built through AutoRegressiveLM re-init via apply(), but
        # standalone components have no such pass.
        self.reset_parameters()

    def reset_parameters(self):
        if not should_initialize():
            return
        nn.init.normal_(self.weight, mean=0.0, std=self.init_std)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)


class LinearCrossEntropyHead(Linear):
    """Language-model head that can compute CE inside its FSDP forward."""

    def forward(
        self,
        x: Tensor,
        targets: Optional[Tensor] = None,
        *,
        label_smoothing: float = 0.0,
        chunk_size: int = 512,
    ) -> Tensor:
        if targets is None:
            return super().forward(x)
        if targets.shape != x.shape[:-1]:
            raise ValueError("targets must match the training token shape")

        if self.bias is None and x.is_cuda and is_available():
            return linear_cross_entropy(
                x.reshape(-1, x.shape[-1]),
                self.weight,
                targets.reshape(-1),
                label_smoothing=label_smoothing,
                chunk_size=chunk_size,
            )
        logits = super().forward(x)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            label_smoothing=label_smoothing,
            reduction="sum",
        )
