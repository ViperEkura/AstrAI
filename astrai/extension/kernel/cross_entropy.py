"""Opt-in CUDA linear cross entropy; no attention-dispatch registration.

The entry point returns a sum. The trainer owns token/global normalization.
Only first-order gradients are supported. Chunked linear CE projects gradients
in forward and scales them in backward;
GEMM tiling and reduction order can change rounding.
"""

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from astrai.extension.runtime.loader import get_module
from astrai.extension.runtime.loader import is_available as _available


def is_available() -> bool:
    return _available("cross_entropy")


def _check(logits, targets, label_smoothing):
    if (
        not logits.is_cuda
        or logits.ndim != 2
        or logits.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        raise ValueError("input must be a CUDA BF16/FP16/FP32 matrix")
    if (
        targets.device != logits.device
        or targets.dtype != torch.int64
        or targets.shape != logits.shape[:1]
    ):
        raise ValueError("targets must be a matching CUDA int64 vector")
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError("label_smoothing must be in [0, 1]")


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, ignore_index, smoothing, chunk_size):
        loss, dx, dw = get_module("cross_entropy").linear_forward(
            hidden,
            weight,
            targets,
            ignore_index,
            smoothing,
            chunk_size,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
        )
        # Keep version checks on the source tensors and immutable FP32 gradients.
        # Scaling after projection avoids overflow/underflow before normalization.
        ctx.save_for_backward(dx, dw, hidden, weight)
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        dx_raw, dw_raw, hidden, weight = ctx.saved_tensors
        dx, dw = get_module("cross_entropy").linear_backward(
            dx_raw, dw_raw, grad_loss.float().contiguous(), hidden, weight
        )
        return (
            dx if ctx.needs_input_grad[0] else None,
            dw if ctx.needs_input_grad[1] else None,
            None,
            None,
            None,
            None,
        )


def linear_cross_entropy(
    hidden: Tensor,
    weight: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    chunk_size: int = 512,
) -> Tensor:
    """CUDA bias-free linear + CE sum, using bounded logits scratch storage.

    ``hidden`` is [tokens, hidden], ``weight`` is [vocab, hidden]. Each chunk
    computes logits, loss and projected gradients once.
    FP32 gradient buffers are scaled and cast in backward without recomputing
    logits or modifying the saved buffers. No external kernel package.
    """
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        hidden, weight = hidden.to(dtype), weight.to(dtype)
    _check(hidden, targets, label_smoothing)
    if (
        weight.device != hidden.device
        or weight.dtype != hidden.dtype
        or weight.ndim != 2
        or weight.shape[1] != hidden.shape[1]
    ):
        raise ValueError("weight must be a matching [vocab, hidden] matrix")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    if not torch.is_grad_enabled():
        return get_module("cross_entropy").linear_forward(
            hidden.contiguous(),
            weight.contiguous(),
            targets.contiguous(),
            int(ignore_index),
            float(label_smoothing),
            chunk_size,
            False,
            False,
        )[0]
    return _LinearCrossEntropy.apply(
        hidden.contiguous(),
        weight.contiguous(),
        targets.contiguous(),
        int(ignore_index),
        float(label_smoothing),
        chunk_size,
    )


__all__ = ["linear_cross_entropy", "is_available"]
