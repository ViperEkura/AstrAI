"""Opt-in CUDA cross entropy; no attention-dispatch registration.

Both entry points return a sum. The trainer owns token/global normalization.
Only first-order gradients are supported. Chunked linear CE recomputes logits
in backward; GEMM tiling and reduction order can change rounding.
"""

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from astrai.extension.loader import get_module
from astrai.extension.loader import is_available as _available


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


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index, smoothing):
        loss, maxima, log_sums = get_module("cross_entropy").forward(
            logits, targets, ignore_index, smoothing
        )
        ctx.save_for_backward(logits, targets, maxima, log_sums)
        ctx.ignore_index, ctx.smoothing = ignore_index, smoothing
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        logits, targets, maxima, log_sums = ctx.saved_tensors
        grad = get_module("cross_entropy").backward(
            logits,
            targets,
            maxima,
            log_sums,
            grad_loss.float().contiguous(),
            ctx.ignore_index,
            ctx.smoothing,
        )
        return grad, None, None, None


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    """CUDA CE sum without a full FP32 logits/log-softmax allocation."""
    _check(logits, targets, label_smoothing)
    return _CrossEntropy.apply(
        logits.contiguous(),
        targets.contiguous(),
        int(ignore_index),
        float(label_smoothing),
    )


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, ignore_index, smoothing, chunk_size):
        loss, maxima, log_sums = get_module("cross_entropy").linear_forward(
            hidden,
            weight,
            targets,
            ignore_index,
            smoothing,
            chunk_size,
        )
        ctx.save_for_backward(hidden, weight, targets, maxima, log_sums)
        ctx.ignore_index, ctx.smoothing = ignore_index, smoothing
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        hidden, weight, targets, maxima, log_sums = ctx.saved_tensors
        dx, dw = get_module("cross_entropy").linear_backward(
            hidden,
            weight,
            targets,
            maxima,
            log_sums,
            grad_loss.float().contiguous(),
            ctx.ignore_index,
            ctx.smoothing,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
        )
        dx = dx if ctx.needs_input_grad[0] else None
        dw = dw if ctx.needs_input_grad[1] else None
        return dx, dw, None, None, None, None


def linear_cross_entropy(
    hidden: Tensor,
    weight: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    chunk_size: int = 512,
) -> Tensor:
    """CUDA bias-free linear + CE sum, using bounded logits scratch storage.

    ``hidden`` is [tokens, hidden], ``weight`` is [vocab, hidden]. Gradients
    reduce all tokens per vocabulary tile in one GEMM, without a full FP32
    weight-gradient buffer. No external kernel package.
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
    return _LinearCrossEntropy.apply(
        hidden.contiguous(),
        weight.contiguous(),
        targets.contiguous(),
        int(ignore_index),
        float(label_smoothing),
        chunk_size,
    )


__all__ = ["cross_entropy", "linear_cross_entropy", "is_available"]
