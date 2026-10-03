"""Cross entropy with FP32 reductions and low-precision logits storage."""

import torch
from torch.autograd.function import once_differentiable

from astrai.extension.loader import get_module
from astrai.extension.loader import is_available as _available


def is_available() -> bool:
    return _available("cross_entropy")


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index, reduction):
        logits, targets = logits.contiguous(), targets.contiguous()
        loss, maxima, log_sums, count = get_module("cross_entropy").forward(
            logits, targets, ignore_index, reduction == "mean"
        )
        ctx.save_for_backward(logits, targets, maxima, log_sums, count)
        ctx.ignore_index = ignore_index
        ctx.mean = reduction == "mean"
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        logits, targets, maxima, log_sums, count = ctx.saved_tensors
        grad_logits = get_module("cross_entropy").backward(
            logits,
            targets,
            maxima,
            log_sums,
            count,
            grad_loss.float().contiguous(),
            ctx.ignore_index,
            ctx.mean,
        )
        return grad_logits, None, None, None


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean/sum CE for CUDA [tokens, vocab] logits, with first-order autograd.

    Same objective as ``F.cross_entropy(logits.float(), targets)`` without
    storing a full FP32 logits/log-softmax tensor. Reduction order differs;
    this entry is opt-in and does not promise bitwise-identical gradients.
    """
    if not logits.is_cuda or logits.ndim != 2:
        raise ValueError("logits must be a CUDA matrix")
    if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("logits must be BF16, FP16, or FP32")
    if (
        targets.device != logits.device
        or targets.dtype != torch.int64
        or targets.shape != logits.shape[:1]
    ):
        raise ValueError("targets must be a matching CUDA int64 vector")
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be mean or sum")
    return _CrossEntropy.apply(logits, targets, int(ignore_index), reduction)


__all__ = ["cross_entropy", "is_available"]
