import math
from typing import Dict, Union

import torch
import torch.nn as nn

from astrai.parallel.executor.gradients import parameter_gradient


def grad_norm(
    model: nn.Module, per_param: bool = False
) -> Union[float, Dict[str, float]]:
    grads = [
        g.detach()
        for p in model.parameters()
        if (g := parameter_gradient(p)) is not None
    ]
    if not grads:
        return 0.0

    total_sq = torch.stack([g.pow(2).sum() for g in grads]).sum()
    if per_param:
        norms = {}
        for name, param in model.named_parameters():
            gradient = parameter_gradient(param)
            if gradient is not None:
                norms[name] = gradient.norm(2).item()
            else:
                norms[name] = 0.0
        norms["total"] = total_sq.sqrt().item()
        return norms
    return total_sq.sqrt().item()


class GradSNRTracker:
    """Track gradient signal-to-noise ratio via EMA of first/second moments.

    SNR = E[g]^2 / Var(g) = E[g]^2 / (E[g^2] - E[g]^2)

    The reported value is the power ratio in decibels: ``10 * log10(SNR)``.

    The tracker accumulates a first-moment tensor and a scalar second-moment
    sum per parameter. The aggregate SNR only needs the sum of squared
    gradients, so a full second-moment tensor would waste model-sized memory.
    Call ``update`` after backward (before ``optimizer.step``) and read
    ``snr`` to get the aggregate SNR across all parameters.
    """

    def __init__(self, beta: float = 0.999, eps: float = 1e-8):
        self.beta = beta
        self.eps = eps
        self._first: Dict[int, torch.Tensor] = {}
        self._second: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module, *, step_span: int = 1) -> None:
        if step_span <= 0:
            raise ValueError("step_span must be positive")
        # With sampled updates, preserve approximately the same EMA decay
        # measured in optimizer steps (intermediate gradients are not observed).
        beta = self.beta**step_span
        for param in model.parameters():
            gradient = parameter_gradient(param)
            if gradient is None:
                continue
            pid = id(param)
            g = gradient.detach()
            squared_norm = g.square().sum(dtype=torch.float32)
            if pid not in self._first:
                self._first[pid] = g.clone()
                self._second[pid] = squared_norm
            else:
                self._first[pid].mul_(beta).add_(g, alpha=1 - beta)
                self._second[pid].mul_(beta).add_(squared_norm, alpha=1 - beta)

    @property
    def snr(self) -> float:
        if not self._first:
            return 0.0
        # Aggregate device scalars before crossing the device/host boundary.
        # This is one synchronization per device, not two per parameter.
        device_pairs = {}
        for pid, m in self._first.items():
            signal = m.square().sum(dtype=torch.float32)
            noise = (self._second[pid] - signal).clamp_min(0)
            device_pairs.setdefault(m.device, []).append(torch.stack((signal, noise)))
        total_signal = 0.0
        total_noise = 0.0
        for pairs in device_pairs.values():
            signal, noise = torch.stack(pairs).sum(dim=0).tolist()
            total_signal += signal
            total_noise += noise
        snr = total_signal / (total_noise + self.eps)
        return 10.0 * math.log10(max(snr, self.eps))


def ctx_get_loss(ctx):
    return ctx.loss


def ctx_get_lr(ctx):
    return ctx.optimizer.param_groups[-1]["lr"]


def ctx_get_val_loss(ctx):
    return ctx.val_loss


def ctx_get_grad_norm(ctx):
    return ctx.grad_norm


def ctx_get_grad_snr(ctx):
    cached = getattr(ctx, "grad_snr_value", None)
    if cached is not None:
        return cached
    tracker = getattr(ctx, "grad_snr_tracker", None)
    if tracker is None:
        return None
    return tracker.snr


def ctx_get_moe_metric(ctx, key):
    return ctx.strategy._moe_metrics.get(key)
