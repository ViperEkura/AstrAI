from typing import Dict

import torch
import torch.nn as nn


def grad_norm(model: nn.Module, per_param: bool = False) -> float | Dict[str, float]:
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0

    total_sq = torch.stack([g.pow(2).sum() for g in grads]).sum()
    if per_param:
        norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                norms[name] = param.grad.norm(2).item()
            else:
                norms[name] = 0.0
        norms["total"] = total_sq.sqrt().item()
        return norms
    return total_sq.sqrt().item()


class GradSNRTracker:
    """Track gradient signal-to-noise ratio via EMA of first/second moments.

    SNR = E[g]^2 / Var(g) = E[g]^2 / (E[g^2] - E[g]^2)

    The tracker accumulates per-parameter EMA moments across optimizer steps.
    Call ``update`` after backward (before ``optimizer.step``) and read
    ``snr`` to get the aggregate SNR across all parameters.
    """

    def __init__(self, beta: float = 0.999, eps: float = 1e-8):
        self.beta = beta
        self.eps = eps
        self._first: Dict[int, torch.Tensor] = {}
        self._second: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        beta = self.beta
        for param in model.parameters():
            if param.grad is None:
                continue
            pid = id(param)
            g = param.grad.detach()
            if pid not in self._first:
                self._first[pid] = g.clone()
                self._second[pid] = g.pow(2).clone()
            else:
                self._first[pid].mul_(beta).add_(g, alpha=1 - beta)
                self._second[pid].mul_(beta).addcmul_(g, g, value=1 - beta)

    @property
    def snr(self) -> float:
        if not self._first:
            return 0.0
        total_signal = 0.0
        total_noise = 0.0
        for m, v in zip(self._first.values(), self._second.values()):
            signal = m.pow(2).sum().item()
            noise = (v - m.pow(2)).clamp(min=0).sum().item()
            total_signal += signal
            total_noise += noise
        return total_signal / (total_noise + self.eps)


def ctx_get_loss(ctx):
    return ctx.loss


def ctx_get_lr(ctx):
    return ctx.optimizer.param_groups[-1]["lr"]


def ctx_get_val_loss(ctx):
    return ctx.val_loss


def ctx_get_grad_norm(ctx):
    return ctx.grad_norm


def ctx_get_grad_snr(ctx):
    tracker = getattr(ctx, "grad_snr_tracker", None)
    if tracker is None:
        return None
    return tracker.snr


def ctx_get_moe_aux_loss(ctx):
    return ctx.strategy._moe_metrics.get("aux_loss")


def ctx_get_router_entropy(ctx):
    return ctx.strategy._moe_metrics.get("router_entropy")


def ctx_get_dead_expert_fraction(ctx):
    return ctx.strategy._moe_metrics.get("dead_expert_fraction")


def ctx_get_load_imbalance_mean(ctx):
    return ctx.strategy._moe_metrics.get("load_imbalance_mean")


def ctx_get_load_imbalance_max(ctx):
    return ctx.strategy._moe_metrics.get("load_imbalance_max")
