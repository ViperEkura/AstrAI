from typing import Dict, Union

import torch
import torch.nn as nn


def grad_norm(
    model: nn.Module, per_param: bool = False
) -> Union[float, Dict[str, float]]:
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


def ctx_get_loss(ctx):
    return ctx.loss


def ctx_get_lr(ctx):
    return ctx.optimizer.param_groups[-1]["lr"]


def ctx_get_val_loss(ctx):
    return ctx.val_loss


def ctx_get_grad_norm(ctx):
    return ctx.grad_norm


def ctx_get_moe_metric(ctx, key):
    return ctx.strategy._moe_metrics.get(key)
