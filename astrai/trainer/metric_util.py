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


def ctx_get_loss(ctx):
    return ctx.loss


def ctx_get_lr(ctx):
    return ctx.optimizer.param_groups[-1]["lr"]


def ctx_get_val_loss(ctx):
    return ctx.val_loss


def ctx_get_grad_norm(ctx):
    return ctx.grad_norm


def ctx_get_language_model_loss(ctx):
    return ctx.language_model_loss


def ctx_get_router_loss(ctx):
    return ctx.router_loss


def ctx_get_router_aux_loss(ctx):
    return ctx.router_aux_loss


def ctx_get_router_z_loss(ctx):
    return ctx.router_z_loss


def ctx_get_router_entropy(ctx):
    return ctx.router_entropy


def ctx_get_expert_load_min(ctx):
    return ctx.expert_load_min


def ctx_get_expert_load_max(ctx):
    return ctx.expert_load_max


def ctx_get_expert_load_cv(ctx):
    return ctx.expert_load_cv
