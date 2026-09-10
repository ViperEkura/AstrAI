"""Legacy Muon + AdamW combined optimizer."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn, optim
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.optim._muon import (
    _adjust_lr,
    _single_tensor_muon,
    _zeropower_via_newtonschulz,
)

from astrai.optim.composite import (
    OptimizerFactory,
    composite_state_dict,
    composite_step,
    composite_zero_grad,
    refresh_param_groups,
)


def _scalar_lr(lr: Any) -> float:
    return lr.item() if isinstance(lr, Tensor) else lr


def _sharded_orthogonalize(update: Tensor, group: Mapping) -> Tensor:
    """Newton-Schulz for a sharded DTensor momentum update.

    NS needs global matmuls, so gather the update to the full matrix,
    orthogonalize it, and scatter the result back onto the update's
    shard layout. ``full_tensor()`` returns the same gathered matrix on
    every rank, so the scatter is a uniform collective.
    """
    full = update.full_tensor()
    ortho = _zeropower_via_newtonschulz(
        full, group["ns_coefficients"], group["ns_steps"], group["eps"]
    )
    return distribute_tensor(ortho, update.device_mesh, update.placements)


class _ShardedMuon(optim.Muon):
    """Muon that materializes sharded DTensor params around Newton-Schulz.

    FSDP2 hands this optimizer dim-0 sharded DTensor parameters. The NS
    iteration needs global matmuls: run it on the gathered full matrix,
    then scatter the orthogonalized update back onto the parameter's
    sharded layout so momentum buffers and weight decay stay sharded.
    Without this, ``og @ og.T`` produces ``Partial(sum)`` DTensors that
    downstream ``addmm`` calls consume without completing the reduction,
    silently corrupting every update (measured 2e-4-9e-4 relative error
    per step at world_size=2).

    Plain (non-DTensor) params are routed through torch's own
    ``_single_tensor_muon`` so unsharded runs stay bit-for-bit identical
    to ``optim.Muon`` and this class carries only the DTensor delta.
    Element-wise ops (momentum lerp, weight decay, the final ``add_``)
    are DTensor-safe and run directly on the shards.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            bufs: list[Tensor] = []
            self._init_group(group, params, grads, bufs)

            plain, sharded = [], []
            for param, grad, buf in zip(params, grads, bufs):
                (sharded if isinstance(param, DTensor) else plain).append(
                    (param, grad, buf)
                )

            if plain:
                pp, gg, bb = (list(t) for t in zip(*plain))
                _single_tensor_muon(
                    pp,
                    gg,
                    bb,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    momentum=group["momentum"],
                    nesterov=group["nesterov"],
                    ns_coefficients=group["ns_coefficients"],
                    ns_steps=group["ns_steps"],
                    eps=group["eps"],
                    adjust_lr_fn=group["adjust_lr_fn"],
                    has_complex=False,
                )

            lr = _scalar_lr(group["lr"])
            for param, grad, buf in sharded:
                buf.lerp_(grad, 1 - group["momentum"])
                update = grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf

                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], param.shape)
                param.mul_(1 - lr * group["weight_decay"])
                param.add_(_sharded_orthogonalize(update, group), alpha=-adjusted_lr)
        return loss


@OptimizerFactory.register("muon_adamw")
class MuonAdamW(optim.Optimizer):
    """Combined Muon (matrix) + AdamW (non-matrix) optimizer."""

    optimizer_name = "muon_adamw"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr_fn: str = "match_rms_adamw",
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        params = [param for param in model.parameters() if param.requires_grad]
        super().__init__(params, defaults)

        matrix_params: list[Tensor] = []
        other_params: list[Tensor] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.dim() >= 2
                and "norm" not in name
                and "bias" not in name
                and "embed" not in name
                and "lm_head" not in name
            ):
                matrix_params.append(param)
            else:
                other_params.append(param)

        self.muon = _ShardedMuon(
            matrix_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        self.adamw = optim.AdamW(
            [{"params": other_params, "weight_decay": 0.0}],
            lr=lr,
            betas=(0.9, 0.95),
            fused=True,
        )

        self.param_groups = refresh_param_groups([self.muon, self.adamw])

    @torch.no_grad()
    def step(self, closure=None):
        return composite_step([self.muon, self.adamw], closure)

    def zero_grad(self, set_to_none: bool = True):
        composite_zero_grad([self.muon, self.adamw], set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return composite_state_dict({"muon": self.muon, "adamw": self.adamw})

    def load_state_dict(self, state_dict: dict[str, Any]):
        if "muon" not in state_dict or "adamw" not in state_dict:
            raise ValueError(
                "Checkpoint optimizer state is not compatible with muon_adamw"
            )
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = refresh_param_groups([self.muon, self.adamw])
