"""Legacy Muon + AdamW combined optimizer."""

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn, optim
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.optim._muon import (
    _adjust_lr,
    _single_tensor_muon,
    _zeropower_via_newtonschulz,
)

from astrai.extension.backend.newton_schulz import newton_schulz
from astrai.optim.composite import (
    OptimizerFactory,
    composite_state_dict,
    composite_step,
    composite_zero_grad,
    refresh_param_groups,
)


def _scalar_lr(lr: Any) -> float:
    return lr.item() if isinstance(lr, Tensor) else lr


def _single_tensor_muon_reuse_buffers(
    params: List[Tensor],
    grads: List[Tensor],
    bufs: List[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: Tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
    has_complex: bool,
    use_ns_kernels: bool = False,
    ns_batch_size: int = 4,
) -> None:
    if has_complex:
        raise ValueError("Complex parameters are not supported")
    lr = _scalar_lr(lr)
    buckets: Dict[Tuple[Any, ...], List[Tuple[Tensor, Tensor, Tensor]]] = {}
    for param, grad, buf in zip(params, grads, bufs):
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")
        key = (tuple(grad.shape), grad.device, grad.dtype, buf.dtype, param.dtype)
        buckets.setdefault(key, []).append((param, grad, buf))
    batch_size = ns_batch_size if use_ns_kernels else 1
    for bucket in buckets.values():
        for start in range(0, len(bucket), batch_size):
            chunk = bucket[start : start + batch_size]
            updates = []
            for param, grad, buf in chunk:
                buf.lerp_(grad, 1 - momentum)
                updates.append(grad.lerp(buf, momentum) if nesterov else buf)
            packed = torch.stack(updates) if len(chunk) > 1 else updates[0]
            output = newton_schulz(
                packed,
                ns_coefficients,
                ns_steps,
                eps,
                backend="auto" if use_ns_kernels else "torch",
            )
            for index, (param, grad, buf) in enumerate(chunk):
                update = output[index] if len(chunk) > 1 else output
                # BF16 non-Nesterov NS normalizes the momentum storage itself.
                # Packing must preserve that state transition on each buffer.
                if len(chunk) > 1 and not nesterov and buf.dtype == torch.bfloat16:
                    buf.copy_(packed[index])
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
                param.mul_(1 - lr * weight_decay)
                param.add_(update, alpha=-adjusted_lr)


def _validate_sharded_matrix(param: DTensor) -> None:
    if param.ndim != 2 or param.device_mesh.ndim != 1 or len(param.placements) != 1:
        raise ValueError("sharded Muon requires a 2D logical matrix on a 1D mesh")
    placement = param.placements[0]
    if not isinstance(placement, (Shard, Replicate)) or (
        isinstance(placement, Shard) and placement.dim not in (0, 1)
    ):
        raise ValueError("sharded Muon supports only Shard(0), Shard(1) or Replicate")


def _sharded_orthogonalize(
    update: DTensor,
    group: Mapping,
    *,
    reuse_ns_buffers: bool = False,
    use_ns_kernels: bool = False,
    preserve_input_mutation: bool = False,
) -> Tensor:
    """Newton-Schulz for a sharded DTensor momentum update.

    NS needs global matmuls, so gather the update to the full matrix,
    orthogonalize it, and scatter the result back onto the update's
    shard layout. ``full_tensor()`` returns the same gathered matrix on
    every rank, so the scatter is a uniform collective.
    """
    full = update.full_tensor()
    if reuse_ns_buffers or use_ns_kernels:
        ortho = newton_schulz(
            full,
            group["ns_coefficients"],
            group["ns_steps"],
            group["eps"],
            backend="auto" if use_ns_kernels else "torch",
        )
    else:
        ortho = _zeropower_via_newtonschulz(
            full, group["ns_coefficients"], group["ns_steps"], group["eps"]
        )
    if preserve_input_mutation and full.dtype == torch.bfloat16:
        # Full-matrix BF16 normalization mutates the caller storage. A
        # gather breaks the non-Nesterov momentum alias, so restore this
        # state transition on each shard before the next optimizer update.
        update.copy_(distribute_tensor(full, update.device_mesh, update.placements))
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

    Plain (non-DTensor) params use torch's ``_single_tensor_muon`` by default.
    The optional scratch-buffer path keeps the same five-step NS recurrence.
    Element-wise ops (momentum lerp, weight decay, the final ``add_``)
    are DTensor-safe and run directly on the shards.
    """

    def __init__(
        self,
        params,
        *,
        reuse_ns_buffers: bool = False,
        use_ns_kernels: bool = False,
        ns_batch_size: int = 4,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self.reuse_ns_buffers = reuse_ns_buffers
        self.use_ns_kernels = use_ns_kernels
        if not isinstance(ns_batch_size, int) or ns_batch_size < 1:
            raise ValueError("ns_batch_size must be a positive integer")
        self.ns_batch_size = ns_batch_size

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            bufs: List[Tensor] = []
            self._init_group(group, params, grads, bufs)

            plain, sharded = [], []
            for param, grad, buf in zip(params, grads, bufs):
                (sharded if isinstance(param, DTensor) else plain).append(
                    (param, grad, buf)
                )

            for param, _, _ in sharded:
                _validate_sharded_matrix(param)

            if plain:
                pp, gg, bb = (list(t) for t in zip(*plain))
                if self.reuse_ns_buffers or self.use_ns_kernels:
                    step_plain = _single_tensor_muon_reuse_buffers
                    extra_kwargs = {
                        "use_ns_kernels": self.use_ns_kernels,
                        "ns_batch_size": self.ns_batch_size,
                    }
                else:
                    step_plain = _single_tensor_muon
                    extra_kwargs = {}
                step_plain(
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
                    **extra_kwargs,
                )

            lr = _scalar_lr(group["lr"])
            for param, grad, buf in sharded:
                buf.lerp_(grad, 1 - group["momentum"])
                update = grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf

                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], param.shape)
                ortho = _sharded_orthogonalize(
                    update,
                    group,
                    reuse_ns_buffers=self.reuse_ns_buffers,
                    use_ns_kernels=self.use_ns_kernels,
                    preserve_input_mutation=not group["nesterov"],
                )
                param.mul_(1 - lr * group["weight_decay"])
                param.add_(ortho, alpha=-adjusted_lr)
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
        reuse_ns_buffers: bool = False,
        use_ns_kernels: bool = False,
        ns_batch_size: int = 4,
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

        matrix_params: List[Tensor] = []
        other_params: List[Tensor] = []
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
            reuse_ns_buffers=reuse_ns_buffers,
            use_ns_kernels=use_ns_kernels,
            ns_batch_size=ns_batch_size,
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

    def state_dict(self) -> Dict[str, Any]:
        return composite_state_dict({"muon": self.muon, "adamw": self.adamw})

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if "muon" not in state_dict or "adamw" not in state_dict:
            raise ValueError(
                "Checkpoint optimizer state is not compatible with muon_adamw"
            )
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = refresh_param_groups([self.muon, self.adamw])
