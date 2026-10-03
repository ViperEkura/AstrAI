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

from astrai.extension.kernel.muon_ns import is_available, muon_ns
from astrai.extension.kernel.muon_sharded import (
    steps_ as sharded_muon_steps_,
)
from astrai.extension.kernel.muon_sharded import (
    supports as supports_sharded_muon,
)
from astrai.optim.composite import (
    OptimizerFactory,
    composite_state_dict,
    composite_step,
    composite_zero_grad,
    refresh_param_groups,
)
from astrai.optim.muon_replicated import ReplicatedMuon


def _scalar_lr(lr: Any) -> float:
    return lr.item() if isinstance(lr, Tensor) else lr


def _zeropower_reuse_buffers(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> Tensor:
    """Run torch Muon's NS recurrence with scratch tensors reused across steps."""
    if ns_steps >= 100 or grad.ndim != 2 or len(ns_coefficients) != 3:
        raise ValueError("invalid Muon Newton-Schulz input")
    a, b, c = ns_coefficients
    x = grad.bfloat16()
    tall = grad.size(0) > grad.size(1)
    if tall:
        x = x.T
    x.div_(x.norm().clamp(min=eps))
    gram = torch.empty((x.size(0), x.size(0)), dtype=x.dtype, device=x.device)
    gram_update = torch.empty_like(gram)
    next_x = torch.empty_like(x)
    for _ in range(ns_steps):
        torch.mm(x, x.T, out=gram)
        torch.addmm(gram, gram, gram, beta=b, alpha=c, out=gram_update)
        torch.addmm(x, gram_update, x, beta=a, out=next_x)
        x, next_x = next_x, x
    return x.T if tall else x


def _single_tensor_muon_reuse_buffers(
    params: list[Tensor],
    grads: list[Tensor],
    bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
    fused_ns: bool = False,
) -> None:
    if has_complex:
        raise ValueError("Complex parameters are not supported")
    lr = _scalar_lr(lr)
    for param, grad, buf in zip(params, grads, bufs):
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf
        if fused_ns:
            update = muon_ns(update, ns_coefficients, ns_steps, eps)
        else:
            update = _zeropower_reuse_buffers(update, ns_coefficients, ns_steps, eps)
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)


def _sharded_orthogonalize(
    update: Tensor, group: Mapping, *, fused_ns: bool = False
) -> Tensor:
    """Newton-Schulz for a sharded DTensor momentum update.

    NS needs global matmuls, so gather the update to the full matrix,
    orthogonalize it, and scatter the result back onto the update's
    shard layout. ``full_tensor()`` returns the same gathered matrix on
    every rank, so the scatter is a uniform collective.
    """
    full = update.full_tensor()
    if fused_ns:
        ortho = muon_ns(full, group["ns_coefficients"], group["ns_steps"], group["eps"])
    else:
        ortho = _zeropower_via_newtonschulz(
            full, group["ns_coefficients"], group["ns_steps"], group["eps"]
        )
    return distribute_tensor(ortho, update.device_mesh, update.placements)


def _muon_step_group(
    params: list[Tensor],
    grads: list[Tensor],
    bufs: list[Tensor],
    group: Mapping,
    *,
    reuse_ns_buffers: bool,
    fused_ns: bool,
    replicated_muon=None,
) -> None:
    """One optimizer entry for local Tensor and sharded DTensor Muon updates."""
    if (
        fused_ns
        and replicated_muon is not None
        and replicated_muon.step(params, grads, bufs, group)
    ):
        return
    plain, sharded = [], []
    for param, grad, buf in zip(params, grads, bufs):
        (sharded if isinstance(param, DTensor) else plain).append((param, grad, buf))

    if plain:
        pp, gg, bb = (list(t) for t in zip(*plain))
        use_fused_ns = fused_ns and all(grad.is_cuda for grad in gg) and is_available()
        if use_fused_ns or reuse_ns_buffers:
            _single_tensor_muon_reuse_buffers(
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
                fused_ns=use_fused_ns,
            )
        else:
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

    use_fused_ns = (
        fused_ns
        and all(grad.device.type == "cuda" for _, grad, _ in sharded)
        and is_available()
    )
    lr = _scalar_lr(group["lr"])
    fused_items = []
    if fused_ns:
        for param, grad, buf in sharded:
            if supports_sharded_muon(param, grad, buf):
                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], param.shape)
                fused_items.append((param, grad, buf, adjusted_lr))
        sharded_muon_steps_(
            fused_items,
            lr=lr,
            weight_decay=group["weight_decay"],
            momentum=group["momentum"],
            nesterov=group["nesterov"],
            ns_coefficients=group["ns_coefficients"],
            ns_steps=group["ns_steps"],
            eps=group["eps"],
        )
    fused_param_ids = {id(param) for param, _, _, _ in fused_items}
    for param, grad, buf in sharded:
        if id(param) in fused_param_ids:
            continue
        adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], param.shape)
        buf.lerp_(grad, 1 - group["momentum"])
        update = grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf

        param.mul_(1 - lr * group["weight_decay"])
        param.add_(
            _sharded_orthogonalize(update, group, fused_ns=use_fused_ns),
            alpha=-adjusted_lr,
        )


class _ShardedMuon(optim.Muon):
    """Muon with one step entry for Tensor and sharded DTensor parameters.

    The default DTensor path gathers the update for global Newton-Schulz,
    then scatters its result. With fused_ns enabled, supported tall dim-0
    shards use local CUDA kernels and reduced Gram matrices instead.
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
        fused_ns: bool = False,
        process_group=None,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self.reuse_ns_buffers = reuse_ns_buffers
        self.fused_ns = fused_ns
        self.replicated_muon = (
            ReplicatedMuon(process_group) if process_group is not None else None
        )

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

            _muon_step_group(
                params,
                grads,
                bufs,
                group,
                reuse_ns_buffers=self.reuse_ns_buffers,
                fused_ns=self.fused_ns,
                replicated_muon=self.replicated_muon,
            )
        return loss


@OptimizerFactory.register("muon_adamw")
class MuonAdamW(optim.Optimizer):
    """Combined Muon (matrix) + AdamW (non-matrix) optimizer.

    ``fused_ns`` opts into the CUDA NS path. For a DDP model, its process
    group also distributes whole-matrix NS work across replicas. An explicit
    ``muon_process_group`` may supply that context when passing the unwrapped
    module; it must contain replicas with synchronized full gradients. Never
    pass a tensor-parallel or FSDP sharding group as replica context.
    """

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
        fused_ns: bool = False,
        muon_process_group=None,
    ):
        if fused_ns and muon_process_group is None:
            # torch.compile(DDP(...)) retains the DDP wrapper as _orig_mod.
            ddp_model = getattr(model, "_orig_mod", model)
            if isinstance(ddp_model, nn.parallel.DistributedDataParallel):
                muon_process_group = ddp_model.process_group
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
            reuse_ns_buffers=reuse_ns_buffers,
            fused_ns=fused_ns,
            process_group=muon_process_group,
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
