import math
from typing import Dict, Optional, Union

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

    total_sq = torch.stack([g.float().square().sum() for g in grads]).sum()
    if per_param:
        norms = {}
        for name, param in model.named_parameters():
            gradient = parameter_gradient(param)
            if gradient is not None:
                norms[name] = gradient.float().norm(2).item()
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


def ctx_get_grad_snr(ctx):
    return getattr(ctx, "grad_snr_value", None)


def ctx_get_moe_metric(ctx, key):
    return ctx.strategy._moe_metrics.get(key)


class GradSNRTracker:
    """Temporal SNR of fixed sampled coordinates of the reduced update gradient.

    At most 65,536 coordinates are sampled across all trainable parameters.
    This measures EMA stability along a changing training trajectory, including
    clipping. It is not a fixed-weight noise scale or an optimal batch size.
    Proposed observations remain separate until the trainer commits an update.
    """

    def __init__(
        self, beta: float = 0.999, eps: float = 1e-8, max_samples: int = 65536
    ):
        if not 0 <= beta < 1 or eps <= 0 or max_samples <= 0:
            raise ValueError("invalid gradient SNR decay, epsilon or sample count")
        self.beta, self.eps, self.max_samples = beta, eps, max_samples
        self.interval = 1
        self.last_step = 0
        self.last_observation_step = 0
        self.observations = 0
        self._first = None
        self._second = None
        self._layout = None
        self._plan = None
        self._indices = None
        self._pending = None

    def _sample(self, model):
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        parameters = [
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        ]
        layout = [(name, tuple(p.shape)) for name, p in parameters]
        if self._layout is not None and layout != self._layout:
            raise ValueError("gradient SNR parameter layout changed")
        if not parameters:
            return torch.empty(0)
        device = parameters[0][1].device
        if any(p.device != device for _, p in parameters):
            raise ValueError("gradient SNR requires parameters on one device per rank")
        if self._plan is None:
            total = sum(p.numel() for _, p in parameters)
            count = min(total, self.max_samples)
            if not count:
                return torch.empty(0, device=device)
            # One reproducible random coordinate in each equal-width stratum.
            if self._indices is None:
                generator = torch.Generator().manual_seed(3407)
                indices = (
                    (
                        torch.arange(count, dtype=torch.float64)
                        + torch.rand(count, dtype=torch.float64, generator=generator)
                    )
                    * (total / count)
                ).long()
            else:
                indices = self._indices.cpu()
            self._plan = []
            offset = 0
            for _, parameter in parameters:
                end = offset + parameter.numel()
                left, right = torch.searchsorted(
                    indices, torch.tensor([offset, end])
                ).tolist()
                self._plan.append((offset, left, right))
                offset = end
            self._indices = indices.to(device)
            self._layout = layout
        samples = torch.zeros(self._indices.numel(), dtype=torch.float32, device=device)
        for (_, parameter), (offset, left, right) in zip(parameters, self._plan):
            gradient = parameter_gradient(parameter)
            if left != right and gradient is not None:
                indices = self._indices[left:right] - offset
                gradient = gradient.detach()
                values = (
                    gradient.view(-1).index_select(0, indices)
                    if gradient.is_contiguous()
                    else gradient[torch.unravel_index(indices, gradient.shape)]
                )
                samples[left:right] = values.float()
        return samples

    def _snr(self, first, second, observations) -> Optional[float]:
        if observations < 2:
            return None
        signal, power = torch.stack((first.double().square().sum(), second)).tolist()
        noise = max(power - signal, 0.0)
        floor = max(self.eps, 8 * torch.finfo(first.dtype).eps)
        if (
            not all(math.isfinite(x) for x in (signal, noise))
            or power <= 0
            or signal <= floor * power
            or noise <= floor * power
        ):
            return None
        return 10 * math.log10(signal / noise)

    @torch.no_grad()
    def stage(self, model: nn.Module, *, step_span: int = 1) -> Optional[float]:
        if step_span <= 0:
            raise ValueError("step_span must be positive")
        self.discard()
        gradient = self._sample(model)
        if not gradient.numel() or not torch.isfinite(gradient).all().item():
            return None
        # Convert before squaring: a FP32 reduction cannot repair BF16 overflow.
        second = gradient.double().square().sum()
        if self._first is None:
            first = gradient
        else:
            beta = self.beta**step_span
            first = (
                self._first.to(gradient.device).mul(beta).add_(gradient, alpha=1 - beta)
            )
            second = (
                self._second.to(gradient.device).mul(beta).add_(second, alpha=1 - beta)
            )
        count = self.observations + 1
        self._pending = (first, second, count)
        return self._snr(first, second, count)

    def commit(self, step: int) -> None:
        if self._pending is not None:
            self._first, self._second, self.observations = self._pending
            self.last_observation_step = step
        self.last_step = step
        self.discard()

    def discard(self) -> None:
        self._pending = None

    def update(self, model: nn.Module, *, step_span: int = 1) -> None:
        """Observe and immediately commit, for standalone diagnostics."""
        if step_span <= 0:
            raise ValueError("step_span must be positive")
        step = self.last_step + step_span
        self.stage(model, step_span=step - self.last_observation_step)
        self.commit(step)

    @property
    def snr(self) -> Optional[float]:
        return self._snr(self._first, self._second, self.observations)

    def state_dict(self) -> Dict:
        return dict(
            version=1,
            beta=self.beta,
            eps=self.eps,
            max_samples=self.max_samples,
            interval=self.interval,
            last_step=self.last_step,
            last_observation_step=self.last_observation_step,
            observations=self.observations,
            layout=self._layout if self.observations else None,
            indices=self._indices.detach().cpu().clone() if self.observations else None,
            first=None if self._first is None else self._first.detach().cpu().clone(),
            second=None
            if self._second is None
            else self._second.detach().cpu().clone(),
        )

    def load_state_dict(self, state: Dict) -> None:
        if state["version"] != 1 or any(
            state[key] != getattr(self, key) for key in ("beta", "eps", "max_samples")
        ):
            raise ValueError("incompatible gradient SNR estimator state")
        first, second, indices = (state[key] for key in ("first", "second", "indices"))
        if first is None:
            valid = state["observations"] == 0 and second is None and indices is None
        else:
            valid = (
                first.dtype == torch.float32
                and first.ndim == 1
                and 0 < first.numel() <= self.max_samples
                and second is not None
                and second.dtype == torch.float64
                and second.ndim == 0
                and indices is not None
                and indices.dtype == torch.int64
                and indices.shape == first.shape
                and state["observations"] > 0
            )
        if (
            not valid
            or not 0 <= state["last_observation_step"] <= state["last_step"]
            or state["interval"] <= 0
        ):
            raise ValueError("invalid gradient SNR moments or cadence")
        self._first, self._second = first, second
        self._layout = state["layout"]
        self.interval, self.last_step, self.observations = (
            state[key] for key in ("interval", "last_step", "observations")
        )
        self.last_observation_step = state["last_observation_step"]
        self._plan = None
        self._indices = indices
        self.discard()
