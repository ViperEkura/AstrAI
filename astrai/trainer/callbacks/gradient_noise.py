"""Passive DDP gradient observations and checkpointable scalar noise estimates."""

import math
from typing import Dict, Optional

import torch
from torch import nn


@torch.no_grad()
def squared_gradient_norm(parameter: nn.Parameter) -> torch.Tensor:
    # Cast before multiplication: a float32 reduction alone still squares in BF16.
    return parameter.grad.detach().float().square().sum(dtype=torch.float64)


class GradientNoiseTracker:
    """EMA of signal power and per-target covariance, measured at fixed weights.

    The batch-size unit is the loss normalizer (valid targets for SEQ/SFT).
    Estimates use Appendix A.1 of arXiv:1812.06162, with weighted rank batches.
    Only scalar committed observations survive a checkpoint.
    """

    def __init__(self, interval: int = 100, beta: float = 0.95):
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("grad_noise_interval must be a positive integer")
        if not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("grad_noise_beta must be finite and in [0, 1)")
        self.interval = interval
        self.beta = beta
        self.samples = 0
        self.signal = 0.0
        self.covariance = 0.0
        self.last_probe_step = 0
        self.pending: Optional[Dict] = None

    def stage(
        self,
        step: int,
        weighted_local_norm: float,
        global_norm: float,
        batch_size: float,
        ranks: int,
    ) -> None:
        self.pending = None
        if ranks < 2 or batch_size <= 0:
            return
        if not all(
            math.isfinite(x) for x in (weighted_local_norm, global_norm, batch_size)
        ):
            return
        covariance = max(
            0.0, (weighted_local_norm - batch_size * global_norm) / (ranks - 1)
        )
        signal = global_norm - covariance / batch_size
        if not math.isfinite(covariance) or not math.isfinite(signal):
            return
        self.pending = dict(
            step=step, signal=signal, covariance=covariance, batch_size=batch_size
        )

    def commit(self, step: int) -> Dict[str, float]:
        if self.pending is None:
            return {}
        observation = self.pending
        self.pending = None
        if observation["step"] != step or step <= self.last_probe_step:
            raise ValueError(
                "Gradient-noise observation does not match the committed step"
            )
        beta = self.beta if self.samples else 0.0
        self.signal = beta * self.signal + (1 - beta) * observation["signal"]
        self.covariance = (
            beta * self.covariance + (1 - beta) * observation["covariance"]
        )
        self.samples += 1
        self.last_probe_step = step
        result = {
            "grad_noise_sample_step": step,
            "grad_noise_batch_size": observation["batch_size"],
            "grad_noise_signal": self.signal,
            "grad_noise_covariance": self.covariance,
        }
        # A noisy or negative signal estimate is warmup, not an enormous ratio.
        if self.samples >= 2 and self.signal > 0:
            ratio = self.covariance / self.signal
            if math.isfinite(ratio):
                result["grad_noise_scale"] = ratio
        return result

    def state_dict(self) -> Dict:
        return dict(
            version=1,
            interval=self.interval,
            beta=self.beta,
            samples=self.samples,
            signal=self.signal,
            covariance=self.covariance,
            last_probe_step=self.last_probe_step,
            normalization="loss_targets",
        )

    def load_state_dict(self, state: Dict, committed_step: int) -> None:
        if state.get("version") != 1 or state.get("normalization") != "loss_targets":
            raise ValueError(
                "Unsupported gradient-noise state version or normalization"
            )
        if state.get("interval") != self.interval or state.get("beta") != self.beta:
            raise ValueError(
                "Cannot resume gradient-noise state with different sampling settings"
            )
        for key in ("samples", "last_probe_step"):
            value = state.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid gradient-noise {key}")
        if (
            state["last_probe_step"] % self.interval
            or state["samples"] > state["last_probe_step"] // self.interval
        ):
            raise ValueError("Invalid gradient-noise sampling cadence")
        if state["last_probe_step"] > committed_step:
            raise ValueError("Gradient-noise state is ahead of the checkpoint")
        if bool(state["samples"]) != bool(state["last_probe_step"]):
            raise ValueError("Inconsistent gradient-noise sample count and step")
        for key in ("signal", "covariance"):
            value = state.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid gradient-noise {key}")
        if state["covariance"] < 0:
            raise ValueError("Negative gradient-noise covariance")
        self.samples = state["samples"]
        self.signal = state["signal"]
        self.covariance = state["covariance"]
        self.last_probe_step = state["last_probe_step"]
        self.pending = None


class GradientNoiseProbe:
    """Observe local accumulated gradients before DDP reducer node hooks run.

    Tensor post-accumulate hooks only read gradients. The native DDP reducer,
    communication hooks, normalization and optimizer update remain untouched.
    Hooks exist only during the last microbatch of a sampled update. Temporary
    retained storage is one scalar per parameter, with per-parameter FP32
    squaring temporaries. It includes already accumulated gradients
    for parameters unused by that final microbatch.
    """

    def __init__(self):
        self.norms = {}
        self.handles = []
        self.active = False
        self.invalid = False

    def _observe(self, index: int, parameter: nn.Parameter) -> None:
        if parameter.grad is None:
            return
        if parameter.grad.is_sparse or parameter.grad.is_complex():
            self.invalid = True
            return
        self.norms[index] = squared_gradient_norm(parameter)

    def start(self, model: nn.Module) -> None:
        self.close()
        self.active = True
        self.invalid = False
        for index, parameter in enumerate(model.parameters()):
            if not parameter.requires_grad:
                continue
            self._observe(index, parameter)
            self.handles.append(
                parameter.register_post_accumulate_grad_hook(
                    lambda value, index=index: self._observe(index, value)
                )
            )

    @torch.no_grad()
    def finish(
        self,
        context,
        local_count: float,
        global_count: float,
        tracker: GradientNoiseTracker,
    ) -> None:
        if not self.active:
            return
        device = next(context.model.parameters()).device
        local_norm = (
            torch.stack(list(self.norms.values())).sum()
            if self.norms
            else torch.zeros((), dtype=torch.float64, device=device)
        )
        stats = torch.stack(
            (
                local_norm / local_count if local_count > 0 else local_norm * 0,
                torch.tensor(
                    float(local_count > 0), dtype=torch.float64, device=device
                ),
                torch.tensor(float(self.invalid), dtype=torch.float64, device=device),
            )
        )
        self.close()
        context.topology.reduce_sum(stats)
        weighted, ranks, invalid = stats.tolist()
        if invalid:
            tracker.pending = None
            return
        global_norms = [
            squared_gradient_norm(p)
            for p in context.model.parameters()
            if p.grad is not None
        ]
        global_norm = (
            float(torch.stack(global_norms).sum().item()) if global_norms else 0.0
        )
        tracker.stage(
            context.optimizer_step + 1, weighted, global_norm, global_count, int(ranks)
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.norms.clear()
        self.active = False
