from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.kernel.cross_entropy import is_available, linear_cross_entropy
from astrai.model.components.initialization import should_initialize


class Linear(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, bias: bool = False, init_std: float = 0.02
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.init_std = init_std
        # A module must never hand out the bytes torch.empty left behind;
        # models built through AutoRegressiveLM re-init via apply(), but
        # standalone components have no such pass.
        self.reset_parameters()

    def reset_parameters(self):
        if not should_initialize():
            return
        nn.init.normal_(self.weight, mean=0.0, std=self.init_std)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearCrossEntropyHead(Linear):
    """Language-model head that can compute CE inside its FSDP forward."""

    def forward(
        self,
        x: Tensor,
        targets: Optional[Tensor] = None,
        *,
        label_smoothing: float = 0.0,
        chunk_size: int = 512,
    ) -> Tensor:
        if targets is None:
            return super().forward(x)
        if targets.shape != x.shape[:-1]:
            raise ValueError("targets must match the training token shape")

        if self.bias is None and x.is_cuda and is_available():
            return linear_cross_entropy(
                x.reshape(-1, x.shape[-1]),
                self.weight,
                targets.reshape(-1),
                label_smoothing=label_smoothing,
                chunk_size=chunk_size,
            )
        logits = super().forward(x)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            label_smoothing=label_smoothing,
            reduction="sum",
        )
