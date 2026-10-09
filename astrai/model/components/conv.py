"""AstrAI Conv1d with scoped parameter initialization."""

from math import sqrt
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from astrai.model.components.initialization import should_initialize


class Conv1d(nn.Module):
    """One-dimensional convolution with PyTorch-compatible parameter names."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, str] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or kernel_size <= 0:
            raise ValueError("channel counts and kernel_size must be positive")
        if groups <= 0 or in_channels % groups or out_channels % groups:
            raise ValueError("groups must divide both channel counts")
        if stride <= 0 or dilation <= 0:
            raise ValueError("stride and dilation must be positive")
        if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError(f"unsupported padding_mode: {padding_mode}")
        if isinstance(padding, str) and padding not in {"same", "valid"}:
            raise ValueError(f"unsupported padding: {padding}")
        if padding_mode != "zeros" and isinstance(padding, str):
            raise ValueError("nonzero padding_mode requires an integer padding")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,)
        self.stride = (stride,)
        self.padding = (padding,) if isinstance(padding, int) else padding
        self.dilation = (dilation,)
        self.groups = groups
        self.padding_mode = padding_mode

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(
                (out_channels, in_channels // groups, kernel_size), **factory_kwargs
            )
        )
        self.bias = (
            nn.Parameter(torch.empty(out_channels, **factory_kwargs)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not should_initialize():
            return
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2]
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            padding = self.padding[0]
            input = F.pad(input, (padding, padding), mode=self.padding_mode)
            padding = 0
        else:
            padding = self.padding
        return F.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )
