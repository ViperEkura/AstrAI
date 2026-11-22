"""Optional FlashAttention dense forward."""

from typing import Optional

import torch
from torch import Tensor

from astrai.extension.backend.attention.common import (
    AttentionSpec,
    _flash_attn,
    flash_attn_available,
)


def supports(spec: AttentionSpec) -> bool:
    return (
        spec.device.type == "cuda"
        and spec.dtype in (torch.float16, torch.bfloat16)
        and spec.mask_kind == "none"
        and flash_attn_available()
    )


def forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    if attn_mask is not None:
        raise ValueError(
            "Flash attention cannot handle a custom attention mask; "
            "use a causal mask or select Torch attention."
        )
    fa = _flash_attn
    if fa is None:
        raise RuntimeError(
            "Flash attention requires the optional 'flash-attn' package. "
            "Install with `pip install flash-attn`."
        )
    out = fa.flash_attn_func(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        causal=is_causal,
        softmax_scale=scale,
    )
    return out.contiguous()
