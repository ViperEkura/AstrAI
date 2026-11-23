"""Shared attention call metadata and Torch helpers."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from astrai.extension.runtime.operation import OpSpec

try:
    import flash_attn as _flash_attn
except ImportError:
    _flash_attn = None


@functools.lru_cache(maxsize=1)
def flash_attn_available() -> bool:
    if not torch.cuda.is_available() or _flash_attn is None:
        return False
    try:
        major = int(_flash_attn.__version__.split(".")[0])
        cc = torch.cuda.get_device_capability()
        cc_num = cc[0] * 10 + cc[1]
        if (major >= 3 and cc_num < 90) or (major < 3 and cc_num < 70):
            return False
        x = torch.zeros(1, 1, 1, 64, device="cuda", dtype=torch.bfloat16)
        return bool(
            torch.isfinite(_flash_attn.flash_attn_func(x, x, x, causal=True))
            .all()
            .item()
        )
    except Exception:
        return False


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return x
    n_heads, head_dim = x.shape[-2:]
    return (
        x.unsqueeze(-2)
        .expand(*x.shape[:-2], n_heads, n_rep, head_dim)
        .reshape(*x.shape[:-2], n_heads * n_rep, head_dim)
    )


@dataclass(frozen=True)
class AttentionSpec(OpSpec):
    head_dim: int
    mask_kind: str
    mask_last_stride: int
    is_causal: bool
    scale: Optional[float]
    mode: str = "dense"

    @property
    def operation(self) -> str:
        return "attention"


def _cache_view(q, k, v, qs):
    """Flatten regular cache inputs for reference backends."""
    original = q.shape
    if q.ndim == 4:
        if q.size(1) > 1:
            qs = torch.arange(
                q.size(0) + 1, device=q.device, dtype=torch.int32
            ) * q.size(1)
        q, k, v = (t.flatten(0, 1) for t in (q, k, v))
    return q, k, v, qs, original
