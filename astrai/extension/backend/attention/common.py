"""Shared attention call metadata and Torch helpers."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch as _torch
from torch import Tensor

from astrai.extension.runtime.operation import OpSpec
from astrai.model.masking import PackedDocumentMask

try:
    import flash_attn as _flash_attn
except ImportError:
    _flash_attn = None

if TYPE_CHECKING:
    from astrai.model.kv_cache import KVCache


@functools.lru_cache(maxsize=1)
def flash_attn_available() -> bool:
    if not _torch.cuda.is_available() or _flash_attn is None:
        return False
    try:
        major = int(_flash_attn.__version__.split(".")[0])
        cc = _torch.cuda.get_device_capability()
        cc_num = cc[0] * 10 + cc[1]
        if (major >= 3 and cc_num < 90) or (major < 3 and cc_num < 70):
            return False
        x = _torch.zeros(1, 1, 1, 64, device="cuda", dtype=_torch.bfloat16)
        return bool(
            _torch.isfinite(_flash_attn.flash_attn_func(x, x, x, causal=True))
            .all()
            .item()
        )
    except Exception:
        return False


def _check_fwd(fwd: Optional[str]) -> None:
    if fwd not in (None, "prefill", "decode"):
        raise ValueError(f"unsupported attention forward mode: {fwd}")


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
    ndim: int
    mask_kind: str
    mask_ndim: int
    mask_last_stride: int
    kv_cache_kind: str
    is_causal: bool
    fwd: Optional[str]
    scale: Optional[float]

    @property
    def operation(self) -> str:
        return "attention"


def _attention_spec(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Optional["KVCache"],
    attn_mask: Optional[Tensor | PackedDocumentMask],
    is_causal: bool,
    fwd: Optional[str],
    scale: Optional[float] = None,
) -> AttentionSpec:
    return AttentionSpec(
        device=q.device,
        dtype=q.dtype,
        phase="train" if fwd is None else "infer",
        needs_backward=_torch.is_grad_enabled()
        and any(t.requires_grad for t in (q, k, v)),
        head_dim=q.size(-1),
        ndim=q.dim(),
        mask_kind="none"
        if attn_mask is None
        else "packed_doc"
        if isinstance(attn_mask, PackedDocumentMask)
        else str(attn_mask.dtype),
        mask_ndim=attn_mask.ndim if isinstance(attn_mask, Tensor) else 0,
        mask_last_stride=attn_mask.stride(-1) if isinstance(attn_mask, Tensor) else 0,
        kv_cache_kind="none" if kv_cache is None else type(kv_cache).__name__,
        is_causal=is_causal,
        fwd=fwd,
        scale=scale,
    )
