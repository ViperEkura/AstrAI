"""Attention over paired segments in contiguous THD token storage."""

import math
from typing import Optional

import torch
from torch import Tensor

from astrai.extension.backend.attention import torch as reference
from astrai.extension.runtime.context import op_context
from astrai.extension.runtime.loader import get_module, is_available
from astrai.extension.runtime.operation import ExplicitSelectionError


def attention_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Optional[Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """Attend within each paired segment, using bottom-right causal alignment."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError("attention_varlen supports forward only")
    if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape:
        raise ValueError("expected THD inputs with matching K/V shapes")
    if (
        k.size(1) == 0
        or q.size(1) == 0
        or q.size(1) % k.size(1)
        or q.size(2) != k.size(2)
        or q.size(2) == 0
        or not q.is_floating_point()
        or any(t.dtype != q.dtype or t.device != q.device for t in (k, v))
    ):
        raise ValueError("incompatible varlen heads, dtype or device")
    qs = cu_seqlens_q
    ks = qs if cu_seqlens_k is None else cu_seqlens_k
    if cu_seqlens_k is None and q.size(0) != k.size(0):
        raise ValueError("shared offsets require equal Q and KV token counts")
    for ptr in (qs, ks):
        if (
            ptr.device != q.device
            or ptr.dtype != torch.int32
            or ptr.ndim != 1
            or ptr.numel() < 2
            or not ptr.is_contiguous()
        ):
            raise ValueError(
                "expected contiguous int32 segment offsets on the token device"
            )
    if qs.numel() != ks.numel():
        raise ValueError("Q and KV segment counts must match")
    scale = q.size(-1) ** -0.5 if scale is None else scale
    if not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
    choice, origin = op_context.snapshot().choice("attention")
    capable = (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.size(-1) in (32, 64, 128, 256)
        and scale > 0
        and is_available("attention")
    )
    if capable and choice in (None, "auto", "cuda"):
        return get_module("attention").attention_varlen(
            q, k, v, qs, ks, is_causal, scale
        )
    if origin == "context" and choice not in ("auto", "torch", "torch_native"):
        raise ExplicitSelectionError(
            f"{choice!r} cannot handle this varlen attention call"
        )
    qoffs, koffs = qs.tolist(), ks.tolist()
    for offsets, tokens in ((qoffs, q.size(0)), (koffs, k.size(0))):
        if (
            offsets[0] != 0
            or offsets[-1] != tokens
            or any(a > b for a, b in zip(offsets, offsets[1:]))
        ):
            raise ValueError("invalid segment offsets")
    result = []
    for i in range(len(qoffs) - 1):
        qr = q[qoffs[i] : qoffs[i + 1]]
        kr = k[koffs[i] : koffs[i + 1]]
        vr = v[koffs[i] : koffs[i + 1]]
        if not qr.size(0) or not kr.size(0):
            result.append(torch.zeros_like(qr))
        else:
            result.append(
                reference.forward(
                    qr[None],
                    kr[None],
                    vr[None],
                    is_causal=is_causal,
                    scale=scale,
                )[0]
            )
    return torch.cat(result)
