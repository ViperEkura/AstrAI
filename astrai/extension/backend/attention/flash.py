"""Optional FlashAttention dense forward."""

from typing import Optional

import torch
from torch import Tensor

from astrai.extension.backend.attention import torch as reference
from astrai.extension.backend.attention.common import (
    AttentionSpec,
    _flash_attn,
    flash_attn_available,
)
from astrai.extension.kernel import attention_validation as contract


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
    contract.dense(q, k, v, scale)
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


def forward_cache(
    q, kc, vc, k, v, past, pages, qs, mask, causal, workspace=(None, None, None)
):
    # FA2's paged path requires blocks in multiples of 256.
    if (
        qs is None
        and kc.ndim == 4
        and kc.size(1) % 256 == 0
        and (q.ndim == 3 or (q.ndim == 4 and q.size(1) == 1))
    ):
        q, k, v, qs, original = contract.cache(q, kc, vc, k, v, past, pages, qs)
        return _flash_attn.flash_attn_with_kvcache(
            q[:, None],
            kc,
            vc,
            k=k[:, None],
            v=v[:, None],
            cache_seqlens=past,
            block_table=pages,
            causal=causal,
        )[:, 0].reshape(original)
    return reference.forward_cache(
        q, kc, vc, k, v, past, pages, qs, mask, causal, dense=forward
    )
