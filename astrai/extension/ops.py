"""GQA attention wrapper functions — one entry point per compiled kernel.

Each wrapper dispatches to its CUDA kernel (loaded in ``loader.py``) when
available, otherwise falls back to ``torch`` SDPA.

Interface (all functions):
    causal_offset: -1 = non-causal; >=0 = absolute position of first Q token
    mask:       2D [batch, kv_len] or 3D [batch, q_len, kv_len] (bool)
    scale:      0.0 = auto (1/sqrt(head_dim)); >0 = explicit
    layout:     "bhld" (default) or "blhd"

Add new kernel wrappers here; split into per-variant files only if this file
grows large.
"""

import math

import torch
import torch.nn.functional as F

from astrai.extension.loader import _available, _modules

_LAYOUT_CODES: dict[str, int] = {"bhld": 0, "blhd": 1}


def _parse_layout(layout: str | int) -> int:
    if isinstance(layout, int):
        return layout
    code = _LAYOUT_CODES.get(layout.lower())
    if code is None:
        raise ValueError(
            f"unknown layout '{layout}', expected one of {list(_LAYOUT_CODES)}"
        )
    return code


def _to_bhld(t: torch.Tensor, layout: int) -> torch.Tensor:
    """Normalize to b h l d view. Zero-copy transpose if layout==1 (b l h d)."""
    if layout == 1:
        return t.transpose(1, 2)
    return t


def _expand_kv_heads(
    k: torch.Tensor, v: torch.Tensor, q_head: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand K/V heads to match Q heads for GQA fallback."""
    kv_head = k.size(1)
    if kv_head == q_head:
        return k, v
    group = q_head // kv_head
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    return k, v


def _build_attn_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor | None,
    causal_offset: int,
    scale: float,
) -> tuple[torch.Tensor | None, float]:
    """Build SDPA-compatible attn_mask + resolved scale.

    q and k must already be in b h l d layout.
    Causal and mask can coexist: causal sets -inf above the diagonal, mask
    sets -inf for padded positions. Both are OR'd into a single bool mask.
    """
    q_len = q.size(2)
    kv_len = k.size(2)
    head_dim = q.size(3)
    resolved_scale = scale if scale and scale > 0 else 1.0 / math.sqrt(head_dim)

    attn_mask = None

    if mask is not None:
        if mask.dim() == 2:
            # [batch, kv_len] → [batch, 1, 1, kv_len]
            attn_mask = mask[:, None, None, :]
        elif mask.dim() == 3:
            # [batch, q_len, kv_len] → [batch, 1, q_len, kv_len]
            attn_mask = mask[:, None, :, :]
        else:
            raise ValueError(f"mask must be 2D or 3D, got {mask.dim()}D")

    if causal_offset >= 0:
        batch = q.size(0)
        # q row i attends to kv cols 0..(causal_offset + i)
        q_idx = torch.arange(q_len, device=q.device).unsqueeze(1)  # [q_len, 1]
        kv_idx = torch.arange(kv_len, device=q.device).unsqueeze(0)  # [1, kv_len]
        causal_bool = kv_idx > (causal_offset + q_idx)  # True = masked out
        causal_mask = causal_bool.unsqueeze(0).expand(
            batch, -1, -1
        )  # [batch, q_len, kv_len]
        causal_mask = causal_mask[:, None, :, :]  # [batch, 1, q_len, kv_len]

        if attn_mask is not None:
            attn_mask = attn_mask | causal_mask
        else:
            attn_mask = causal_mask

    return attn_mask, resolved_scale


def _torch_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    causal_offset: int,
    scale: float,
    q_layout: int,
    kv_layout: int | None = None,
) -> torch.Tensor:
    """Reference attention via ``scaled_dot_product_attention``.

    q_layout / kv_layout: 0 = b h l d, 1 = b l h d.
    If kv_layout is None, uses q_layout (Q and K/V share the same layout).
    """
    if kv_layout is None:
        kv_layout = q_layout
    q = _to_bhld(q, q_layout)
    k = _to_bhld(k, kv_layout)
    v = _to_bhld(v, kv_layout)
    k, v = _expand_kv_heads(k, v, q.size(1))
    attn_mask, resolved_scale = _build_attn_mask(q, k, mask, causal_offset, scale)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=False, scale=resolved_scale
    )
    # Restore Q's original layout
    if q_layout == 1:
        out = out.transpose(1, 2)
    return out


def _gather_kv_from_pages(
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_size: int,
    kv_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather contiguous K/V from paged cache for torch SDPA fallback.

    Shapes:
        page_table : [batch, max_pages] (int64)
        k_cache    : [n_pages, page_size, n_kv_heads, head_dim]
        v_cache    : same as k_cache
    Returns:
        k, v : [batch, kv_len, n_kv_heads, head_dim]  (b l h d)
    """
    batch, max_pages = page_table.shape
    _, ps, n_kv_heads, head_dim = k_cache.shape
    if ps != page_size:
        raise ValueError(f"k_cache page_size mismatch: {ps} vs {page_size}")

    # Vectorized gather: build physical page + offset indices, then advanced-index
    positions = torch.arange(kv_len, device=page_table.device)
    logical_pages = positions // page_size  # [kv_len]
    page_offsets = positions % page_size  # [kv_len]

    phys_pages = page_table[:, logical_pages]  # [batch, kv_len]
    # k_cache[phys_pages, page_offsets] → [batch, kv_len, n_kv_heads, head_dim] (b l h d)
    k = k_cache[phys_pages, page_offsets]
    v = v_cache[phys_pages, page_offsets]
    return k, v


def attn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    causal_offset: int = -1,
    scale: float = 0.0,
    layout: str = "bhld",
) -> torch.Tensor:
    li = _parse_layout(layout)
    if _available["attn_decode"]:
        return _modules["attn_decode"].attn_decode(
            q,
            k,
            v,
            mask=mask,
            causal_offset=causal_offset,
            scale=scale,
            layout=li,
        )
    return _torch_fallback(q, k, v, mask, causal_offset, scale, q_layout=li)


def attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    causal_offset: int = -1,
    scale: float = 0.0,
    layout: str = "bhld",
) -> torch.Tensor:
    li = _parse_layout(layout)
    if _available["attn_prefill"]:
        return _modules["attn_prefill"].attn_prefill(
            q,
            k,
            v,
            mask=mask,
            causal_offset=causal_offset,
            scale=scale,
            layout=li,
        )
    return _torch_fallback(q, k, v, mask, causal_offset, scale, q_layout=li)


def attn_paged_decode(
    q: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_size: int,
    kv_len: int,
    mask: torch.Tensor | None = None,
    causal_offset: int = -1,
    scale: float = 0.0,
    layout: str = "bhld",
) -> torch.Tensor:
    li = _parse_layout(layout)
    if _available["attn_paged_decode"]:
        return _modules["attn_paged_decode"].attn_paged_decode(
            q,
            page_table,
            k_cache,
            v_cache,
            page_size,
            kv_len,
            mask=mask,
            causal_offset=causal_offset,
            scale=scale,
            layout=li,
        )
    # Gathered K/V are always b l h d
    k, v = _gather_kv_from_pages(page_table, k_cache, v_cache, page_size, kv_len)
    return _torch_fallback(
        q, k, v, mask, causal_offset, scale, q_layout=li, kv_layout=1
    )


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    causal_offset: int = -1,
    scale: float = 0.0,
    layout: str = "bhld",
) -> torch.Tensor:
    """Dispatch to decode or prefill attention based on the query length.

    A query length of one is the decode case; longer queries use prefill.
    The paged-cache decode path cannot be selected here because its page-table
    arguments are not part of this interface.
    """
    li = _parse_layout(layout)

    if q.ndim not in (2, 3, 4) or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError(
            "q, k, and v must all have the same rank in {2, 3, 4}, "
            f"got {q.ndim}D, {k.ndim}D, {v.ndim}D"
        )
    if k.shape != v.shape:
        raise ValueError(
            f"k and v must have the same shape, got {k.shape} and {v.shape}"
        )

    original_ndim = q.ndim
    if original_ndim == 2:
        # [L, D] -> [1, 1, L, D] or [1, L, 1, D]
        q = q.unsqueeze(0).unsqueeze(1 if li == 0 else 2)
        k = k.unsqueeze(0).unsqueeze(1 if li == 0 else 2)
        v = v.unsqueeze(0).unsqueeze(1 if li == 0 else 2)
    elif original_ndim == 3:
        # [B, L, D] -> single-head 4D input.
        q = q.unsqueeze(1 if li == 0 else 2)
        k = k.unsqueeze(1 if li == 0 else 2)
        v = v.unsqueeze(1 if li == 0 else 2)

    q_len = q.size(2 if li == 0 else 1)
    if q_len == 1:
        out = attn_decode(q, k, v, mask, causal_offset, scale, layout)
    else:
        out = attn_prefill(q, k, v, mask, causal_offset, scale, layout)

    if original_ndim == 2:
        return out.squeeze(0).squeeze(0 if li == 0 else 1)
    if original_ndim == 3:
        return out.squeeze(1 if li == 0 else 2)
    return out
