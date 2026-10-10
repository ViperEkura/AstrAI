"""Torch reference attention backend."""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from . import ATTN_BACKEND, AttentionBackend, AttentionBackendFactory, repeat_kv

if TYPE_CHECKING:
    from astrai.model.kv_cache import KVCache


def _mask_view(mask: Optional[Tensor]) -> Optional[Tensor]:
    """Attention masks use batch/key, batch/query/key or batch/head/query/key."""
    if mask is None:
        return None
    if mask.dtype != torch.bool and not mask.is_floating_point():
        raise ValueError("attention mask must be boolean or floating point")
    if mask.ndim == 2:
        return mask[:, None, None, :]
    if mask.ndim == 3:
        return mask[:, None, :, :]
    if mask.ndim != 4:
        raise ValueError("attention mask must be 2D, 3D or 4D")
    return mask


def _causal_mask(q_len: int, kv_len: int, device: torch.device) -> Tensor:
    q_pos = torch.arange(q_len, device=device) + kv_len - q_len
    k_pos = torch.arange(kv_len, device=device)
    return q_pos[:, None] >= k_pos[None, :]


def _combine_mask(allowed: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return allowed
    if mask.dtype == torch.bool:
        return allowed & mask
    return mask.masked_fill(~allowed, float("-inf"))


def _packed_mask(mask: Optional[Tensor], allowed: Tensor, max_q: int) -> Tensor:
    """Apply the paged mask's broadcast and out-of-extent rules."""
    if mask is None:
        return allowed
    max_kv = allowed.size(-1)
    missing = False if mask.dtype == torch.bool else float("-inf")
    mask_rows = mask.size(-2)
    mask = mask[..., :max_q, :max_kv]
    if mask_rows != 1 and mask.size(-2) < max_q:
        mask = F.pad(mask, (0, 0, 0, max_q - mask.size(-2)), value=missing)
    if mask.size(-1) < max_kv:
        mask = F.pad(mask, (0, max_kv - mask.size(-1)), value=missing)
    return _combine_mask(allowed, mask)


@AttentionBackendFactory.register(ATTN_BACKEND.TORCH_NATIVE.value)
class TorchNativeBackend(AttentionBackend):
    """Reference backend using torch SDPA with indirect KV cache indexing.

    Writes new K/V into the cache buffers, gathers the full sequence K/V
    via ``req_to_token`` indirect indexing, then calls
    ``F.scaled_dot_product_attention``.

    Packed inference (3-D q) pads the ragged batch to [B, max_q, max_kv]
    and runs a single batched SDPA call with a combined causal+padding
    mask, then unpacks back to the flat layout.

    For training (``kv_cache is None``), skips cache I/O entirely and
    runs SDPA directly on the projected q/k/v.
    """

    supports_scale = True
    priority = 100

    @classmethod
    def available(cls) -> bool:
        return True

    def supports_call(
        self,
        q: Tensor,
        kv_cache: Optional["KVCache"],
        attn_mask: Optional[Tensor],
        is_causal: bool,
        fwd: Optional[str],
        *,
        scale: Optional[float] = None,
    ) -> bool:
        return True

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        fwd: Optional[str] = None,
        *,
        scale: Optional[float] = None,
    ) -> Tensor:
        self._check_fwd(fwd)
        if q.ndim == 4:
            n_rep = q.size(2) // k.size(2)
            if n_rep > 1:
                k = repeat_kv(k, n_rep)
                v = repeat_kv(v, n_rep)
            mask = _mask_view(attn_mask)
            sdpa_causal = is_causal
            if is_causal and (mask is not None or q.size(1) != k.size(1)):
                causal = _causal_mask(q.size(1), k.size(1), q.device)
                mask = _combine_mask(causal, mask)
                sdpa_causal = False
            return (
                F.scaled_dot_product_attention(
                    q.permute(0, 2, 1, 3),
                    k.permute(0, 2, 1, 3),
                    v.permute(0, 2, 1, 3),
                    mask,
                    is_causal=sdpa_causal,
                    scale=scale,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )

        if kv_cache is None or kv_cache.qo_indptr is None:
            raise ValueError("packed attention requires KV cache metadata")
        mask_view = _mask_view(attn_mask)
        if mask_view is not None:
            if mask_view.device != q.device:
                raise ValueError("packed mask must be on Q's device")
            if mask_view.size(0) not in (1, kv_cache.seq_lens.size(0)):
                raise ValueError("packed mask batch mismatch")
            if mask_view.size(1) not in (1, q.size(1)):
                raise ValueError("packed mask head mismatch")
            max_mask_rows = 1 if fwd == "decode" else q.size(0)
            if mask_view.size(-2) < 1 or mask_view.size(-2) > max_mask_rows:
                raise ValueError("packed mask query axis must use request-local rows")
            if not 0 < mask_view.size(-1) <= kv_cache.req_to_token.size(1):
                raise ValueError("packed mask key axis exceeds cache capacity")
        kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
        kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v

        # Pad the ragged batch to [B, max_q, max_kv] so one batched SDPA call
        # replaces B per-request calls. The bool mask folds the per-request
        # causal offset (seq_len - q_len) and the kv padding together; padded
        # q rows gather slot/row 0 and are dropped by the [q_valid] unpack,
        # which restores the packed qo_indptr order.
        qo_indptr = kv_cache.qo_indptr
        q_lens = qo_indptr[1:] - qo_indptr[:-1]
        seq_lens = kv_cache.seq_lens
        max_q = int(q_lens.max())
        max_kv = int(seq_lens.max())

        pos = torch.arange(max_kv, device=q.device)
        kv_valid = pos.unsqueeze(0) < seq_lens.unsqueeze(1)
        token_index = kv_cache.req_to_token[kv_cache.req_pool_indices, :max_kv]
        token_index = token_index.masked_fill(~kv_valid, 0)
        k_b = kv_cache.k_buffer[layer_id, token_index]
        v_b = kv_cache.v_buffer[layer_id, token_index]
        n_rep = q.size(1) // k.size(1)
        if n_rep > 1:
            k_b = repeat_kv(k_b, n_rep)
            v_b = repeat_kv(v_b, n_rep)

        q_pos = torch.arange(max_q, device=q.device)
        q_valid = q_pos.unsqueeze(0) < q_lens.unsqueeze(1)
        q_index = (qo_indptr[:-1].unsqueeze(1) + q_pos.unsqueeze(0)).masked_fill(
            ~q_valid, 0
        )
        causal = (
            (seq_lens - q_lens).unsqueeze(1).unsqueeze(2) + q_pos.view(1, max_q, 1)
        ) >= pos.view(1, 1, max_kv)
        allowed = kv_valid[:, None, None, :]
        if is_causal:
            allowed = allowed & causal.unsqueeze(1)
        mask = _packed_mask(mask_view, allowed, max_q)

        out = F.scaled_dot_product_attention(
            q[q_index].permute(0, 2, 1, 3),
            k_b.permute(0, 2, 1, 3),
            v_b.permute(0, 2, 1, 3),
            attn_mask=mask,
            scale=scale,
        )
        return out.permute(0, 2, 1, 3)[q_valid]
