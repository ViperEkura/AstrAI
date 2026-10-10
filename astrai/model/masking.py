"""Mask contracts for full and recurrent decoder attention."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class LayerAttentionMask:
    tensor: Optional[Tensor]
    is_causal: bool


def process_attention_mask(
    input_mask: Optional[Tensor],
    *,
    causal: bool = False,
) -> Optional[Tensor]:
    """Expand masks, optionally adding causality to a 2-D key-padding mask.

    Explicit 3-D/4-D masks already define which query/key pairs may attend.
    """
    if input_mask is None:
        return None
    if input_mask.dim() == 2:
        mask = input_mask[:, None, None, :]
        if causal:
            seq_len = input_mask.size(-1)
            causal_mask = torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=input_mask.device
            ).tril()
            if input_mask.dtype == torch.bool or not input_mask.is_floating_point():
                mask = mask & causal_mask
            else:
                # Preserve additive attention biases on allowed keys.
                mask = mask.expand(-1, 1, seq_len, -1).masked_fill(
                    ~causal_mask, float("-inf")
                )
        return mask
    if input_mask.dim() == 3:
        return input_mask[:, None, :, :]
    return input_mask


def prepare_decoder_masks(
    input_mask: Optional[Tensor],
    layer_types: tuple[str, ...],
) -> dict[str, LayerAttentionMask]:
    """Prepare one mask per attention type, shared by all matching layers."""
    kinds = set(layer_types)
    result: dict[str, LayerAttentionMask] = {}
    full_attention = kinds & {"gqa", "mla"}
    if full_attention:
        # Keep a 2-D padding mask on its key axis. The attention backend applies
        # causality using each request's query position, including cached chunks.
        key_padding = input_mask is not None and input_mask.ndim == 2
        full = LayerAttentionMask(
            process_attention_mask(input_mask),
            is_causal=input_mask is None or key_padding,
        )
        for kind in full_attention:
            result[kind] = full
    if "gdn" in kinds:
        if input_mask is not None:
            if input_mask.ndim != 2:
                raise ValueError(
                    "Packed document masks need GDN state resets; train with "
                    "unpacked sequences until boundary resets are supported"
                )
            if input_mask.dtype != torch.bool:
                raise ValueError("GDN requires a boolean 2D right-padding mask")
        result["gdn"] = LayerAttentionMask(input_mask, is_causal=True)
    return result
