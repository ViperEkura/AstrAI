"""Stateless wrappers around compiled extension kernels."""

from astrai.extension.kernel.attention import (
    TensorLayout,
    attn_decode,
    attn_paged_decode,
    attn_paged_prefill,
    attn_prefill,
)
from astrai.extension.kernel.cross_entropy import cross_entropy
from astrai.extension.kernel.muon_ns import muon_ns
from astrai.extension.kernel.rotary import rotary_emb

__all__ = [
    "TensorLayout",
    "attn_decode",
    "attn_paged_decode",
    "attn_paged_prefill",
    "attn_prefill",
    "cross_entropy",
    "muon_ns",
    "rotary_emb",
]
