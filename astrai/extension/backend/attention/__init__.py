"""Attention implementations and per-call dispatcher."""

from astrai.extension.backend.attention.common import AttentionSpec
from astrai.extension.backend.attention.dispatcher import (
    AttentionDispatcher,
    attention,
    paged_attention,
)
from astrai.extension.backend.attention.varlen import attention_varlen

__all__ = [
    "AttentionDispatcher",
    "AttentionSpec",
    "attention",
    "attention_varlen",
    "paged_attention",
]
