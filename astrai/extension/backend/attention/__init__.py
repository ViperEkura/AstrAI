"""Attention implementations and per-call dispatcher."""

from astrai.extension.backend.attention.common import AttentionSpec
from astrai.extension.backend.attention.dispatcher import (
    AttentionDispatcher,
    attention,
)

__all__ = [
    "AttentionDispatcher",
    "AttentionSpec",
    "attention",
]
