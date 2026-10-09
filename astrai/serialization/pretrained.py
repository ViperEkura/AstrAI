"""Shared weight conversion and coverage checks for pretrained policies."""

from collections.abc import Collection, Mapping
from typing import Any

import torch
from torch import nn

from astrai.config.base import BaseConfig
from astrai.serialization.hf_adapter import (
    convert_hf_weights,
    looks_like_hf_state_dict,
)


def prepare_pretrained_weights(
    state_dict: Mapping[str, torch.Tensor],
    config: BaseConfig,
    *,
    mapping: Mapping[str, Any] | None = None,
    weights_format: str = "auto",
    strict: bool = True,
) -> dict[str, torch.Tensor]:
    """Apply the model-owned HF mapping to either loading entry point."""
    if weights_format not in ("auto", "astrai", "hf"):
        raise ValueError(
            "weights_format must be one of 'auto', 'astrai', 'hf', "
            f"got {weights_format!r}"
        )
    if weights_format == "hf" or (
        weights_format == "auto" and looks_like_hf_state_dict(state_dict)
    ):
        return convert_hf_weights(state_dict, config, mapping=mapping, strict=strict)
    return dict(state_dict)


def load_pretrained_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
    allowed_missing_keys: Collection[str] = (),
):
    """Check required tensors before model loaders can synthesize defaults.

    The model's exported state defines its required tensors, including its
    tied-weight ownership. Callers may exempt explicitly created parameters
    (for example new LoRA adapters), but not a missing pretrained backbone.
    """
    allowed = set(allowed_missing_keys)
    missing = sorted(set(model.state_dict()) - set(state_dict) - allowed)
    if strict and missing:
        raise RuntimeError(
            f"Pretrained policy is missing {len(missing)} required tensor(s): "
            + ", ".join(missing[:10])
        )
    result = model.load_state_dict(state_dict, strict=False)
    # A permissive model loader may synthesize an untied head. Keep those
    # absent source tensors visible in the explicit warm-start warning.
    if not strict:
        result.missing_keys.extend(sorted(set(missing) - set(result.missing_keys)))
    missing = sorted(set(result.missing_keys) - allowed)
    if strict and (missing or result.unexpected_keys):
        raise RuntimeError(
            "Pretrained policy state dict mismatch: "
            f"missing={missing[:10]}, unexpected={result.unexpected_keys[:10]}"
        )
    return result
