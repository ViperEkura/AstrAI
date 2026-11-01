"""
AutoModel base class for model loading and saving.
"""

import logging
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn as nn

from astrai.config.model_config import BaseModelConfig, ConfigFactory
from astrai.factory import BaseFactory
from astrai.model.components.initialization import skip_parameter_init
from astrai.serialization import (
    adapt_config,
    convert_hf_weights,
    load_hf_mapping,
    load_model_config,
    load_model_weights,
    looks_like_hf_state_dict,
    save_model,
)

logger = logging.getLogger(__name__)


def validate_model_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    strict: bool = True,
    allowed_missing_keys: Collection[str] = (),
):
    """Check source coverage and shapes without changing model parameters."""
    expected = model.state_dict()
    targets = dict(model.named_parameters(remove_duplicate=False))
    targets.update(expected)
    missing = sorted(set(expected) - set(state_dict) - set(allowed_missing_keys))
    unexpected = sorted(set(state_dict) - set(targets))
    mismatched = [
        name
        for name in set(state_dict) & set(targets)
        if isinstance(targets[name], torch.Tensor)
        and (
            not isinstance(state_dict[name], torch.Tensor)
            or state_dict[name].shape != targets[name].shape
        )
    ]
    if mismatched or (strict and (missing or unexpected)):
        raise RuntimeError(
            "Model state dict mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
            f"shape_mismatch={sorted(mismatched)[:10]}"
        )
    return missing, unexpected


class ModelFactory(BaseFactory[nn.Module]):
    """Pure factory for model dispatch, separated from nn.Module state."""


class AutoModel(nn.Module):
    """Model base class with loading/saving and generation."""

    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        disable_random_init: bool = True,
        strict: bool = True,
        weights_format: str = "auto",
    ) -> nn.Module:
        """Load a model directory.

        Args:
            path: Directory containing ``config.json`` and optionally
                ``model.safetensors``.
            disable_random_init: Skip AstrAI parameter initialization when a
                complete checkpoint is loaded into a supported model.
            strict: Require complete model weights and reject unmapped HF keys.
                False permits missing/unexpected keys, but never shape mismatches.
            weights_format: ``"auto"`` reads a model-directory mapping and
                detects compatible HF weight keys; ``"astrai"`` skips conversion;
                ``"hf"`` requires a mapping and forces weight conversion.
        """
        if weights_format not in ("auto", "astrai", "hf"):
            raise ValueError(
                f"weights_format must be one of 'auto', 'astrai', 'hf', "
                f"got {weights_format!r}"
            )

        model_path = Path(path)

        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        raw = load_model_config(str(model_path))
        mapping = load_hf_mapping(model_path)
        is_hf_config = mapping is not None
        if weights_format == "hf" and mapping is None:
            raise FileNotFoundError(
                f"HF config mapping not found: {model_path / 'hf_mapping.json'}"
            )
        if is_hf_config:
            raw = adapt_config(raw, str(model_path))
        elif raw.get("model_type") not in (None, "autoregressive_lm"):
            raise FileNotFoundError(
                f"HF config mapping not found: {model_path / 'hf_mapping.json'}"
            )

        config = ConfigFactory.load(raw)
        model_type = config.model_type or "autoregressive_lm"

        actual_cls = ModelFactory.get_component_class(model_type)

        weights_path = model_path / "model.safetensors"
        index_path = model_path / "model.safetensors.index.json"
        has_weights = weights_path.exists() or index_path.exists()
        fast_load = bool(disable_random_init and strict and has_weights)
        with skip_parameter_init(fast_load):
            model = actual_cls(config)

        if not has_weights and strict:
            raise FileNotFoundError(f"No model weights found in {model_path}")
        if has_weights:
            state_dict = load_model_weights(str(model_path))
            if weights_format == "hf" or (
                weights_format == "auto" and looks_like_hf_state_dict(state_dict)
            ):
                state_dict = convert_hf_weights(
                    state_dict, config, mapping=mapping, strict=strict
                )
            missing, unexpected = validate_model_state_dict(
                model, state_dict, strict=strict
            )
            result = model.load_state_dict(state_dict, strict=strict)
            missing = sorted(set(missing) | set(result.missing_keys))
            unexpected = sorted(set(unexpected) | set(result.unexpected_keys))
            if missing or unexpected:
                logger.warning(
                    "Partial model loading: missing=%s, unexpected=%s",
                    missing[:10],
                    unexpected[:10],
                )

        return model

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
    ):
        save_model(
            config=self.config.to_dict(),
            state_dict=self.state_dict(),
            save_directory=str(save_directory),
        )
