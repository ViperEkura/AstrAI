"""
AutoModel base class for model loading and saving.
"""

from pathlib import Path
from typing import Union

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
            strict: Passed to ``load_state_dict``.
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

        if has_weights:
            state_dict = load_model_weights(str(model_path))
            is_hf_weights = weights_format == "hf" or (
                weights_format == "auto" and looks_like_hf_state_dict(state_dict)
            )
            if is_hf_weights:
                state_dict = convert_hf_weights(state_dict, config, mapping=mapping)
            model.load_state_dict(state_dict, strict=strict)

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
