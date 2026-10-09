"""
AutoModel base class for model loading and saving.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Union

import torch.nn as nn

from astrai.config.model_config import BaseModelConfig, ConfigFactory
from astrai.factory import BaseFactory
from astrai.serialization import (
    adapt_config,
    load_hf_mapping,
    load_model_config,
    load_model_weights,
    save_model,
)
from astrai.serialization.pretrained import (
    load_pretrained_state_dict,
    prepare_pretrained_weights,
)


@contextmanager
def _disable_random_init(enable: bool = True):
    if not enable:
        yield
        return

    names = (
        "xavier_normal_",
        "xavier_uniform_",
        "kaiming_normal_",
        "kaiming_uniform_",
        "zeros_",
        "ones_",
        "constant_",
        "normal_",
        "uniform_",
    )
    orig = {n: getattr(nn.init, n) for n in names if hasattr(nn.init, n)}
    for n in orig:
        setattr(nn.init, n, lambda *a, **kw: None)
    try:
        yield
    finally:
        for n, fn in orig.items():
            setattr(nn.init, n, fn)


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
            disable_random_init: Replace parameter initializers with no-ops
                while building the model.
            strict: Require all policy tensors and reject unmapped HF keys.
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

        with _disable_random_init(enable=disable_random_init):
            model = actual_cls(config)

        weights_path = model_path / "model.safetensors"
        index_path = model_path / "model.safetensors.index.json"
        if weights_path.exists() or index_path.exists():
            state_dict = load_model_weights(str(model_path))
            state_dict = prepare_pretrained_weights(
                state_dict,
                config,
                mapping=mapping,
                weights_format=weights_format,
                strict=strict,
            )
            load_pretrained_state_dict(model, state_dict, strict=strict)

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
