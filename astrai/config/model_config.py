from dataclasses import dataclass
from typing import Any, Dict, Optional

from astrai.config.base import BaseConfig
from astrai.factory import BaseFactory


class ConfigFactory(BaseFactory[BaseConfig]):
    """Factory that dispatches config classes by ``model_type``."""

    @classmethod
    def load(cls, raw: Dict[str, Any]) -> BaseConfig:
        model_type = raw.get("model_type") or "autoregressive_lm"
        config_cls = cls.get_component_class(model_type)
        return config_cls.from_dict(raw)


@dataclass
class BaseModelConfig(BaseConfig):
    """Base config with ``model_type`` dispatch and file I/O."""

    model_type: Optional[str] = None


@dataclass
@ConfigFactory.register("autoregressive_lm")
class AutoRegressiveLMConfig(BaseModelConfig):
    """Configuration for autoregressive language model."""

    vocab_size: Optional[int] = None
    dim: Optional[int] = None
    n_layers: Optional[int] = None
    norm_eps: Optional[float] = None
    dim_ffn: Optional[int] = None
    tie_weight: Optional[bool] = None

    max_len: Optional[int] = None
    rope_theta: Optional[float] = None
    rope_scaling: Optional[dict] = None

    attn_type: str = "gqa"
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    use_qk_norm: Optional[bool] = None
    use_gated_attention: Optional[bool] = None

    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None

    ffn_type: str = "mlp"
    n_routed_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None
    n_activated_experts: Optional[int] = None
    topk_method: Optional[str] = None


@dataclass
@ConfigFactory.register("embedding")
class EncoderConfig(BaseModelConfig):
    """Configuration for embedding encoder model."""

    vocab_size: Optional[int] = None
    dim: Optional[int] = None
    n_layers: Optional[int] = None
    norm_eps: Optional[float] = None
    dim_ffn: Optional[int] = None

    max_len: Optional[int] = None
    rope_theta: Optional[float] = None
    rope_scaling: Optional[dict] = None

    attn_type: str = "gqa"
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None
    use_qk_norm: Optional[bool] = None
    use_gated_attention: Optional[bool] = None

    ffn_type: str = "mlp"
    pooling_type: Optional[str] = None
    normalize_embeddings: Optional[bool] = None
