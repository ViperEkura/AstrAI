from typing import Any, Dict, Optional

from pydantic import field_validator, model_validator
from pydantic.dataclasses import dataclass

from astrai.config.base import BaseConfig
from astrai.factory import BaseFactory

_ATTN_TYPES = frozenset({"gqa", "mla"})
_FFN_TYPES = frozenset({"mlp", "moe"})


class ConfigFactory(BaseFactory[BaseConfig]):
    """Factory that dispatches config classes by ``model_type``."""

    @classmethod
    def load(cls, raw: Dict[str, Any]) -> BaseConfig:
        model_type = raw.get("model_type") or "autoregressive_lm"
        config_cls = cls.get_component_class(model_type)
        return config_cls.from_dict(raw)


@dataclass
class BaseModelConfig(BaseConfig):
    """Base config with ``model_type`` dispatch and file I/O.

    Args:
        model_type (Optional[str]): Model type identifier for AutoModel dispatch. Defaults to None.
        neftune_alpha (float): NEFTune noise alpha, 0=disabled, typical: 5.0. Defaults to 0.0.
    """

    model_type: Optional[str] = None
    neftune_alpha: float = 0.0


@dataclass
@ConfigFactory.register("autoregressive_lm")
class AutoRegressiveLMConfig(BaseModelConfig):
    """Configuration for autoregressive language model.

    Args:
        model_type (Optional[str]): Model type identifier for AutoModel dispatch. Defaults to None.
        neftune_alpha (float): NEFTune noise alpha, 0=disabled, typical: 5.0. Defaults to 0.0.
        vocab_size (Optional[int]): Vocabulary size. Defaults to None.
        hidden_size (Optional[int]): Hidden dimension size. Defaults to None.
        num_hidden_layers (Optional[int]): Number of transformer layers. Defaults to None.
        rms_norm_eps (Optional[float]): Epsilon for RMSNorm. Defaults to None.
        intermediate_size (Optional[int]): Intermediate size in FFN. Defaults to None.
        tie_word_embeddings (Optional[bool]): Whether to tie embedding and lm_head weights. Defaults to None.
        max_position_embeddings (Optional[int]): Maximum sequence length the model was trained with. Defaults to None.
        rope_theta (Optional[float]): Base frequency for RoPE. Defaults to None.
        rope_scaling (Optional[dict]): RoPE scaling config, e.g. {"type": "linear", "factor": 4.0}. Defaults to None.
        attn_type (str): Attention type: 'gqa' or 'mla'. Defaults to "gqa".
        num_attention_heads (Optional[int]): Number of query attention heads. Defaults to None.
        num_key_value_heads (Optional[int]): Number of key/value heads for GQA. Defaults to None.
        use_qk_norm (Optional[bool]): Whether to apply RMSNorm to Q/K. Defaults to None.
        use_gated_attention (Optional[bool]): Whether to use gated attention. Defaults to None.
        kv_lora_rank (Optional[int]): KV compression rank, MLA only. Defaults to None.
        qk_nope_head_dim (Optional[int]): Non-RoPE head dimension, MLA only. Defaults to None.
        qk_rope_head_dim (Optional[int]): RoPE head dimension, MLA only. Defaults to None.
        ffn_type (str): FFN type: 'mlp' or 'moe'. Defaults to "mlp".
        n_routed_experts (Optional[int]): Number of routed experts, MoE only. Defaults to None.
        n_shared_experts (Optional[int]): Number of shared experts, MoE only. Defaults to None.
        n_activated_experts (Optional[int]): Number of activated experts per token, MoE only. Defaults to None.
        topk_method (Optional[str]): Top-k routing method, MoE only. Defaults to None.
        moe_intermediate_size (Optional[int]): Expert hidden dim, defaults to intermediate_size if None. MoE only.
        shared_expert_intermediate_size (Optional[int]): Shared expert hidden dim, defaults to intermediate_size if None. MoE only.
        norm_topk_prob (bool): Normalize top-k routing probabilities. Defaults to True.
        decoder_sparse_step (int): Frequency of MoE layers, 1=every layer. Defaults to 1.
        mlp_only_layers (Optional[list[int]]): Layer indices using dense MLP instead of MoE. Defaults to None.
    """

    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    rms_norm_eps: Optional[float] = None
    intermediate_size: Optional[int] = None
    tie_word_embeddings: Optional[bool] = None
    max_position_embeddings: Optional[int] = None
    rope_theta: Optional[float] = None
    rope_scaling: Optional[dict] = None
    attn_type: str = "gqa"
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
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
    moe_intermediate_size: Optional[int] = None
    shared_expert_intermediate_size: Optional[int] = None
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: Optional[list[int]] = None
    moe_aux_loss_coef: float = 0.01

    @field_validator("attn_type")
    def _validate_attn_type(cls, v: str) -> str:
        if v not in _ATTN_TYPES:
            raise ValueError(
                f"attn_type must be one of {sorted(_ATTN_TYPES)}, got {v!r}"
            )
        return v

    @field_validator("ffn_type")
    def _validate_ffn_type(cls, v: str) -> str:
        if v not in _FFN_TYPES:
            raise ValueError(f"ffn_type must be one of {sorted(_FFN_TYPES)}, got {v!r}")
        return v

    @field_validator("decoder_sparse_step")
    def _validate_decoder_sparse_step(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"decoder_sparse_step must be at least 1, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_moe_topology(self) -> "AutoRegressiveLMConfig":
        if self.ffn_type != "moe":
            return self

        if self.n_routed_experts is None or self.n_routed_experts <= 0:
            raise ValueError("n_routed_experts must be positive for MoE")
        if self.n_shared_experts is None or self.n_shared_experts < 0:
            raise ValueError("n_shared_experts must be non-negative for MoE")
        if self.n_activated_experts is None or self.n_activated_experts <= 0:
            raise ValueError("n_activated_experts must be positive for MoE")
        if self.n_activated_experts > self.n_routed_experts:
            raise ValueError("n_activated_experts cannot exceed n_routed_experts")
        if self.topk_method not in (None, "greedy"):
            raise ValueError(f"unsupported topk_method: {self.topk_method!r}")
        return self


@dataclass
@ConfigFactory.register("embedding")
class EncoderConfig(BaseModelConfig):
    """Configuration for embedding encoder model.

    Args:
        model_type (Optional[str]): Model type identifier for AutoModel dispatch. Defaults to None.
        neftune_alpha (float): NEFTune noise alpha, 0=disabled, typical: 5.0. Defaults to 0.0.
        vocab_size (Optional[int]): Vocabulary size. Defaults to None.
        hidden_size (Optional[int]): Hidden dimension size. Defaults to None.
        num_hidden_layers (Optional[int]): Number of transformer layers. Defaults to None.
        rms_norm_eps (Optional[float]): Epsilon for RMSNorm. Defaults to None.
        intermediate_size (Optional[int]): Intermediate size in FFN. Defaults to None.
        max_position_embeddings (Optional[int]): Maximum sequence length the model was trained with. Defaults to None.
        rope_theta (Optional[float]): Base frequency for RoPE. Defaults to None.
        rope_scaling (Optional[dict]): RoPE scaling config, e.g. {"type": "linear", "factor": 4.0}. Defaults to None.
        attn_type (str): Attention type: 'gqa' or 'mla'. Defaults to "gqa".
        num_attention_heads (Optional[int]): Number of query attention heads. Defaults to None.
        num_key_value_heads (Optional[int]): Number of key/value heads for GQA. Defaults to None.
        use_qk_norm (Optional[bool]): Whether to apply RMSNorm to Q/K. Defaults to None.
        use_gated_attention (Optional[bool]): Whether to use gated attention. Defaults to None.
        ffn_type (str): FFN type: 'mlp' or 'moe'. Defaults to "mlp".
        pooling_type (Optional[str]): Pooling strategy for embedding, e.g. 'mean', 'cls'. Defaults to None.
        normalize_embeddings (Optional[bool]): Whether to L2-normalize output embeddings. Defaults to None.
    """

    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    rms_norm_eps: Optional[float] = None
    intermediate_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    rope_theta: Optional[float] = None
    rope_scaling: Optional[dict] = None
    attn_type: str = "gqa"
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    use_qk_norm: Optional[bool] = None
    use_gated_attention: Optional[bool] = None
    ffn_type: str = "mlp"
    pooling_type: Optional[str] = None
    normalize_embeddings: Optional[bool] = None

    @field_validator("attn_type")
    def _validate_attn_type(cls, v: str) -> str:
        if v not in _ATTN_TYPES:
            raise ValueError(
                f"attn_type must be one of {sorted(_ATTN_TYPES)}, got {v!r}"
            )
        return v

    @field_validator("ffn_type")
    def _validate_ffn_type(cls, v: str) -> str:
        if v not in _FFN_TYPES:
            raise ValueError(f"ffn_type must be one of {sorted(_FFN_TYPES)}, got {v!r}")
        return v
