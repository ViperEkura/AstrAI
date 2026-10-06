from dataclasses import asdict, field
from typing import Any, Dict, Optional

from pydantic import ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass
from pydantic_core import ArgsKwargs

from astrai.config.base import BaseConfig
from astrai.factory import BaseFactory

_ENCODER_ATTN_TYPES = frozenset({"gqa", "mla"})
_FFN_TYPES = frozenset({"mlp", "moe"})

ATTENTION_TYPES = frozenset({"gqa", "mla", "gdn"})


@dataclass(config=ConfigDict(extra="forbid"))
class GQAConfig:
    head_dim: Optional[int] = None
    rotary_dim: Optional[int] = None

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "GQAConfig":
        if self.head_dim is not None and self.head_dim < 1:
            raise ValueError("gqa.head_dim must be positive")
        if self.rotary_dim is not None:
            if self.rotary_dim < 1 or self.rotary_dim % 2:
                raise ValueError("gqa.rotary_dim must be a positive even number")
            if self.head_dim is not None and self.rotary_dim > self.head_dim:
                raise ValueError("gqa.rotary_dim cannot exceed gqa.head_dim")
        return self


@dataclass(config=ConfigDict(extra="forbid"))
class GDNConfig:
    num_key_heads: Optional[int] = None
    num_value_heads: Optional[int] = None
    key_head_dim: Optional[int] = None
    value_head_dim: Optional[int] = None
    conv_kernel_size: int = 4

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "GDNConfig":
        for name in (
            "num_key_heads",
            "num_value_heads",
            "key_head_dim",
            "value_head_dim",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(
                    f"gated deltanet dimensions must be positive, got {value}"
                )
        if self.conv_kernel_size < 1:
            raise ValueError("gdn.conv_kernel_size must be at least 1")
        return self


@dataclass(config=ConfigDict(extra="forbid"))
class MLAConfig:
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None


@dataclass(config=ConfigDict(extra="forbid"))
class AttentionConfig:
    default_type: str = "gqa"
    layers: Optional[list[str]] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    qk_norm: Optional[bool] = None
    output_gate: Optional[bool] = None
    gqa: GQAConfig = field(default_factory=GQAConfig)
    gdn: GDNConfig = field(default_factory=GDNConfig)
    mla: MLAConfig = field(default_factory=MLAConfig)

    @field_validator("default_type")
    def _validate_default_type(cls, value: str) -> str:
        if value not in ATTENTION_TYPES:
            raise ValueError(f"unsupported attention type: {value!r}")
        return value

    @field_validator("layers")
    def _validate_layers(cls, values: Optional[list[str]]) -> Optional[list[str]]:
        if values is not None and any(value not in ATTENTION_TYPES for value in values):
            raise ValueError("attention.layers contains an unsupported type")
        return values

    @model_validator(mode="after")
    def _validate_heads(self) -> "AttentionConfig":
        if self.num_heads is not None and self.num_heads < 1:
            raise ValueError("attention.num_heads must be positive")
        if self.num_kv_heads is not None and self.num_kv_heads < 1:
            raise ValueError("attention.num_kv_heads must be positive")
        if (
            self.num_heads is not None
            and self.num_kv_heads is not None
            and self.num_heads % self.num_kv_heads
        ):
            raise ValueError("attention.num_heads must divide by num_kv_heads")
        return self

    def type_for_layer(self, layer_id: int) -> str:
        return self.default_type if self.layers is None else self.layers[layer_id]

    def module_kwargs(self) -> dict:
        return {
            "n_heads": self.num_heads,
            "n_kv_heads": self.num_kv_heads,
            "use_qk_norm": self.qk_norm,
            "use_gated_attention": self.output_gate,
            "head_dim": self.gqa.head_dim,
            "rotary_dim": self.gqa.rotary_dim,
            "gdn_num_key_heads": self.gdn.num_key_heads,
            "gdn_num_value_heads": self.gdn.num_value_heads,
            "gdn_key_head_dim": self.gdn.key_head_dim,
            "gdn_value_head_dim": self.gdn.value_head_dim,
            "gdn_conv_kernel_size": self.gdn.conv_kernel_size,
            "kv_lora_rank": self.mla.kv_lora_rank,
            "qk_nope_head_dim": self.mla.qk_nope_head_dim,
            "qk_rope_head_dim": self.mla.qk_rope_head_dim,
        }


_LEGACY_ATTENTION_PATHS = {
    "attn_type": ("default_type",),
    "layer_types": ("layers",),
    "num_attention_heads": ("num_heads",),
    "num_key_value_heads": ("num_kv_heads",),
    "use_qk_norm": ("qk_norm",),
    "use_gated_attention": ("output_gate",),
    "head_dim": ("gqa", "head_dim"),
    "rotary_dim": ("gqa", "rotary_dim"),
    "gdn_num_key_heads": ("gdn", "num_key_heads"),
    "gdn_num_value_heads": ("gdn", "num_value_heads"),
    "gdn_key_head_dim": ("gdn", "key_head_dim"),
    "gdn_value_head_dim": ("gdn", "value_head_dim"),
    "gdn_conv_kernel_size": ("gdn", "conv_kernel_size"),
    "kv_lora_rank": ("mla", "kv_lora_rank"),
    "qk_nope_head_dim": ("mla", "qk_nope_head_dim"),
    "qk_rope_head_dim": ("mla", "qk_rope_head_dim"),
}


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
        attention (AttentionConfig): Per-layer attention topology and GQA/GDN/MLA settings.
        source_model_type (Optional[str]): Original HF model type when imported.
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
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    source_model_type: Optional[str] = None
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

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_attention(cls, data):
        if isinstance(data, ArgsKwargs):
            values = dict(data.kwargs or {})
            args = data.args
        elif isinstance(data, dict):
            values = dict(data)
            args = None
        else:
            return data

        attention = values.get("attention")
        if isinstance(attention, AttentionConfig):
            attention = asdict(attention)
        elif isinstance(attention, dict):
            attention = dict(attention)
        elif attention is None:
            attention = {}
        else:
            return data

        migrated = False
        for legacy_name, path in _LEGACY_ATTENTION_PATHS.items():
            if legacy_name not in values:
                continue
            value = values.pop(legacy_name)
            migrated = True
            if value is None:
                continue
            if legacy_name == "layer_types":
                names = {"full_attention": "gqa", "linear_attention": "gdn"}
                value = [names.get(name, name) for name in value]
            target = attention
            for part in path[:-1]:
                target = target.setdefault(part, {})
            existing = target.get(path[-1])
            if existing is not None and existing != value:
                raise ValueError(
                    f"attention.{'.'.join(path)} conflicts with {legacy_name}"
                )
            target[path[-1]] = value

        if migrated or "attention" in values:
            values["attention"] = attention
        if args is not None:
            return ArgsKwargs(args, values)
        return values

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
    def _validate_layer_topology(self) -> "AutoRegressiveLMConfig":
        attention = self.attention
        layers = attention.layers
        if layers is not None and self.num_hidden_layers is not None:
            if len(layers) != self.num_hidden_layers:
                raise ValueError("attention.layers length must match num_hidden_layers")
        if self.hidden_size is not None and attention.num_heads is not None:
            head_dim = attention.gqa.head_dim
            if head_dim is None and self.hidden_size % attention.num_heads:
                raise ValueError("hidden_size must be divisible by attention.num_heads")
            head_dim = head_dim or self.hidden_size // attention.num_heads
            if (
                attention.gqa.rotary_dim is not None
                and attention.gqa.rotary_dim > head_dim
            ):
                raise ValueError("attention.gqa.rotary_dim cannot exceed head_dim")
        return self

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

    # Read-only aliases for callers using the previous flat schema.
    @property
    def attn_type(self):
        return self.attention.default_type

    @property
    def layer_types(self):
        return self.attention.layers

    @property
    def num_attention_heads(self):
        return self.attention.num_heads

    @property
    def num_key_value_heads(self):
        return self.attention.num_kv_heads

    @property
    def use_qk_norm(self):
        return self.attention.qk_norm

    @property
    def use_gated_attention(self):
        return self.attention.output_gate

    @property
    def head_dim(self):
        return self.attention.gqa.head_dim

    @property
    def rotary_dim(self):
        return self.attention.gqa.rotary_dim

    @property
    def gdn_num_key_heads(self):
        return self.attention.gdn.num_key_heads

    @property
    def gdn_num_value_heads(self):
        return self.attention.gdn.num_value_heads

    @property
    def gdn_key_head_dim(self):
        return self.attention.gdn.key_head_dim

    @property
    def gdn_value_head_dim(self):
        return self.attention.gdn.value_head_dim

    @property
    def gdn_conv_kernel_size(self):
        return self.attention.gdn.conv_kernel_size

    @property
    def kv_lora_rank(self):
        return self.attention.mla.kv_lora_rank

    @property
    def qk_nope_head_dim(self):
        return self.attention.mla.qk_nope_head_dim

    @property
    def qk_rope_head_dim(self):
        return self.attention.mla.qk_rope_head_dim


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
        if v not in _ENCODER_ATTN_TYPES:
            raise ValueError(
                f"attn_type must be one of {sorted(_ENCODER_ATTN_TYPES)}, got {v!r}"
            )
        return v

    @field_validator("ffn_type")
    def _validate_ffn_type(cls, v: str) -> str:
        if v not in _FFN_TYPES:
            raise ValueError(f"ffn_type must be one of {sorted(_FFN_TYPES)}, got {v!r}")
        return v
