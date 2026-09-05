"""Versioned, compact, and fail-closed MoE route trace artifacts.

This module defines the data contract only. It does not capture routes from a
model, alter expert dispatch, or apply replay during training. The separation
keeps the default model path unchanged while rollout and training backends gain
a common artifact they can validate before later integrations consume it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import safetensors.torch as safetensors
import torch
from torch import Tensor

ROUTE_TRACE_SCHEMA_VERSION = 0

_CODEC_NAME = "astrai-route-trace-safetensors-raw-v0"
_MANIFEST_TENSOR = "__route_trace_manifest_v0__"
_MAX_MANIFEST_BYTES = 1024 * 1024
_DEFAULT_MAX_SERIALIZED_BYTES = 1024 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_DTYPE_BY_NAME = {
    "bool": torch.bool,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
}
_TENSOR_NAMES = ("topk_ids", "selected_weights", "topk_margin", "valid_mask")


class RouteTraceValidationError(ValueError):
    """A route trace cannot be interpreted under the declared contract."""


class RouteTraceLevel(str, Enum):
    """Payload tiers supported by the version-zero contract."""

    IDS = "ids"
    IDS_WEIGHTS_MARGIN = "ids_weights_margin"


class SelectedWeightSemantics(str, Enum):
    """How selected expert weights relate to the full router distribution."""

    FULL_SOFTMAX_SELECTED = "full-softmax-selected"
    SELECTED_RENORMALIZED = "selected-renormalized"


class PaddingLayout(str, Enum):
    """Validity-mask ordering for the flattened token span."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    EXPLICIT_MASK = "explicit-mask"


class TokenSpanKind(str, Enum):
    """Relationship of the captured span to prompt and response tokens."""

    FULL_SEQUENCE = "full-sequence"
    PROMPT = "prompt"
    RESPONSE = "response"
    CONTIGUOUS = "contiguous-span"


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RouteTraceValidationError(
            "value is not canonical-JSON serializable"
        ) from exc
    return encoded.encode("utf-8")


def canonical_json_digest(value: Any) -> str:
    """Return a content-addressed SHA-256 digest for JSON-compatible metadata."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_int(
    name: str, value: Any, *, minimum: int, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RouteTraceValidationError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise RouteTraceValidationError(f"{name} must be <= {maximum}")
    return value


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 1024:
        raise RouteTraceValidationError(f"{name} must be a non-empty bounded string")
    return value


def _require_digest(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise RouteTraceValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_exact_keys(
    value: Any, expected: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RouteTraceValidationError(f"{label} must be a mapping")
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        raise RouteTraceValidationError(
            f"{label} keys do not match version-zero schema; missing={missing}, unknown={unknown}"
        )
    return value


def _parse_enum(enum_type: type[Enum], name: str, value: Any) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise RouteTraceValidationError(f"unsupported {name}: {value!r}") from exc


@dataclass(frozen=True)
class RouterSchemaV0:
    """Semantic identity required to interpret route IDs and selected weights."""

    num_moe_layers: int
    num_experts: int
    top_k: int
    expert_parallel_world_size: int
    score_function: str
    score_dtype: str
    expert_id_ordering: str
    selected_weight_semantics: SelectedWeightSemantics
    token_ordering: str
    padding_layout: PaddingLayout
    backend: str
    kernel_semantics_version: str
    parallel_layout_hash: str

    def __post_init__(self) -> None:
        _require_int("num_moe_layers", self.num_moe_layers, minimum=1)
        _require_int("num_experts", self.num_experts, minimum=1, maximum=2**32)
        _require_int("top_k", self.top_k, minimum=1, maximum=self.num_experts)
        _require_int(
            "expert_parallel_world_size", self.expert_parallel_world_size, minimum=1
        )
        for name in (
            "score_function",
            "score_dtype",
            "expert_id_ordering",
            "token_ordering",
            "backend",
            "kernel_semantics_version",
        ):
            _require_text(name, getattr(self, name))
        if not isinstance(self.selected_weight_semantics, SelectedWeightSemantics):
            raise RouteTraceValidationError(
                "selected_weight_semantics must use the declared enum"
            )
        if not isinstance(self.padding_layout, PaddingLayout):
            raise RouteTraceValidationError("padding_layout must use the declared enum")
        _require_digest("parallel_layout_hash", self.parallel_layout_hash)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible schema representation."""
        return {
            "backend": self.backend,
            "expert_id_ordering": self.expert_id_ordering,
            "expert_parallel_world_size": self.expert_parallel_world_size,
            "kernel_semantics_version": self.kernel_semantics_version,
            "num_experts": self.num_experts,
            "num_moe_layers": self.num_moe_layers,
            "padding_layout": self.padding_layout.value,
            "parallel_layout_hash": self.parallel_layout_hash,
            "score_dtype": self.score_dtype,
            "score_function": self.score_function,
            "selected_weight_semantics": self.selected_weight_semantics.value,
            "token_ordering": self.token_ordering,
            "top_k": self.top_k,
        }

    @property
    def fingerprint(self) -> str:
        """Return the content address used as ``router_schema_hash``."""
        return canonical_json_digest(
            {"route_trace_schema_version": ROUTE_TRACE_SCHEMA_VERSION, **self.to_dict()}
        )

    @classmethod
    def from_dict(cls, value: Any) -> RouterSchemaV0:
        """Parse a strict version-zero router schema."""
        fields = {
            "backend",
            "expert_id_ordering",
            "expert_parallel_world_size",
            "kernel_semantics_version",
            "num_experts",
            "num_moe_layers",
            "padding_layout",
            "parallel_layout_hash",
            "score_dtype",
            "score_function",
            "selected_weight_semantics",
            "token_ordering",
            "top_k",
        }
        value = _require_exact_keys(value, fields, "router_schema")
        return cls(
            num_moe_layers=value["num_moe_layers"],
            num_experts=value["num_experts"],
            top_k=value["top_k"],
            expert_parallel_world_size=value["expert_parallel_world_size"],
            score_function=value["score_function"],
            score_dtype=value["score_dtype"],
            expert_id_ordering=value["expert_id_ordering"],
            selected_weight_semantics=_parse_enum(
                SelectedWeightSemantics,
                "selected_weight_semantics",
                value["selected_weight_semantics"],
            ),
            token_ordering=value["token_ordering"],
            padding_layout=_parse_enum(
                PaddingLayout, "padding_layout", value["padding_layout"]
            ),
            backend=value["backend"],
            kernel_semantics_version=value["kernel_semantics_version"],
            parallel_layout_hash=value["parallel_layout_hash"],
        )


@dataclass(frozen=True)
class RouteIdentityV0:
    """Versions and revisions that attribute a trace to one behavior policy."""

    policy_version: int
    model_revision: str
    router_state_version: int
    checkpoint_revision: str
    router_schema_hash: str
    load_controller_version: int = 0

    def __post_init__(self) -> None:
        _require_int("policy_version", self.policy_version, minimum=0)
        _require_int("router_state_version", self.router_state_version, minimum=0)
        _require_int("load_controller_version", self.load_controller_version, minimum=0)
        _require_text("model_revision", self.model_revision)
        _require_text("checkpoint_revision", self.checkpoint_revision)
        _require_digest("router_schema_hash", self.router_schema_hash)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible identity representation."""
        return {
            "checkpoint_revision": self.checkpoint_revision,
            "load_controller_version": self.load_controller_version,
            "model_revision": self.model_revision,
            "policy_version": self.policy_version,
            "router_schema_hash": self.router_schema_hash,
            "router_state_version": self.router_state_version,
        }

    @classmethod
    def from_dict(cls, value: Any) -> RouteIdentityV0:
        """Parse a strict version-zero route identity."""
        fields = {
            "checkpoint_revision",
            "load_controller_version",
            "model_revision",
            "policy_version",
            "router_schema_hash",
            "router_state_version",
        }
        value = _require_exact_keys(value, fields, "identity")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class RouteTokenLayoutV0:
    """One sample's flattened contiguous token span and prompt boundary."""

    sample_id: str
    sequence_token_count: int
    prompt_token_count: int
    token_offset: int
    token_count: int
    span_kind: TokenSpanKind

    def __post_init__(self) -> None:
        _require_text("sample_id", self.sample_id)
        _require_int("sequence_token_count", self.sequence_token_count, minimum=0)
        _require_int(
            "prompt_token_count",
            self.prompt_token_count,
            minimum=0,
            maximum=self.sequence_token_count,
        )
        _require_int("token_offset", self.token_offset, minimum=0)
        _require_int("token_count", self.token_count, minimum=0)
        if self.token_offset + self.token_count > self.sequence_token_count:
            raise RouteTraceValidationError(
                "captured token span exceeds the sequence boundary"
            )
        if not isinstance(self.span_kind, TokenSpanKind):
            raise RouteTraceValidationError("span_kind must use the declared enum")
        if self.span_kind is TokenSpanKind.FULL_SEQUENCE and (
            self.token_offset != 0 or self.token_count != self.sequence_token_count
        ):
            raise RouteTraceValidationError(
                "full-sequence span must cover the complete sequence"
            )
        if self.span_kind is TokenSpanKind.PROMPT and (
            self.token_offset != 0 or self.token_count != self.prompt_token_count
        ):
            raise RouteTraceValidationError(
                "prompt span must exactly cover the prompt boundary"
            )
        if (
            self.span_kind is TokenSpanKind.RESPONSE
            and self.token_offset < self.prompt_token_count
        ):
            raise RouteTraceValidationError(
                "response span cannot begin inside the prompt"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible token layout."""
        return {
            "prompt_token_count": self.prompt_token_count,
            "sample_id": self.sample_id,
            "sequence_token_count": self.sequence_token_count,
            "span_kind": self.span_kind.value,
            "token_count": self.token_count,
            "token_offset": self.token_offset,
        }

    @classmethod
    def from_dict(cls, value: Any) -> RouteTokenLayoutV0:
        """Parse a strict version-zero token layout."""
        fields = {
            "prompt_token_count",
            "sample_id",
            "sequence_token_count",
            "span_kind",
            "token_count",
            "token_offset",
        }
        value = _require_exact_keys(value, fields, "token_layout")
        return cls(
            sample_id=value["sample_id"],
            sequence_token_count=value["sequence_token_count"],
            prompt_token_count=value["prompt_token_count"],
            token_offset=value["token_offset"],
            token_count=value["token_count"],
            span_kind=_parse_enum(TokenSpanKind, "span_kind", value["span_kind"]),
        )


def recommended_route_id_dtype(num_experts: int) -> torch.dtype:
    """Return the narrowest unsigned dtype that represents every expert ID."""
    _require_int("num_experts", num_experts, minimum=1, maximum=2**32)
    if num_experts <= 2**8:
        return torch.uint8
    if num_experts <= 2**16:
        return torch.uint16
    return torch.uint32


def pack_route_ids(topk_ids: Tensor, num_experts: int) -> Tensor:
    """Validate expert IDs and copy them into the canonical compact dtype."""
    if not isinstance(topk_ids, Tensor):
        raise RouteTraceValidationError("topk_ids must be a torch.Tensor")
    integral_dtypes = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
    )
    if topk_ids.dtype not in integral_dtypes:
        raise RouteTraceValidationError("topk_ids must use an integer dtype")
    _require_int("num_experts", num_experts, minimum=1, maximum=2**32)
    ids64 = topk_ids.detach().to(dtype=torch.int64)
    if ids64.numel() and (ids64.min().item() < 0 or ids64.max().item() >= num_experts):
        raise RouteTraceValidationError(
            "topk_ids contains an expert outside the declared range"
        )
    return (
        topk_ids.detach()
        .to(dtype=recommended_route_id_dtype(num_experts), copy=True)
        .contiguous()
    )


@dataclass(frozen=True, eq=False)
class RouteTraceV0:
    """A compact route artifact for one contiguous token span.

    Tensors use token-major shapes ``[T, L, K]`` for IDs and weights,
    ``[T, L]`` for margins, and ``[T]`` for the optional validity mask. The
    constructor validates all identity, shape, dtype, range, and padding
    invariants. Consumers must validate again at their trust boundary because
    tensors remain mutable Python objects even though this dataclass is frozen.
    """

    identity: RouteIdentityV0
    router_schema: RouterSchemaV0
    token_layout: RouteTokenLayoutV0
    level: RouteTraceLevel
    topk_ids: Tensor
    selected_weights: Tensor | None = None
    topk_margin: Tensor | None = None
    valid_mask: Tensor | None = None

    def __post_init__(self) -> None:
        validate_route_trace(self)

    @property
    def valid_token_count(self) -> int:
        """Return the number of captured tokens marked valid."""
        if self.valid_mask is None:
            return self.token_layout.token_count
        return int(self.valid_mask.sum().item())

    @property
    def payload_nbytes(self) -> int:
        """Return exact tensor payload bytes, excluding the manifest/container."""
        tensors = (
            self.topk_ids,
            self.selected_weights,
            self.topk_margin,
            self.valid_mask,
        )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
            if tensor is not None
        )

    @property
    def artifact_digest(self) -> str:
        """Return the content address of the canonical serialized artifact."""
        return hashlib.sha256(RouteTraceCodecV0.dumps(self)).hexdigest()


def _effective_valid_mask(trace: RouteTraceV0) -> Tensor:
    if trace.valid_mask is not None:
        return trace.valid_mask
    return torch.ones(
        trace.token_layout.token_count, dtype=torch.bool, device=trace.topk_ids.device
    )


def _validate_padding_mask(trace: RouteTraceV0, mask: Tensor) -> None:
    layout = trace.router_schema.padding_layout
    if layout is PaddingLayout.EXPLICIT_MASK and trace.valid_mask is None:
        raise RouteTraceValidationError("explicit-mask layout requires valid_mask")
    if layout is PaddingLayout.NONE and mask.numel() and not bool(mask.all().item()):
        raise RouteTraceValidationError(
            "padding_layout=none cannot contain invalid tokens"
        )
    if mask.numel() < 2:
        return
    if layout is PaddingLayout.LEFT and bool((mask[:-1] & ~mask[1:]).any().item()):
        raise RouteTraceValidationError(
            "left padding mask must be false tokens followed by true tokens"
        )
    if layout is PaddingLayout.RIGHT and bool((~mask[:-1] & mask[1:]).any().item()):
        raise RouteTraceValidationError(
            "right padding mask must be true tokens followed by false tokens"
        )


def _validate_float_tensor(
    name: str, tensor: Tensor, expected_shape: tuple[int, ...], device: torch.device
) -> None:
    if not isinstance(tensor, Tensor):
        raise RouteTraceValidationError(f"{name} must be a torch.Tensor")
    if tensor.layout is not torch.strided or tensor.device.type == "meta":
        raise RouteTraceValidationError(f"{name} must be a materialized strided tensor")
    if tensor.dtype not in _FLOAT_DTYPES:
        raise RouteTraceValidationError(
            f"{name} must use float16, bfloat16, or float32"
        )
    if tuple(tensor.shape) != expected_shape:
        raise RouteTraceValidationError(
            f"{name} shape must be {expected_shape}, got {tuple(tensor.shape)}"
        )
    if tensor.device != device:
        raise RouteTraceValidationError(
            f"{name} must be on the same device as topk_ids"
        )
    if tensor.requires_grad:
        raise RouteTraceValidationError(f"{name} must be detached from autograd")
    negative_zero = (tensor == 0) & torch.signbit(tensor)
    if negative_zero.numel() and bool(negative_zero.any().item()):
        raise RouteTraceValidationError(
            f"{name} must use canonical positive zero values"
        )


def validate_route_trace(trace: RouteTraceV0) -> None:
    """Validate one trace's complete internal contract, failing closed."""
    if not isinstance(trace.identity, RouteIdentityV0):
        raise RouteTraceValidationError("identity must be RouteIdentityV0")
    if not isinstance(trace.router_schema, RouterSchemaV0):
        raise RouteTraceValidationError("router_schema must be RouterSchemaV0")
    if not isinstance(trace.token_layout, RouteTokenLayoutV0):
        raise RouteTraceValidationError("token_layout must be RouteTokenLayoutV0")
    if not isinstance(trace.level, RouteTraceLevel):
        raise RouteTraceValidationError("level must use the declared enum")
    if trace.identity.router_schema_hash != trace.router_schema.fingerprint:
        raise RouteTraceValidationError(
            "router_schema_hash does not match the declared router schema"
        )
    if not isinstance(trace.topk_ids, Tensor):
        raise RouteTraceValidationError("topk_ids must be a torch.Tensor")
    if (
        trace.topk_ids.layout is not torch.strided
        or trace.topk_ids.device.type == "meta"
    ):
        raise RouteTraceValidationError(
            "topk_ids must be a materialized strided tensor"
        )

    schema = trace.router_schema
    token_count = trace.token_layout.token_count
    expected_ids_shape = (token_count, schema.num_moe_layers, schema.top_k)
    if tuple(trace.topk_ids.shape) != expected_ids_shape:
        raise RouteTraceValidationError(
            f"topk_ids shape must be {expected_ids_shape}, got {tuple(trace.topk_ids.shape)}"
        )
    expected_id_dtype = recommended_route_id_dtype(schema.num_experts)
    if trace.topk_ids.dtype is not expected_id_dtype:
        raise RouteTraceValidationError(
            f"topk_ids must use compact dtype {expected_id_dtype}, got {trace.topk_ids.dtype}"
        )

    if trace.valid_mask is not None:
        if not isinstance(trace.valid_mask, Tensor):
            raise RouteTraceValidationError("valid_mask must be a torch.Tensor")
        if (
            trace.valid_mask.layout is not torch.strided
            or trace.valid_mask.device.type == "meta"
        ):
            raise RouteTraceValidationError(
                "valid_mask must be a materialized strided tensor"
            )
        if trace.valid_mask.dtype is not torch.bool or tuple(
            trace.valid_mask.shape
        ) != (token_count,):
            raise RouteTraceValidationError(
                f"valid_mask must be bool with shape {(token_count,)}"
            )
        if trace.valid_mask.device != trace.topk_ids.device:
            raise RouteTraceValidationError(
                "valid_mask must be on the same device as topk_ids"
            )
    mask = _effective_valid_mask(trace)
    _validate_padding_mask(trace, mask)

    ids64 = trace.topk_ids.to(dtype=torch.int64)
    if ids64.numel() and (
        ids64.min().item() < 0 or ids64.max().item() >= schema.num_experts
    ):
        raise RouteTraceValidationError(
            "topk_ids contains an expert outside the declared range"
        )
    valid_rows = ids64[mask]
    if schema.top_k > 1 and valid_rows.numel():
        ordered_ids = valid_rows.sort(dim=-1).values
        if bool((ordered_ids[..., 1:] == ordered_ids[..., :-1]).any().item()):
            raise RouteTraceValidationError(
                "topk_ids contains duplicate experts in a valid top-k row"
            )
    invalid_mask = ~mask
    if (
        invalid_mask.numel()
        and bool(invalid_mask.any().item())
        and bool(torch.count_nonzero(ids64[invalid_mask]).item())
    ):
        raise RouteTraceValidationError(
            "invalid token rows must use canonical zero expert IDs"
        )

    rich_level = trace.level is RouteTraceLevel.IDS_WEIGHTS_MARGIN
    if rich_level and (trace.selected_weights is None or trace.topk_margin is None):
        raise RouteTraceValidationError(
            "ids_weights_margin level requires weights and margins"
        )
    if not rich_level and (
        trace.selected_weights is not None or trace.topk_margin is not None
    ):
        raise RouteTraceValidationError("ids level cannot carry weights or margins")

    if trace.selected_weights is not None:
        _validate_float_tensor(
            "selected_weights",
            trace.selected_weights,
            expected_ids_shape,
            trace.topk_ids.device,
        )
        valid_weights = trace.selected_weights.float()[mask]
        if valid_weights.numel():
            if not bool(torch.isfinite(valid_weights).all().item()) or bool(
                (valid_weights < 0).any().item()
            ):
                raise RouteTraceValidationError(
                    "selected_weights must be finite and non-negative"
                )
            weight_sums = valid_weights.sum(dim=-1)
            if (
                schema.selected_weight_semantics
                is SelectedWeightSemantics.SELECTED_RENORMALIZED
            ):
                if not bool(
                    torch.allclose(
                        weight_sums, torch.ones_like(weight_sums), rtol=0, atol=5e-3
                    )
                ):
                    raise RouteTraceValidationError(
                        "renormalized selected weights must sum to one"
                    )
            elif bool(((weight_sums <= 0) | (weight_sums > 1.005)).any().item()):
                raise RouteTraceValidationError(
                    "full-softmax selected mass must be in (0, 1]"
                )
        if (
            invalid_mask.numel()
            and bool(invalid_mask.any().item())
            and bool(torch.count_nonzero(trace.selected_weights[invalid_mask]).item())
        ):
            raise RouteTraceValidationError(
                "invalid token rows must use canonical zero weights"
            )

    if trace.topk_margin is not None:
        expected_margin_shape = (token_count, schema.num_moe_layers)
        _validate_float_tensor(
            "topk_margin",
            trace.topk_margin,
            expected_margin_shape,
            trace.topk_ids.device,
        )
        valid_margins = trace.topk_margin.float()[mask]
        if valid_margins.numel() and (
            not bool(torch.isfinite(valid_margins).all().item())
            or bool((valid_margins < 0).any().item())
        ):
            raise RouteTraceValidationError(
                "topk_margin must be finite and non-negative"
            )
        if (
            invalid_mask.numel()
            and bool(invalid_mask.any().item())
            and bool(torch.count_nonzero(trace.topk_margin[invalid_mask]).item())
        ):
            raise RouteTraceValidationError(
                "invalid token rows must use canonical zero margins"
            )


def require_compatible_route_trace(
    trace: RouteTraceV0,
    *,
    expected_identity: RouteIdentityV0,
    expected_router_schema: RouterSchemaV0,
    expected_token_layout: RouteTokenLayoutV0,
    minimum_level: RouteTraceLevel = RouteTraceLevel.IDS,
) -> None:
    """Require exact replay identity and semantics at a consumer boundary."""
    validate_route_trace(trace)
    if trace.identity != expected_identity:
        raise RouteTraceValidationError(
            "route identity does not match the expected behavior artifact"
        )
    if trace.router_schema != expected_router_schema:
        raise RouteTraceValidationError(
            "router schema does not match the expected consumer schema"
        )
    if trace.token_layout != expected_token_layout:
        raise RouteTraceValidationError(
            "token layout does not match the expected consumer span"
        )
    if not isinstance(minimum_level, RouteTraceLevel):
        raise RouteTraceValidationError("minimum_level must use the declared enum")
    level_order = {RouteTraceLevel.IDS: 0, RouteTraceLevel.IDS_WEIGHTS_MARGIN: 1}
    if level_order[trace.level] < level_order[minimum_level]:
        raise RouteTraceValidationError(
            f"trace level {trace.level.value} does not satisfy {minimum_level.value}"
        )


def require_semantically_aligned_traces(
    behavior: RouteTraceV0, current: RouteTraceV0
) -> None:
    """Require two traces to share tensor interpretation and token alignment."""
    validate_route_trace(behavior)
    validate_route_trace(current)
    if behavior.router_schema != current.router_schema:
        raise RouteTraceValidationError(
            "route metrics require an exact router schema match"
        )
    if behavior.token_layout != current.token_layout:
        raise RouteTraceValidationError(
            "route metrics require an exact token layout match"
        )
    if behavior.topk_ids.device != current.topk_ids.device:
        raise RouteTraceValidationError(
            "route metrics require tensors on the same device"
        )
    behavior_mask = _effective_valid_mask(behavior)
    current_mask = _effective_valid_mask(current)
    if not torch.equal(behavior_mask, current_mask):
        raise RouteTraceValidationError(
            "route metrics require identical validity masks"
        )


def _tensor_descriptor(tensor: Tensor) -> tuple[dict[str, Any], Tensor]:
    if sys.byteorder != "little":
        raise RouteTraceValidationError(
            "route trace codec requires a little-endian host"
        )
    cpu_tensor = tensor.detach().contiguous().cpu()
    raw_tensor = cpu_tensor.view(torch.uint8).reshape(-1).clone()
    raw_bytes = raw_tensor.numpy().tobytes()
    descriptor = {
        "dtype": str(cpu_tensor.dtype).removeprefix("torch."),
        "nbytes": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "shape": list(cpu_tensor.shape),
    }
    return descriptor, raw_tensor


def _decode_tensor(name: str, descriptor: Any, raw_tensor: Tensor) -> Tensor:
    descriptor = _require_exact_keys(
        descriptor, {"dtype", "nbytes", "sha256", "shape"}, f"tensor descriptor {name}"
    )
    dtype_name = descriptor["dtype"]
    if dtype_name not in _DTYPE_BY_NAME:
        raise RouteTraceValidationError(
            f"unsupported tensor dtype for {name}: {dtype_name!r}"
        )
    shape = descriptor["shape"]
    if not isinstance(shape, list) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in shape
    ):
        raise RouteTraceValidationError(f"tensor shape for {name} is invalid")
    nbytes = _require_int(f"tensor nbytes for {name}", descriptor["nbytes"], minimum=0)
    _require_digest(f"tensor sha256 for {name}", descriptor["sha256"])
    if raw_tensor.dtype is not torch.uint8 or raw_tensor.ndim != 1:
        raise RouteTraceValidationError(
            f"raw tensor for {name} must be one-dimensional uint8"
        )
    if raw_tensor.numel() != nbytes:
        raise RouteTraceValidationError(
            f"raw tensor size for {name} does not match its descriptor"
        )
    raw_bytes = raw_tensor.numpy().tobytes()
    if hashlib.sha256(raw_bytes).hexdigest() != descriptor["sha256"]:
        raise RouteTraceValidationError(f"raw tensor checksum mismatch for {name}")
    dtype = _DTYPE_BY_NAME[dtype_name]
    expected_nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if expected_nbytes != nbytes:
        raise RouteTraceValidationError(
            f"logical tensor size for {name} does not match its descriptor"
        )
    return raw_tensor.view(dtype).reshape(tuple(shape)).clone()


class RouteTraceCodecV0:
    """Safe bytes codec backed by safetensors and a strict JSON manifest.

    Unsigned ID dtypes unsupported by older safetensors releases are stored as
    raw uint8 buffers with checked logical dtype, shape, byte count, and SHA-256
    descriptors. Decoding always returns detached CPU tensors and rejects
    unknown fields, tensors, versions, trailing schema changes, and corruption.
    """

    @staticmethod
    def dumps(trace: RouteTraceV0) -> bytes:
        """Serialize a validated trace without pickle or executable objects."""
        validate_route_trace(trace)
        manifest: dict[str, Any] = {
            "byte_order": "little",
            "codec": _CODEC_NAME,
            "identity": trace.identity.to_dict(),
            "level": trace.level.value,
            "route_trace_schema_version": ROUTE_TRACE_SCHEMA_VERSION,
            "router_schema": trace.router_schema.to_dict(),
            "tensors": {},
            "token_layout": trace.token_layout.to_dict(),
        }
        buffers: dict[str, Tensor] = {}
        for name in _TENSOR_NAMES:
            tensor = getattr(trace, name)
            if tensor is None:
                continue
            descriptor, raw_tensor = _tensor_descriptor(tensor)
            manifest["tensors"][name] = descriptor
            buffers[f"tensor.{name}"] = raw_tensor
        manifest_bytes = _canonical_json_bytes(manifest)
        if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
            raise RouteTraceValidationError(
                "route trace manifest exceeds the size limit"
            )
        buffers[_MANIFEST_TENSOR] = torch.frombuffer(
            bytearray(manifest_bytes), dtype=torch.uint8
        ).clone()
        try:
            return safetensors.save(buffers)
        except Exception as exc:
            raise RouteTraceValidationError("failed to serialize route trace") from exc

    @staticmethod
    def loads(
        data: bytes | bytearray | memoryview,
        *,
        max_serialized_bytes: int = _DEFAULT_MAX_SERIALIZED_BYTES,
    ) -> RouteTraceV0:
        """Decode and fully validate an untrusted route trace byte payload."""
        _require_int("max_serialized_bytes", max_serialized_bytes, minimum=1)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise RouteTraceValidationError("serialized route trace must be bytes-like")
        data = bytes(data)
        if len(data) > max_serialized_bytes:
            raise RouteTraceValidationError(
                "serialized route trace exceeds the size limit"
            )
        try:
            tensors = safetensors.load(data)
        except Exception as exc:
            raise RouteTraceValidationError(
                "invalid safetensors route trace container"
            ) from exc
        if _MANIFEST_TENSOR not in tensors:
            raise RouteTraceValidationError("route trace manifest tensor is missing")
        manifest_tensor = tensors[_MANIFEST_TENSOR]
        if manifest_tensor.dtype is not torch.uint8 or manifest_tensor.ndim != 1:
            raise RouteTraceValidationError(
                "route trace manifest tensor must be one-dimensional uint8"
            )
        if manifest_tensor.numel() > _MAX_MANIFEST_BYTES:
            raise RouteTraceValidationError(
                "route trace manifest exceeds the size limit"
            )
        try:
            manifest = json.loads(bytes(manifest_tensor.tolist()).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RouteTraceValidationError(
                "route trace manifest is not valid UTF-8 JSON"
            ) from exc
        manifest = _require_exact_keys(
            manifest,
            {
                "byte_order",
                "codec",
                "identity",
                "level",
                "route_trace_schema_version",
                "router_schema",
                "tensors",
                "token_layout",
            },
            "route trace manifest",
        )
        if manifest["byte_order"] != "little" or sys.byteorder != "little":
            raise RouteTraceValidationError(
                f"unsupported route trace byte order: {manifest['byte_order']!r}"
            )
        if manifest["codec"] != _CODEC_NAME:
            raise RouteTraceValidationError(
                f"unsupported route trace codec: {manifest['codec']!r}"
            )
        try:
            _require_int(
                "route_trace_schema_version",
                manifest["route_trace_schema_version"],
                minimum=ROUTE_TRACE_SCHEMA_VERSION,
                maximum=ROUTE_TRACE_SCHEMA_VERSION,
            )
        except RouteTraceValidationError as exc:
            raise RouteTraceValidationError(
                f"unsupported route trace schema version: {manifest['route_trace_schema_version']!r}"
            ) from exc
        tensor_descriptors = manifest["tensors"]
        if not isinstance(tensor_descriptors, Mapping):
            raise RouteTraceValidationError(
                "route trace tensor table must be a mapping"
            )
        unknown_tensor_names = set(tensor_descriptors) - set(_TENSOR_NAMES)
        if unknown_tensor_names:
            raise RouteTraceValidationError(
                f"route trace contains unknown logical tensors: {sorted(unknown_tensor_names)}"
            )
        expected_container_names = {_MANIFEST_TENSOR} | {
            f"tensor.{name}" for name in tensor_descriptors
        }
        if set(tensors) != expected_container_names:
            raise RouteTraceValidationError(
                "route trace container tensor names do not match the manifest"
            )
        decoded = {
            name: _decode_tensor(name, descriptor, tensors[f"tensor.{name}"])
            for name, descriptor in tensor_descriptors.items()
        }
        return RouteTraceV0(
            identity=RouteIdentityV0.from_dict(manifest["identity"]),
            router_schema=RouterSchemaV0.from_dict(manifest["router_schema"]),
            token_layout=RouteTokenLayoutV0.from_dict(manifest["token_layout"]),
            level=_parse_enum(RouteTraceLevel, "level", manifest["level"]),
            topk_ids=decoded.get("topk_ids"),
            selected_weights=decoded.get("selected_weights"),
            topk_margin=decoded.get("topk_margin"),
            valid_mask=decoded.get("valid_mask"),
        )
