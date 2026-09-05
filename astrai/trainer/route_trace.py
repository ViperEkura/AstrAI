"""Immutable binding between MoE route traces and online rollout tokens.

The model-side capture path is deliberately outside this module. A producer
that already has one :class:`~astrai.moe.RouteTraceV0` per generated response
can use :meth:`RolloutRouteTraceBatchV0.bind` to attach those artifacts to a
rollout. The resulting payloads are immutable bytes, and every cache boundary
can validate their policy and token identity without decoding large route
tensors again.
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from astrai.moe import (
    PaddingLayout,
    RouteIdentityV0,
    RouterSchemaV0,
    RouteTokenLayoutV0,
    RouteTraceCodecV0,
    RouteTraceLevel,
    RouteTraceV0,
    RouteTraceValidationError,
    TokenSpanKind,
    canonical_json_digest,
    validate_route_trace,
)

ROLLOUT_ROUTE_TRACE_SCHEMA_VERSION = 0

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

__all__ = [
    "ROLLOUT_ROUTE_TRACE_SCHEMA_VERSION",
    "RolloutRouteTraceBatchV0",
    "RolloutRouteTraceError",
    "RolloutRouteTraceItemV0",
    "rollout_route_sample_id",
]


class RolloutRouteTraceError(RouteTraceValidationError):
    """A route-trace batch cannot be attributed to its rollout tokens."""


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RolloutRouteTraceError(f"{name} must be an integer >= {minimum}")
    return value


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 1024:
        raise RolloutRouteTraceError(f"{name} must be a non-empty bounded string")
    return value


def _require_digest(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise RolloutRouteTraceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def rollout_route_sample_id(rollout_id: str, batch_index: int, group_index: int) -> str:
    """Derive the only valid per-response sample ID for a rollout grid cell."""
    _require_text("rollout_id", rollout_id)
    _require_int("batch_index", batch_index, minimum=0)
    _require_int("group_index", group_index, minimum=0)
    return canonical_json_digest(
        {
            "batch_index": batch_index,
            "group_index": group_index,
            "rollout_id": rollout_id,
            "schema_version": ROLLOUT_ROUTE_TRACE_SCHEMA_VERSION,
        }
    )


def _tensor_digest(name: str, tensor: Tensor) -> str:
    if not isinstance(tensor, Tensor):
        raise RolloutRouteTraceError(f"{name} must be a torch.Tensor")
    if tensor.layout is not torch.strided or tensor.device.type == "meta":
        raise RolloutRouteTraceError(f"{name} must be a materialized strided tensor")
    if tensor.requires_grad:
        raise RolloutRouteTraceError(f"{name} must be detached from autograd")
    if sys.byteorder != "little":
        raise RolloutRouteTraceError(
            "rollout trace binding requires little-endian tensors"
        )
    cpu_tensor = tensor.detach().contiguous().cpu()
    raw = cpu_tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
    return canonical_json_digest(
        {
            "byte_order": "little",
            "dtype": str(cpu_tensor.dtype).removeprefix("torch."),
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "shape": list(cpu_tensor.shape),
        }
    )


def _effective_mask(trace: RouteTraceV0) -> Tensor:
    if trace.valid_mask is not None:
        return trace.valid_mask.detach().cpu()
    return torch.ones(trace.token_layout.token_count, dtype=torch.bool)


def _validate_rollout_tensors(
    *,
    prompts: Tensor,
    prompt_mask: Tensor,
    responses: Tensor,
    response_mask: Tensor,
    logprobs_old: Tensor,
) -> tuple[int, int, int, int, str, str, str]:
    tensors = {
        "prompts": prompts,
        "prompt_mask": prompt_mask,
        "responses": responses,
        "response_mask": response_mask,
        "logprobs_old": logprobs_old,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, Tensor):
            raise RolloutRouteTraceError(f"{name} must be a torch.Tensor")
        if tensor.layout is not torch.strided or tensor.device.type == "meta":
            raise RolloutRouteTraceError(
                f"{name} must be a materialized strided tensor"
            )
        if tensor.requires_grad:
            raise RolloutRouteTraceError(f"{name} must be detached from autograd")

    if prompts.dtype is not torch.long or prompts.ndim != 2:
        raise RolloutRouteTraceError("prompts must be int64 with shape [B, P]")
    if responses.dtype is not torch.long or responses.ndim != 3:
        raise RolloutRouteTraceError("responses must be int64 with shape [B, G, R]")
    if prompt_mask.dtype is not torch.bool or prompt_mask.shape != prompts.shape:
        raise RolloutRouteTraceError("prompt_mask must be bool with shape [B, P]")
    if response_mask.dtype is not torch.bool or response_mask.shape != responses.shape:
        raise RolloutRouteTraceError("response_mask must be bool with shape [B, G, R]")
    if not logprobs_old.is_floating_point() or logprobs_old.shape != responses.shape:
        raise RolloutRouteTraceError(
            "logprobs_old must be floating with shape [B, G, R]"
        )
    if len({tensor.device for tensor in tensors.values()}) != 1:
        raise RolloutRouteTraceError(
            "rollout token tensors and masks must share one device"
        )

    batch_size, prompt_width = prompts.shape
    response_batch_size, group_size, response_width = responses.shape
    if batch_size < 1 or group_size < 1 or prompt_width < 1 or response_width < 1:
        raise RolloutRouteTraceError(
            "rollout token tensors require positive B, G, P, and R dimensions"
        )
    if response_batch_size != batch_size:
        raise RolloutRouteTraceError("prompt and response batch dimensions differ")
    if not bool(prompt_mask.any(dim=-1).all().item()):
        raise RolloutRouteTraceError("each rollout prompt must contain a valid token")

    if prompt_width > 1 and bool(
        (prompt_mask[:, :-1] & ~prompt_mask[:, 1:]).any().item()
    ):
        raise RolloutRouteTraceError(
            "prompt_mask must use canonical left-padding order"
        )
    if response_width > 1 and bool(
        ((~response_mask[..., :-1]) & response_mask[..., 1:]).any().item()
    ):
        raise RolloutRouteTraceError(
            "response_mask must use canonical right-padding order"
        )
    if bool(torch.count_nonzero(prompts[~prompt_mask]).item()):
        raise RolloutRouteTraceError("padded prompt token IDs must be canonical zero")
    if bool(torch.count_nonzero(responses[~response_mask]).item()):
        raise RolloutRouteTraceError("padded response token IDs must be canonical zero")
    if bool((prompts[prompt_mask] < 0).any().item()) or bool(
        (responses[response_mask] < 0).any().item()
    ):
        raise RolloutRouteTraceError("valid rollout token IDs must be non-negative")
    if not bool(torch.isfinite(logprobs_old[response_mask]).all().item()):
        raise RolloutRouteTraceError(
            "valid behavior logprobs must contain only finite values"
        )
    if bool((logprobs_old[response_mask] > 0).any().item()):
        raise RolloutRouteTraceError("valid behavior logprobs must be non-positive")
    padded_logprobs = logprobs_old[~response_mask]
    if padded_logprobs.numel() and (
        bool(torch.count_nonzero(padded_logprobs).item())
        or bool(torch.signbit(padded_logprobs).any().item())
    ):
        raise RolloutRouteTraceError(
            "padded behavior logprobs must use canonical positive zero"
        )

    prompt_batch_digest = canonical_json_digest(
        {
            "prompt_mask": _tensor_digest("prompt_mask", prompt_mask),
            "prompts": _tensor_digest("prompts", prompts),
        }
    )
    response_batch_digest = canonical_json_digest(
        {
            "response_mask": _tensor_digest("response_mask", response_mask),
            "responses": _tensor_digest("responses", responses),
        }
    )
    behavior_logprobs_digest = _tensor_digest("logprobs_old", logprobs_old)
    return (
        batch_size,
        group_size,
        prompt_width,
        response_width,
        prompt_batch_digest,
        response_batch_digest,
        behavior_logprobs_digest,
    )


@dataclass(frozen=True)
class RolloutRouteTraceItemV0:
    """One immutable serialized trace and its verified lightweight header."""

    payload: bytes = field(repr=False)
    artifact_digest: str
    identity: RouteIdentityV0
    router_schema: RouterSchemaV0
    token_layout: RouteTokenLayoutV0
    level: RouteTraceLevel
    valid_mask_digest: str

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes:
            raise RolloutRouteTraceError("route trace payload must be immutable bytes")
        _require_digest("artifact_digest", self.artifact_digest)
        _require_digest("valid_mask_digest", self.valid_mask_digest)
        if hashlib.sha256(self.payload).hexdigest() != self.artifact_digest:
            raise RolloutRouteTraceError("route trace payload digest does not match")
        try:
            decoded = RouteTraceCodecV0.loads(self.payload)
        except RouteTraceValidationError as exc:
            raise RolloutRouteTraceError("route trace payload is invalid") from exc
        if decoded.identity != self.identity:
            raise RolloutRouteTraceError("route trace identity header does not match")
        if decoded.router_schema != self.router_schema:
            raise RolloutRouteTraceError("route trace schema header does not match")
        if decoded.token_layout != self.token_layout:
            raise RolloutRouteTraceError("route trace token header does not match")
        if decoded.level is not self.level:
            raise RolloutRouteTraceError("route trace level header does not match")
        if (
            _tensor_digest("route trace valid mask", _effective_mask(decoded))
            != self.valid_mask_digest
        ):
            raise RolloutRouteTraceError("route trace validity header does not match")

    @classmethod
    def from_trace(cls, trace: RouteTraceV0) -> RolloutRouteTraceItemV0:
        """Freeze one validated trace into immutable transport bytes."""
        if not isinstance(trace, RouteTraceV0):
            raise RolloutRouteTraceError("route trace item must be RouteTraceV0")
        validate_route_trace(trace)
        payload = RouteTraceCodecV0.dumps(trace)
        return cls(
            payload=payload,
            artifact_digest=hashlib.sha256(payload).hexdigest(),
            identity=trace.identity,
            router_schema=trace.router_schema,
            token_layout=trace.token_layout,
            level=trace.level,
            valid_mask_digest=_tensor_digest(
                "route trace valid mask", _effective_mask(trace)
            ),
        )

    @classmethod
    def from_payload(
        cls,
        payload: bytes | bytearray | memoryview,
        *,
        max_serialized_bytes: int,
    ) -> RolloutRouteTraceItemV0:
        """Build a verified item from an untrusted serialized trace payload."""
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise RolloutRouteTraceError("route trace payload must be bytes-like")
        payload = bytes(payload)
        try:
            trace = RouteTraceCodecV0.loads(
                payload,
                max_serialized_bytes=max_serialized_bytes,
            )
        except RouteTraceValidationError as exc:
            raise RolloutRouteTraceError(
                f"route trace payload is invalid: {exc}"
            ) from exc
        return cls(
            payload=payload,
            artifact_digest=hashlib.sha256(payload).hexdigest(),
            identity=trace.identity,
            router_schema=trace.router_schema,
            token_layout=trace.token_layout,
            level=trace.level,
            valid_mask_digest=_tensor_digest(
                "route trace valid mask", _effective_mask(trace)
            ),
        )

    def decode(self) -> RouteTraceV0:
        """Decode a fresh CPU trace after verifying immutable content identity."""
        if hashlib.sha256(self.payload).hexdigest() != self.artifact_digest:
            raise RolloutRouteTraceError("route trace payload digest does not match")
        return RouteTraceCodecV0.loads(self.payload)

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-compatible header without copying the tensor payload."""
        return {
            "artifact_digest": self.artifact_digest,
            "identity": self.identity.to_dict(),
            "level": self.level.value,
            "payload_nbytes": len(self.payload),
            "router_schema": self.router_schema.to_dict(),
            "token_layout": self.token_layout.to_dict(),
            "valid_mask_digest": self.valid_mask_digest,
        }


@dataclass(frozen=True)
class RolloutRouteTraceBatchV0:
    """A content-addressed B-by-G grid bound to one rollout token batch."""

    rollout_id: str
    policy_version: int
    prompt_batch_digest: str
    response_batch_digest: str
    behavior_logprobs_digest: str
    items: tuple[tuple[RolloutRouteTraceItemV0, ...], ...]
    schema_version: int = ROLLOUT_ROUTE_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_int("schema_version", self.schema_version, minimum=0)
        if self.schema_version != ROLLOUT_ROUTE_TRACE_SCHEMA_VERSION:
            raise RolloutRouteTraceError(
                f"unsupported rollout route trace schema version {self.schema_version}"
            )
        _require_text("rollout_id", self.rollout_id)
        _require_int("policy_version", self.policy_version, minimum=0)
        _require_digest("prompt_batch_digest", self.prompt_batch_digest)
        _require_digest("response_batch_digest", self.response_batch_digest)
        _require_digest("behavior_logprobs_digest", self.behavior_logprobs_digest)
        if type(self.items) is not tuple or not self.items:
            raise RolloutRouteTraceError(
                "route trace items must be a non-empty tuple grid"
            )
        if any(type(row) is not tuple or not row for row in self.items):
            raise RolloutRouteTraceError(
                "each route trace item row must be a non-empty tuple"
            )
        group_size = len(self.items[0])
        if any(len(row) != group_size for row in self.items):
            raise RolloutRouteTraceError(
                "route trace item rows have unequal group sizes"
            )

        first = self.items[0][0]
        if not isinstance(first, RolloutRouteTraceItemV0):
            raise RolloutRouteTraceError(
                "route trace item grid contains an invalid item"
            )
        sample_ids: set[str] = set()
        for row in self.items:
            for item in row:
                if not isinstance(item, RolloutRouteTraceItemV0):
                    raise RolloutRouteTraceError(
                        "route trace item grid contains an invalid item"
                    )
                if item.identity.policy_version != self.policy_version:
                    raise RolloutRouteTraceError(
                        "route trace policy version does not match its batch"
                    )
                if item.identity != first.identity:
                    raise RolloutRouteTraceError(
                        "one rollout trace batch cannot mix behavior identities"
                    )
                if item.router_schema != first.router_schema:
                    raise RolloutRouteTraceError(
                        "one rollout trace batch cannot mix router schemas"
                    )
                if item.level is not first.level:
                    raise RolloutRouteTraceError(
                        "one rollout trace batch cannot mix trace levels"
                    )
                sample_id = item.token_layout.sample_id
                if sample_id in sample_ids:
                    raise RolloutRouteTraceError(
                        "route trace sample IDs must be unique within a rollout"
                    )
                sample_ids.add(sample_id)

    @property
    def batch_size(self) -> int:
        return len(self.items)

    @property
    def group_size(self) -> int:
        return len(self.items[0])

    @property
    def identity(self) -> RouteIdentityV0:
        return self.items[0][0].identity

    @property
    def router_schema(self) -> RouterSchemaV0:
        return self.items[0][0].router_schema

    @property
    def level(self) -> RouteTraceLevel:
        return self.items[0][0].level

    @property
    def artifact_digest(self) -> str:
        """Return the content address of token identity and ordered trace IDs."""
        return canonical_json_digest(self.manifest())

    @property
    def payload_nbytes(self) -> int:
        """Return the exact combined serialized trace bytes in this batch."""
        return sum(len(item.payload) for row in self.items for item in row)

    @classmethod
    def bind(
        cls,
        *,
        rollout_id: str,
        policy_version: int,
        prompts: Tensor,
        prompt_mask: Tensor,
        responses: Tensor,
        response_mask: Tensor,
        logprobs_old: Tensor,
        traces: Sequence[Sequence[RouteTraceV0]],
    ) -> RolloutRouteTraceBatchV0:
        """Freeze traces and bind them to an exact rollout tensor identity."""
        (
            batch_size,
            group_size,
            _,
            _,
            prompt_batch_digest,
            response_batch_digest,
            behavior_logprobs_digest,
        ) = _validate_rollout_tensors(
            prompts=prompts,
            prompt_mask=prompt_mask,
            responses=responses,
            response_mask=response_mask,
            logprobs_old=logprobs_old,
        )
        if isinstance(traces, (str, bytes)) or not isinstance(traces, Sequence):
            raise RolloutRouteTraceError("traces must be a rectangular sequence grid")
        if any(
            isinstance(row, (str, bytes)) or not isinstance(row, Sequence)
            for row in traces
        ):
            raise RolloutRouteTraceError("traces must be a rectangular sequence grid")
        if len(traces) != batch_size or any(len(row) != group_size for row in traces):
            raise RolloutRouteTraceError(
                "rollout B/G dimensions do not match the route trace grid"
            )
        if any(not isinstance(trace, RouteTraceV0) for row in traces for trace in row):
            raise RolloutRouteTraceError("route trace item must be RouteTraceV0")
        items = tuple(
            tuple(RolloutRouteTraceItemV0.from_trace(trace) for trace in row)
            for row in traces
        )
        batch = cls(
            rollout_id=rollout_id,
            policy_version=policy_version,
            prompt_batch_digest=prompt_batch_digest,
            response_batch_digest=response_batch_digest,
            behavior_logprobs_digest=behavior_logprobs_digest,
            items=items,
        )
        batch.validate_against(
            policy_version=policy_version,
            prompts=prompts,
            prompt_mask=prompt_mask,
            responses=responses,
            response_mask=response_mask,
            logprobs_old=logprobs_old,
        )
        return batch

    def validate_against(
        self,
        *,
        policy_version: int,
        prompts: Tensor,
        prompt_mask: Tensor,
        responses: Tensor,
        response_mask: Tensor,
        logprobs_old: Tensor,
    ) -> None:
        """Require exact policy, token bytes, grid, boundary, and mask identity."""
        _require_int("policy_version", policy_version, minimum=0)
        if policy_version != self.policy_version:
            raise RolloutRouteTraceError(
                "rollout policy version does not match its route trace batch"
            )
        (
            batch_size,
            group_size,
            _,
            response_width,
            prompt_batch_digest,
            response_batch_digest,
            behavior_logprobs_digest,
        ) = _validate_rollout_tensors(
            prompts=prompts,
            prompt_mask=prompt_mask,
            responses=responses,
            response_mask=response_mask,
            logprobs_old=logprobs_old,
        )
        if (batch_size, group_size) != (self.batch_size, self.group_size):
            raise RolloutRouteTraceError(
                "rollout B/G dimensions do not match the route trace grid"
            )
        if prompt_batch_digest != self.prompt_batch_digest:
            raise RolloutRouteTraceError(
                "rollout prompt tokens do not match the route trace binding"
            )
        if response_batch_digest != self.response_batch_digest:
            raise RolloutRouteTraceError(
                "rollout response tokens do not match the route trace binding"
            )
        if behavior_logprobs_digest != self.behavior_logprobs_digest:
            raise RolloutRouteTraceError(
                "behavior logprobs do not match the route trace binding"
            )

        prompt_lengths = prompt_mask.sum(dim=-1).detach().cpu().tolist()
        response_masks = response_mask.detach().cpu()
        for batch_index, row in enumerate(self.items):
            prompt_length = int(prompt_lengths[batch_index])
            for group_index, item in enumerate(row):
                layout = item.token_layout
                if layout.sample_id != rollout_route_sample_id(
                    self.rollout_id, batch_index, group_index
                ):
                    raise RolloutRouteTraceError(
                        "route trace sample ID does not match its rollout grid position"
                    )
                expected_layout = {
                    "prompt_token_count": prompt_length,
                    "sequence_token_count": prompt_length + response_width,
                    "span_kind": TokenSpanKind.RESPONSE,
                    "token_count": response_width,
                    "token_offset": prompt_length,
                }
                actual_layout = {
                    "prompt_token_count": layout.prompt_token_count,
                    "sequence_token_count": layout.sequence_token_count,
                    "span_kind": layout.span_kind,
                    "token_count": layout.token_count,
                    "token_offset": layout.token_offset,
                }
                if actual_layout != expected_layout:
                    raise RolloutRouteTraceError(
                        "route trace token boundaries do not match the rollout response"
                    )
                expected_mask = response_masks[batch_index, group_index]
                if (
                    _tensor_digest("response route mask", expected_mask)
                    != item.valid_mask_digest
                ):
                    raise RolloutRouteTraceError(
                        "route trace validity mask does not match the rollout response"
                    )
                if bool(
                    (~expected_mask).any().item()
                ) and item.router_schema.padding_layout not in (
                    PaddingLayout.RIGHT,
                    PaddingLayout.EXPLICIT_MASK,
                ):
                    raise RolloutRouteTraceError(
                        "padded rollout responses require right or explicit trace padding"
                    )

    def decode_traces(self) -> tuple[tuple[RouteTraceV0, ...], ...]:
        """Decode fresh CPU trace objects while preserving B/G ordering."""
        return tuple(tuple(item.decode() for item in row) for row in self.items)

    def manifest(self) -> dict[str, Any]:
        """Return a strict JSON-compatible identity manifest without payloads."""
        return {
            "items": [[item.manifest() for item in row] for row in self.items],
            "behavior_logprobs_digest": self.behavior_logprobs_digest,
            "payload_nbytes": self.payload_nbytes,
            "policy_version": self.policy_version,
            "prompt_batch_digest": self.prompt_batch_digest,
            "response_batch_digest": self.response_batch_digest,
            "rollout_id": self.rollout_id,
            "schema_version": self.schema_version,
        }
