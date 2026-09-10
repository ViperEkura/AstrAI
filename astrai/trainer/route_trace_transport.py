"""Bounded, content-addressed shards for rollout route-trace transport.

The transport layer splits an already validated
:class:`~astrai.trainer.route_trace.RolloutRouteTraceBatchV0` into independently
verifiable byte frames. It does not prescribe Ray, NCCL, shared-memory, or
filesystem delivery. Receivers can validate one assigned shard without loading
the other route tensors and can only reconstruct a batch after every declared
shard is present and matches the original batch digest.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from astrai.moe import canonical_json_digest
from astrai.trainer.route_trace import (
    ROLLOUT_ROUTE_TRACE_SCHEMA_VERSION,
    RolloutRouteTraceBatchV0,
    RolloutRouteTraceError,
    RolloutRouteTraceItemV0,
)

ROLLOUT_ROUTE_TRACE_SHARD_SCHEMA_VERSION = 0
DEFAULT_ROUTE_TRACE_SHARD_BYTES = 16 * 1024 * 1024
DEFAULT_ROUTE_TRACE_SHARD_ITEMS = 128
DEFAULT_ROUTE_TRACE_TRANSPORT_BYTES = 1024 * 1024 * 1024

_FRAME_MAGIC = b"ARTRSHD0"
_FRAME_PREFIX = struct.Struct(">8sI")
_MAX_HEADER_BYTES = 8 * 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_SHARD_COUNT = 131_072
_MAX_ITEMS_PER_SHARD = 65_536
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

__all__ = [
    "DEFAULT_ROUTE_TRACE_SHARD_BYTES",
    "DEFAULT_ROUTE_TRACE_SHARD_ITEMS",
    "DEFAULT_ROUTE_TRACE_TRANSPORT_BYTES",
    "ROLLOUT_ROUTE_TRACE_SHARD_SCHEMA_VERSION",
    "RolloutRouteTraceShardCodecV0",
    "RolloutRouteTraceShardDescriptorV0",
    "RolloutRouteTraceShardManifestV0",
    "RolloutRouteTraceShardV0",
    "RolloutRouteTraceTransportV0",
]


def _require_int(
    name: str,
    value: Any,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RolloutRouteTraceError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise RolloutRouteTraceError(f"{name} must be <= {maximum}")
    return value


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or len(value) > 1024:
        raise RolloutRouteTraceError(f"{name} must be a non-empty bounded string")
    return value


def _require_digest(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise RolloutRouteTraceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_exact_keys(
    value: Any,
    expected: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RolloutRouteTraceError(f"{label} must be a mapping")
    keys = set(value)
    if keys != expected:
        raise RolloutRouteTraceError(
            f"{label} keys do not match version-zero schema; "
            f"missing={sorted(expected - keys)}, unknown={sorted(keys - expected)}"
        )
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RolloutRouteTraceError(
            "route shard metadata is not canonical-JSON serializable"
        ) from exc
    return rendered.encode("utf-8")


def _decode_json_object(data: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RolloutRouteTraceError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise RolloutRouteTraceError(f"{label} must contain a JSON object")
    if _canonical_json_bytes(value) != data:
        raise RolloutRouteTraceError(f"{label} must use canonical JSON encoding")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class RolloutRouteTraceShardDescriptorV0:
    """Payload-free identity and range of one encoded shard."""

    shard_index: int
    first_flat_index: int
    item_count: int
    payload_nbytes: int
    encoded_nbytes: int
    encoded_sha256: str

    def __post_init__(self) -> None:
        _require_int("shard_index", self.shard_index, minimum=0)
        _require_int("first_flat_index", self.first_flat_index, minimum=0)
        _require_int(
            "item_count",
            self.item_count,
            minimum=1,
            maximum=_MAX_ITEMS_PER_SHARD,
        )
        _require_int("payload_nbytes", self.payload_nbytes, minimum=1)
        _require_int(
            "encoded_nbytes",
            self.encoded_nbytes,
            minimum=_FRAME_PREFIX.size + 2,
        )
        if self.encoded_nbytes <= self.payload_nbytes:
            raise RolloutRouteTraceError(
                "encoded_nbytes must include metadata beyond the item payloads"
            )
        _require_digest("encoded_sha256", self.encoded_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoded_nbytes": self.encoded_nbytes,
            "encoded_sha256": self.encoded_sha256,
            "first_flat_index": self.first_flat_index,
            "item_count": self.item_count,
            "payload_nbytes": self.payload_nbytes,
            "shard_index": self.shard_index,
        }

    @classmethod
    def from_dict(cls, value: Any) -> RolloutRouteTraceShardDescriptorV0:
        fields = {
            "encoded_nbytes",
            "encoded_sha256",
            "first_flat_index",
            "item_count",
            "payload_nbytes",
            "shard_index",
        }
        value = _require_exact_keys(value, fields, "route shard descriptor")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class RolloutRouteTraceShardV0:
    """Decoded items from one independently authenticated shard frame."""

    batch_artifact_digest: str
    batch_size: int
    group_size: int
    shard_index: int
    first_flat_index: int
    items: tuple[RolloutRouteTraceItemV0, ...]
    schema_version: int = ROLLOUT_ROUTE_TRACE_SHARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_int(
            "schema_version",
            self.schema_version,
            minimum=ROLLOUT_ROUTE_TRACE_SHARD_SCHEMA_VERSION,
            maximum=ROLLOUT_ROUTE_TRACE_SHARD_SCHEMA_VERSION,
        )
        _require_digest("batch_artifact_digest", self.batch_artifact_digest)
        _require_int("batch_size", self.batch_size, minimum=1)
        _require_int("group_size", self.group_size, minimum=1)
        _require_int("shard_index", self.shard_index, minimum=0)
        _require_int("first_flat_index", self.first_flat_index, minimum=0)
        if type(self.items) is not tuple or not self.items:
            raise RolloutRouteTraceError("route shard items must be a non-empty tuple")
        if len(self.items) > _MAX_ITEMS_PER_SHARD:
            raise RolloutRouteTraceError("route shard item count exceeds the limit")
        if any(not isinstance(item, RolloutRouteTraceItemV0) for item in self.items):
            raise RolloutRouteTraceError(
                "route shard contains an invalid route trace item"
            )
        if self.first_flat_index + len(self.items) > self.batch_size * self.group_size:
            raise RolloutRouteTraceError("route shard range exceeds its rollout grid")

    @property
    def payload_nbytes(self) -> int:
        return sum(len(item.payload) for item in self.items)


def _item_header(item: RolloutRouteTraceItemV0) -> dict[str, Any]:
    return {
        "artifact_digest": item.artifact_digest,
        "payload_nbytes": len(item.payload),
        "sample_id": item.token_layout.sample_id,
    }


def _shard_header(shard: RolloutRouteTraceShardV0) -> dict[str, Any]:
    return {
        "batch_artifact_digest": shard.batch_artifact_digest,
        "batch_size": shard.batch_size,
        "first_flat_index": shard.first_flat_index,
        "group_size": shard.group_size,
        "items": [_item_header(item) for item in shard.items],
        "schema_version": shard.schema_version,
        "shard_index": shard.shard_index,
    }


class RolloutRouteTraceShardCodecV0:
    """Length-prefixed, non-executable codec for one route-trace shard."""

    @staticmethod
    def dumps(shard: RolloutRouteTraceShardV0) -> bytes:
        if not isinstance(shard, RolloutRouteTraceShardV0):
            raise RolloutRouteTraceError("shard must be RolloutRouteTraceShardV0")
        header = _canonical_json_bytes(_shard_header(shard))
        if len(header) > _MAX_HEADER_BYTES:
            raise RolloutRouteTraceError("route shard header exceeds the size limit")
        return b"".join(
            (
                _FRAME_PREFIX.pack(_FRAME_MAGIC, len(header)),
                header,
                *(item.payload for item in shard.items),
            )
        )

    @staticmethod
    def loads(
        data: bytes | bytearray | memoryview,
        *,
        max_shard_bytes: int = DEFAULT_ROUTE_TRACE_SHARD_BYTES,
        max_item_bytes: int = DEFAULT_ROUTE_TRACE_SHARD_BYTES,
        max_items: int = DEFAULT_ROUTE_TRACE_SHARD_ITEMS,
    ) -> RolloutRouteTraceShardV0:
        _require_int("max_shard_bytes", max_shard_bytes, minimum=1)
        _require_int("max_item_bytes", max_item_bytes, minimum=1)
        _require_int(
            "max_items",
            max_items,
            minimum=1,
            maximum=_MAX_ITEMS_PER_SHARD,
        )
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise RolloutRouteTraceError("encoded route shard must be bytes-like")
        data = bytes(data)
        if len(data) > max_shard_bytes:
            raise RolloutRouteTraceError("encoded route shard exceeds the size limit")
        if len(data) < _FRAME_PREFIX.size:
            raise RolloutRouteTraceError("encoded route shard is truncated")
        magic, header_nbytes = _FRAME_PREFIX.unpack_from(data)
        if magic != _FRAME_MAGIC:
            raise RolloutRouteTraceError("encoded route shard magic does not match")
        if header_nbytes < 2 or header_nbytes > _MAX_HEADER_BYTES:
            raise RolloutRouteTraceError("route shard header size is invalid")
        body_offset = _FRAME_PREFIX.size + header_nbytes
        if body_offset > len(data):
            raise RolloutRouteTraceError("encoded route shard header is truncated")
        header = _decode_json_object(
            data[_FRAME_PREFIX.size : body_offset],
            label="route shard header",
        )
        fields = {
            "batch_artifact_digest",
            "batch_size",
            "first_flat_index",
            "group_size",
            "items",
            "schema_version",
            "shard_index",
        }
        header = _require_exact_keys(header, fields, "route shard header")
        _require_int(
            "schema_version",
            header["schema_version"],
            minimum=ROLLOUT_ROUTE_TRACE_SHARD_SCHEMA_VERSION,
            maximum=ROLLOUT_ROUTE_TRACE_SHARD_SCHEMA_VERSION,
        )
        _require_digest("batch_artifact_digest", header["batch_artifact_digest"])
        batch_size = _require_int("batch_size", header["batch_size"], minimum=1)
        group_size = _require_int("group_size", header["group_size"], minimum=1)
        shard_index = _require_int("shard_index", header["shard_index"], minimum=0)
        first_flat_index = _require_int(
            "first_flat_index", header["first_flat_index"], minimum=0
        )
        item_headers = header["items"]
        if not isinstance(item_headers, list) or not item_headers:
            raise RolloutRouteTraceError(
                "route shard header items must be a non-empty list"
            )
        if len(item_headers) > max_items:
            raise RolloutRouteTraceError("route shard item count exceeds the limit")
        if first_flat_index + len(item_headers) > batch_size * group_size:
            raise RolloutRouteTraceError("route shard range exceeds its rollout grid")

        item_fields = {"artifact_digest", "payload_nbytes", "sample_id"}
        cursor = body_offset
        items = []
        for index, raw_item_header in enumerate(item_headers):
            item_header = _require_exact_keys(
                raw_item_header,
                item_fields,
                f"route shard item header {index}",
            )
            artifact_digest = _require_digest(
                f"route shard item {index} artifact_digest",
                item_header["artifact_digest"],
            )
            payload_nbytes = _require_int(
                f"route shard item {index} payload_nbytes",
                item_header["payload_nbytes"],
                minimum=1,
                maximum=max_item_bytes,
            )
            sample_id = _require_text(
                f"route shard item {index} sample_id", item_header["sample_id"]
            )
            end = cursor + payload_nbytes
            if end > len(data):
                raise RolloutRouteTraceError("route shard item payload is truncated")
            payload = data[cursor:end]
            cursor = end
            if _sha256(payload) != artifact_digest:
                raise RolloutRouteTraceError(
                    "route shard item payload digest does not match"
                )
            item = RolloutRouteTraceItemV0.from_payload(
                payload,
                max_serialized_bytes=max_item_bytes,
            )
            if item.token_layout.sample_id != sample_id:
                raise RolloutRouteTraceError(
                    "route shard item sample ID does not match its payload"
                )
            items.append(item)
        if cursor != len(data):
            raise RolloutRouteTraceError("encoded route shard has trailing bytes")
        return RolloutRouteTraceShardV0(
            batch_artifact_digest=header["batch_artifact_digest"],
            batch_size=batch_size,
            group_size=group_size,
            shard_index=shard_index,
            first_flat_index=first_flat_index,
            items=tuple(items),
        )


@dataclass(frozen=True)
class RolloutRouteTraceShardManifestV0:
    """Small manifest needed to verify shards independently and reassemble."""

    batch_artifact_digest: str
    rollout_id: str
    policy_version: int
    prompt_batch_digest: str
    response_batch_digest: str
    behavior_logprobs_digest: str
    batch_size: int
    group_size: int
    total_payload_nbytes: int
    shards: tuple[RolloutRouteTraceShardDescriptorV0, ...]
    batch_schema_version: int = ROLLOUT_ROUTE_TRACE_SCHEMA_VERSION
    schema_version: int = ROLLOUT_ROUTE_TRACE_SHARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_int(
            "schema_version",
            self.schema_version,
            minimum=ROLLOUT_ROUTE_TRACE_SHARD_SCHEMA_VERSION,
            maximum=ROLLOUT_ROUTE_TRACE_SHARD_SCHEMA_VERSION,
        )
        _require_int(
            "batch_schema_version",
            self.batch_schema_version,
            minimum=ROLLOUT_ROUTE_TRACE_SCHEMA_VERSION,
            maximum=ROLLOUT_ROUTE_TRACE_SCHEMA_VERSION,
        )
        _require_digest("batch_artifact_digest", self.batch_artifact_digest)
        _require_text("rollout_id", self.rollout_id)
        _require_int("policy_version", self.policy_version, minimum=0)
        _require_digest("prompt_batch_digest", self.prompt_batch_digest)
        _require_digest("response_batch_digest", self.response_batch_digest)
        _require_digest("behavior_logprobs_digest", self.behavior_logprobs_digest)
        _require_int("batch_size", self.batch_size, minimum=1)
        _require_int("group_size", self.group_size, minimum=1)
        _require_int("total_payload_nbytes", self.total_payload_nbytes, minimum=1)
        if type(self.shards) is not tuple or not self.shards:
            raise RolloutRouteTraceError(
                "route shard descriptors must be a non-empty tuple"
            )
        if len(self.shards) > _MAX_SHARD_COUNT:
            raise RolloutRouteTraceError("route shard count exceeds the limit")

        next_flat_index = 0
        payload_nbytes = 0
        for shard_index, descriptor in enumerate(self.shards):
            if not isinstance(descriptor, RolloutRouteTraceShardDescriptorV0):
                raise RolloutRouteTraceError(
                    "route shard manifest contains an invalid descriptor"
                )
            if descriptor.shard_index != shard_index:
                raise RolloutRouteTraceError(
                    "route shard descriptor indices must be canonical and contiguous"
                )
            if descriptor.first_flat_index != next_flat_index:
                raise RolloutRouteTraceError(
                    "route shard item ranges must be canonical and contiguous"
                )
            next_flat_index += descriptor.item_count
            payload_nbytes += descriptor.payload_nbytes
        if next_flat_index != self.batch_size * self.group_size:
            raise RolloutRouteTraceError(
                "route shard ranges do not exactly cover the rollout grid"
            )
        if payload_nbytes != self.total_payload_nbytes:
            raise RolloutRouteTraceError(
                "route shard payload sizes do not match the manifest total"
            )

    @property
    def artifact_digest(self) -> str:
        return canonical_json_digest(self.to_dict())

    @property
    def encoded_nbytes(self) -> int:
        return sum(descriptor.encoded_nbytes for descriptor in self.shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_artifact_digest": self.batch_artifact_digest,
            "batch_schema_version": self.batch_schema_version,
            "batch_size": self.batch_size,
            "behavior_logprobs_digest": self.behavior_logprobs_digest,
            "group_size": self.group_size,
            "policy_version": self.policy_version,
            "prompt_batch_digest": self.prompt_batch_digest,
            "response_batch_digest": self.response_batch_digest,
            "rollout_id": self.rollout_id,
            "schema_version": self.schema_version,
            "shards": [descriptor.to_dict() for descriptor in self.shards],
            "total_payload_nbytes": self.total_payload_nbytes,
        }

    def dumps(self) -> bytes:
        payload = _canonical_json_bytes(self.to_dict())
        if len(payload) > _MAX_MANIFEST_BYTES:
            raise RolloutRouteTraceError("route shard manifest exceeds the size limit")
        return payload

    @classmethod
    def loads(
        cls,
        data: bytes | bytearray | memoryview,
        *,
        max_manifest_bytes: int = _MAX_MANIFEST_BYTES,
        max_shards: int = _MAX_SHARD_COUNT,
        max_shard_bytes: int = DEFAULT_ROUTE_TRACE_SHARD_BYTES,
        max_transport_bytes: int = DEFAULT_ROUTE_TRACE_TRANSPORT_BYTES,
    ) -> RolloutRouteTraceShardManifestV0:
        _require_int("max_manifest_bytes", max_manifest_bytes, minimum=1)
        _require_int(
            "max_shards",
            max_shards,
            minimum=1,
            maximum=_MAX_SHARD_COUNT,
        )
        _require_int("max_shard_bytes", max_shard_bytes, minimum=1)
        _require_int("max_transport_bytes", max_transport_bytes, minimum=1)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise RolloutRouteTraceError("route shard manifest must be bytes-like")
        data = bytes(data)
        if len(data) > max_manifest_bytes:
            raise RolloutRouteTraceError("route shard manifest exceeds the size limit")
        value = _decode_json_object(data, label="route shard manifest")
        fields = {
            "batch_artifact_digest",
            "batch_schema_version",
            "batch_size",
            "behavior_logprobs_digest",
            "group_size",
            "policy_version",
            "prompt_batch_digest",
            "response_batch_digest",
            "rollout_id",
            "schema_version",
            "shards",
            "total_payload_nbytes",
        }
        value = _require_exact_keys(value, fields, "route shard manifest")
        raw_descriptors = value["shards"]
        if not isinstance(raw_descriptors, list) or not raw_descriptors:
            raise RolloutRouteTraceError(
                "route shard manifest descriptors must be a non-empty list"
            )
        if len(raw_descriptors) > max_shards:
            raise RolloutRouteTraceError("route shard count exceeds the limit")
        kwargs = {name: value[name] for name in fields - {"shards"}}
        manifest = cls(
            **kwargs,
            shards=tuple(
                RolloutRouteTraceShardDescriptorV0.from_dict(descriptor)
                for descriptor in raw_descriptors
            ),
        )
        if any(
            descriptor.encoded_nbytes > max_shard_bytes
            for descriptor in manifest.shards
        ):
            raise RolloutRouteTraceError("declared route shard exceeds max_shard_bytes")
        if len(data) + manifest.encoded_nbytes > max_transport_bytes:
            raise RolloutRouteTraceError(
                "declared route transport exceeds max_transport_bytes"
            )
        return manifest

    def shard_indices_for_rank(self, rank: int, world_size: int) -> tuple[int, ...]:
        """Assign canonical shard indices round-robin for any positive world size."""
        _require_int("world_size", world_size, minimum=1)
        _require_int("rank", rank, minimum=0)
        if rank >= world_size:
            raise RolloutRouteTraceError("rank must be smaller than world_size")
        return tuple(range(rank, len(self.shards), world_size))

    def verify_shard(
        self,
        shard_index: int,
        payload: bytes | bytearray | memoryview,
    ) -> RolloutRouteTraceShardV0:
        _require_int("shard_index", shard_index, minimum=0)
        if shard_index >= len(self.shards):
            raise RolloutRouteTraceError("shard_index is outside the manifest")
        descriptor = self.shards[shard_index]
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise RolloutRouteTraceError("encoded route shard must be bytes-like")
        payload = bytes(payload)
        if len(payload) != descriptor.encoded_nbytes:
            raise RolloutRouteTraceError(
                "encoded route shard size does not match its descriptor"
            )
        if _sha256(payload) != descriptor.encoded_sha256:
            raise RolloutRouteTraceError(
                "encoded route shard digest does not match its descriptor"
            )
        shard = RolloutRouteTraceShardCodecV0.loads(
            payload,
            max_shard_bytes=descriptor.encoded_nbytes,
            max_item_bytes=descriptor.payload_nbytes,
            max_items=descriptor.item_count,
        )
        if shard.batch_artifact_digest != self.batch_artifact_digest:
            raise RolloutRouteTraceError("route shard belongs to a different batch")
        if (shard.batch_size, shard.group_size) != (
            self.batch_size,
            self.group_size,
        ):
            raise RolloutRouteTraceError("route shard rollout grid does not match")
        if shard.shard_index != descriptor.shard_index:
            raise RolloutRouteTraceError("route shard index does not match")
        if shard.first_flat_index != descriptor.first_flat_index:
            raise RolloutRouteTraceError("route shard range does not match")
        if len(shard.items) != descriptor.item_count:
            raise RolloutRouteTraceError("route shard item count does not match")
        if shard.payload_nbytes != descriptor.payload_nbytes:
            raise RolloutRouteTraceError("route shard payload size does not match")
        return shard

    def assemble(
        self,
        shard_payloads: Sequence[bytes | bytearray | memoryview],
    ) -> RolloutRouteTraceBatchV0:
        """Reconstruct the original batch only from the exact complete shard set."""
        if isinstance(
            shard_payloads, (str, bytes, bytearray, memoryview)
        ) or not isinstance(shard_payloads, Sequence):
            raise RolloutRouteTraceError("shard_payloads must be an ordered sequence")
        if len(shard_payloads) != len(self.shards):
            raise RolloutRouteTraceError(
                "shard payload count does not match the manifest"
            )
        flat_items = []
        for shard_index, payload in enumerate(shard_payloads):
            flat_items.extend(self.verify_shard(shard_index, payload).items)
        rows = tuple(
            tuple(flat_items[start : start + self.group_size])
            for start in range(0, len(flat_items), self.group_size)
        )
        batch = RolloutRouteTraceBatchV0(
            rollout_id=self.rollout_id,
            policy_version=self.policy_version,
            prompt_batch_digest=self.prompt_batch_digest,
            response_batch_digest=self.response_batch_digest,
            behavior_logprobs_digest=self.behavior_logprobs_digest,
            items=rows,
            schema_version=self.batch_schema_version,
        )
        if batch.payload_nbytes != self.total_payload_nbytes:
            raise RolloutRouteTraceError(
                "reassembled route trace payload size does not match"
            )
        if batch.artifact_digest != self.batch_artifact_digest:
            raise RolloutRouteTraceError(
                "reassembled route trace batch digest does not match"
            )
        return batch


def _encoded_nbytes(
    *,
    batch_artifact_digest: str,
    batch_size: int,
    group_size: int,
    shard_index: int,
    first_flat_index: int,
    items: Sequence[RolloutRouteTraceItemV0],
) -> int:
    empty = RolloutRouteTraceShardV0(
        batch_artifact_digest=batch_artifact_digest,
        batch_size=batch_size,
        group_size=group_size,
        shard_index=shard_index,
        first_flat_index=first_flat_index,
        items=(items[0],),
    )
    base_header = _shard_header(empty)
    base_header["items"] = []
    base_nbytes = len(_canonical_json_bytes(base_header))
    item_header_nbytes = sum(
        len(_canonical_json_bytes(_item_header(item))) for item in items
    )
    separators = max(0, len(items) - 1)
    header_nbytes = base_nbytes + item_header_nbytes + separators
    return _FRAME_PREFIX.size + header_nbytes + sum(len(item.payload) for item in items)


@dataclass(frozen=True)
class RolloutRouteTraceTransportV0:
    """Sender-side manifest and immutable shard byte frames."""

    manifest: RolloutRouteTraceShardManifestV0
    shard_payloads: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, RolloutRouteTraceShardManifestV0):
            raise RolloutRouteTraceError(
                "manifest must be RolloutRouteTraceShardManifestV0"
            )
        if type(self.shard_payloads) is not tuple:
            raise RolloutRouteTraceError("shard_payloads must be an immutable tuple")
        if len(self.shard_payloads) != len(self.manifest.shards):
            raise RolloutRouteTraceError(
                "shard payload count does not match the manifest"
            )
        for descriptor, payload in zip(self.manifest.shards, self.shard_payloads):
            if type(payload) is not bytes:
                raise RolloutRouteTraceError(
                    "encoded route shard payloads must be immutable bytes"
                )
            if len(payload) != descriptor.encoded_nbytes:
                raise RolloutRouteTraceError(
                    "encoded route shard size does not match its descriptor"
                )
            if _sha256(payload) != descriptor.encoded_sha256:
                raise RolloutRouteTraceError(
                    "encoded route shard digest does not match its descriptor"
                )

    @classmethod
    def from_batch(
        cls,
        batch: RolloutRouteTraceBatchV0,
        *,
        max_shard_bytes: int = DEFAULT_ROUTE_TRACE_SHARD_BYTES,
        max_items_per_shard: int = DEFAULT_ROUTE_TRACE_SHARD_ITEMS,
    ) -> RolloutRouteTraceTransportV0:
        """Partition a bound rollout batch into deterministic bounded shards."""
        if not isinstance(batch, RolloutRouteTraceBatchV0):
            raise RolloutRouteTraceError("batch must be RolloutRouteTraceBatchV0")
        _require_int("max_shard_bytes", max_shard_bytes, minimum=1)
        _require_int(
            "max_items_per_shard",
            max_items_per_shard,
            minimum=1,
            maximum=_MAX_ITEMS_PER_SHARD,
        )
        flat_items = tuple(item for row in batch.items for item in row)
        batch_artifact_digest = batch.artifact_digest
        chunks: list[tuple[int, tuple[RolloutRouteTraceItemV0, ...]]] = []
        current: list[RolloutRouteTraceItemV0] = []
        first_flat_index = 0
        shard_index = 0
        for flat_index, item in enumerate(flat_items):
            candidate = (*current, item)
            candidate_nbytes = _encoded_nbytes(
                batch_artifact_digest=batch_artifact_digest,
                batch_size=batch.batch_size,
                group_size=batch.group_size,
                shard_index=shard_index,
                first_flat_index=first_flat_index,
                items=candidate,
            )
            exceeds = (
                len(candidate) > max_items_per_shard
                or candidate_nbytes > max_shard_bytes
            )
            if exceeds and current:
                chunks.append((first_flat_index, tuple(current)))
                shard_index += 1
                first_flat_index = flat_index
                current = [item]
                single_nbytes = _encoded_nbytes(
                    batch_artifact_digest=batch_artifact_digest,
                    batch_size=batch.batch_size,
                    group_size=batch.group_size,
                    shard_index=shard_index,
                    first_flat_index=first_flat_index,
                    items=current,
                )
                if single_nbytes > max_shard_bytes:
                    raise RolloutRouteTraceError(
                        "one route trace item exceeds max_shard_bytes"
                    )
            elif exceeds:
                raise RolloutRouteTraceError(
                    "one route trace item exceeds max_shard_bytes"
                )
            else:
                current.append(item)
        if current:
            chunks.append((first_flat_index, tuple(current)))
        if len(chunks) > _MAX_SHARD_COUNT:
            raise RolloutRouteTraceError("route shard count exceeds the limit")

        shard_payloads = []
        descriptors = []
        for shard_index, (first_flat_index, items) in enumerate(chunks):
            shard = RolloutRouteTraceShardV0(
                batch_artifact_digest=batch_artifact_digest,
                batch_size=batch.batch_size,
                group_size=batch.group_size,
                shard_index=shard_index,
                first_flat_index=first_flat_index,
                items=items,
            )
            payload = RolloutRouteTraceShardCodecV0.dumps(shard)
            if len(payload) > max_shard_bytes:
                raise RolloutRouteTraceError(
                    "encoded route shard exceeds max_shard_bytes"
                )
            shard_payloads.append(payload)
            descriptors.append(
                RolloutRouteTraceShardDescriptorV0(
                    shard_index=shard_index,
                    first_flat_index=first_flat_index,
                    item_count=len(items),
                    payload_nbytes=shard.payload_nbytes,
                    encoded_nbytes=len(payload),
                    encoded_sha256=_sha256(payload),
                )
            )
        manifest = RolloutRouteTraceShardManifestV0(
            batch_artifact_digest=batch_artifact_digest,
            rollout_id=batch.rollout_id,
            policy_version=batch.policy_version,
            prompt_batch_digest=batch.prompt_batch_digest,
            response_batch_digest=batch.response_batch_digest,
            behavior_logprobs_digest=batch.behavior_logprobs_digest,
            batch_size=batch.batch_size,
            group_size=batch.group_size,
            total_payload_nbytes=batch.payload_nbytes,
            shards=tuple(descriptors),
            batch_schema_version=batch.schema_version,
        )
        return cls(manifest=manifest, shard_payloads=tuple(shard_payloads))

    @property
    def transport_nbytes(self) -> int:
        return len(self.manifest.dumps()) + self.manifest.encoded_nbytes

    def payloads_for_rank(
        self,
        rank: int,
        world_size: int,
    ) -> tuple[tuple[int, bytes], ...]:
        """Return this rank's deterministic shard subset without decoding it."""
        indices = self.manifest.shard_indices_for_rank(rank, world_size)
        return tuple((index, self.shard_payloads[index]) for index in indices)

    def assemble(self) -> RolloutRouteTraceBatchV0:
        return self.manifest.assemble(self.shard_payloads)
