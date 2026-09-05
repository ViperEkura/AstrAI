"""Diagnostics for MoE routes across activation-checkpoint recomputation.

The helpers in this module observe route IDs that a checkpointed module already
returns. They do not choose experts, alter dispatch, or replay a route. Runtime
integration can therefore remain opt-in while still detecting when backward
recomputation followed a different sparse path from the original forward.
"""

from __future__ import annotations

import hashlib
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch import Tensor

from astrai.moe.route_trace import canonical_json_digest

ROUTE_RECOMPUTE_SCHEMA_VERSION = 0

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_INTEGER_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
)
_SUMMARY_COUNT_FIELDS = (
    "forward_pair_count",
    "completed_forward_pair_count",
    "compared_pair_count",
    "exact_pair_count",
    "mismatch_pair_count",
    "invalid_pair_count",
    "compared_layer_count",
    "compared_token_count",
    "compared_slot_count",
    "mismatch_layer_count",
    "mismatch_token_count",
    "mismatch_slot_count",
    "rank_observation_inconsistent",
)

__all__ = [
    "ROUTE_RECOMPUTE_SCHEMA_VERSION",
    "RecomputeRouteMismatchError",
    "RecomputeRoutePairReportV0",
    "RecomputeRouteValidationError",
    "RouteCheckpointPairV0",
    "RouteRecomputeDiagnosticsV0",
    "RouteRecomputeSummaryV0",
    "compare_recompute_routes",
    "synchronize_route_recompute_summary",
]


class RecomputeRouteValidationError(ValueError):
    """Route observations cannot be compared under the declared contract."""


class RecomputeRouteMismatchError(RuntimeError):
    """Checkpoint recomputation produced an unsafe route diagnostic."""


def _require_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecomputeRouteValidationError(f"{name} must be a non-negative integer")
    return value


def _require_digest(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise RecomputeRouteValidationError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _normalize_route_tensor(name: str, tensor: Tensor) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise RecomputeRouteValidationError(f"{name} must be a torch.Tensor")
    if tensor.layout is not torch.strided or tensor.device.type == "meta":
        raise RecomputeRouteValidationError(
            f"{name} must be a materialized strided tensor"
        )
    if tensor.dtype not in _INTEGER_DTYPES:
        raise RecomputeRouteValidationError(f"{name} must use an integer dtype")
    if tensor.ndim < 2 or tensor.shape[-1] < 1:
        raise RecomputeRouteValidationError(
            f"{name} must have token dimensions followed by a positive top-k axis"
        )
    if sys.byteorder != "little":
        raise RecomputeRouteValidationError(
            "route recompute diagnostics require a little-endian host"
        )

    normalized = tensor.detach().to(dtype=torch.int64).contiguous().cpu().clone()
    if normalized.numel() and int(normalized.min().item()) < 0:
        raise RecomputeRouteValidationError(f"{name} contains a negative expert ID")
    rows = normalized.reshape(-1, normalized.shape[-1])
    if rows.shape[-1] > 1 and rows.numel():
        ordered = rows.sort(dim=-1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
            raise RecomputeRouteValidationError(
                f"{name} contains duplicate experts in a top-k row"
            )
    return normalized


def _normalize_route_layers(
    value: Tensor | Sequence[Tensor], *, label: str
) -> tuple[Tensor, ...]:
    if isinstance(value, Tensor):
        layers = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        layers = tuple(value)
    else:
        raise RecomputeRouteValidationError(
            f"{label} routes must be a tensor or a sequence of tensors"
        )
    if not layers:
        raise RecomputeRouteValidationError(
            f"{label} routes must contain at least one routed layer"
        )
    return tuple(
        _normalize_route_tensor(f"{label} routes[{index}]", layer)
        for index, layer in enumerate(layers)
    )


def _layer_descriptor(tensor: Tensor) -> dict[str, Any]:
    raw = tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
    return {
        "dtype": "int64",
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(tensor.shape),
    }


def _route_layers_digest(layers: tuple[Tensor, ...]) -> str:
    return canonical_json_digest(
        {
            "layers": [_layer_descriptor(layer) for layer in layers],
            "schema_version": ROUTE_RECOMPUTE_SCHEMA_VERSION,
        }
    )


@dataclass(frozen=True)
class RecomputeRoutePairReportV0:
    """Exact route comparison for one checkpointed module invocation."""

    forward_route_hash: str
    recompute_route_hash: str
    layer_count: int
    token_count: int
    slot_count: int
    mismatch_layer_count: int
    mismatch_token_count: int
    mismatch_slot_count: int
    schema_version: int = ROUTE_RECOMPUTE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROUTE_RECOMPUTE_SCHEMA_VERSION:
            raise RecomputeRouteValidationError(
                f"unsupported route recompute schema version {self.schema_version}"
            )
        _require_digest("forward_route_hash", self.forward_route_hash)
        _require_digest("recompute_route_hash", self.recompute_route_hash)
        for name in (
            "layer_count",
            "token_count",
            "slot_count",
            "mismatch_layer_count",
            "mismatch_token_count",
            "mismatch_slot_count",
        ):
            _require_non_negative_int(name, getattr(self, name))
        if self.layer_count < 1:
            raise RecomputeRouteValidationError("layer_count must be positive")
        if self.mismatch_layer_count > self.layer_count:
            raise RecomputeRouteValidationError(
                "mismatch_layer_count exceeds layer_count"
            )
        if self.mismatch_token_count > self.token_count:
            raise RecomputeRouteValidationError(
                "mismatch_token_count exceeds token_count"
            )
        if self.mismatch_slot_count > self.slot_count:
            raise RecomputeRouteValidationError(
                "mismatch_slot_count exceeds slot_count"
            )
        no_mismatch = not (
            self.mismatch_layer_count
            or self.mismatch_token_count
            or self.mismatch_slot_count
        )
        if no_mismatch != (self.forward_route_hash == self.recompute_route_hash):
            raise RecomputeRouteValidationError(
                "route hashes and mismatch counts disagree"
            )

    @property
    def exact_match(self) -> bool:
        return self.mismatch_slot_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_match": self.exact_match,
            "forward_route_hash": self.forward_route_hash,
            "layer_count": self.layer_count,
            "mismatch_layer_count": self.mismatch_layer_count,
            "mismatch_slot_count": self.mismatch_slot_count,
            "mismatch_token_count": self.mismatch_token_count,
            "recompute_route_hash": self.recompute_route_hash,
            "schema_version": self.schema_version,
            "slot_count": self.slot_count,
            "token_count": self.token_count,
        }


def _compare_normalized_routes(
    forward_layers: tuple[Tensor, ...], recompute_layers: tuple[Tensor, ...]
) -> RecomputeRoutePairReportV0:
    if len(forward_layers) != len(recompute_layers):
        raise RecomputeRouteValidationError(
            "forward and recompute routed-layer counts differ"
        )

    mismatch_layer_count = 0
    mismatch_token_count = 0
    mismatch_slot_count = 0
    token_count = 0
    slot_count = 0
    for index, (forward, recompute) in enumerate(zip(forward_layers, recompute_layers)):
        if forward.shape != recompute.shape:
            raise RecomputeRouteValidationError(
                f"forward and recompute route shapes differ at layer {index}"
            )
        forward_rows = forward.reshape(-1, forward.shape[-1])
        recompute_rows = recompute.reshape(-1, recompute.shape[-1])
        slot_mismatch = forward_rows != recompute_rows
        token_mismatch = slot_mismatch.any(dim=-1)
        layer_mismatch = bool(token_mismatch.any().item())
        mismatch_layer_count += int(layer_mismatch)
        mismatch_token_count += int(token_mismatch.sum().item())
        mismatch_slot_count += int(slot_mismatch.sum().item())
        token_count += forward_rows.shape[0]
        slot_count += forward.numel()

    return RecomputeRoutePairReportV0(
        forward_route_hash=_route_layers_digest(forward_layers),
        recompute_route_hash=_route_layers_digest(recompute_layers),
        layer_count=len(forward_layers),
        token_count=token_count,
        slot_count=slot_count,
        mismatch_layer_count=mismatch_layer_count,
        mismatch_token_count=mismatch_token_count,
        mismatch_slot_count=mismatch_slot_count,
    )


def compare_recompute_routes(
    forward_routes: Tensor | Sequence[Tensor],
    recompute_routes: Tensor | Sequence[Tensor],
) -> RecomputeRoutePairReportV0:
    """Compare ordered top-k IDs from forward and checkpoint recomputation."""
    forward_layers = _normalize_route_layers(forward_routes, label="forward")
    recompute_layers = _normalize_route_layers(recompute_routes, label="recompute")
    return _compare_normalized_routes(forward_layers, recompute_layers)


@dataclass(frozen=True)
class RouteRecomputeSummaryV0:
    """Bounded counters for one batch or an exact merge across ranks."""

    forward_pair_count: int = 0
    completed_forward_pair_count: int = 0
    compared_pair_count: int = 0
    exact_pair_count: int = 0
    mismatch_pair_count: int = 0
    invalid_pair_count: int = 0
    compared_layer_count: int = 0
    compared_token_count: int = 0
    compared_slot_count: int = 0
    mismatch_layer_count: int = 0
    mismatch_token_count: int = 0
    mismatch_slot_count: int = 0
    rank_observation_inconsistent: int = 0
    schema_version: int = ROUTE_RECOMPUTE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROUTE_RECOMPUTE_SCHEMA_VERSION:
            raise RecomputeRouteValidationError(
                f"unsupported route recompute schema version {self.schema_version}"
            )
        for name in _SUMMARY_COUNT_FIELDS:
            _require_non_negative_int(name, getattr(self, name))
        if self.rank_observation_inconsistent not in (0, 1):
            raise RecomputeRouteValidationError(
                "rank_observation_inconsistent must be zero or one"
            )
        if self.exact_pair_count + self.mismatch_pair_count != self.compared_pair_count:
            raise RecomputeRouteValidationError(
                "exact and mismatch pair counts must equal compared_pair_count"
            )
        if self.completed_forward_pair_count > self.forward_pair_count:
            raise RecomputeRouteValidationError(
                "completed_forward_pair_count exceeds forward_pair_count"
            )
        if self.compared_pair_count > self.completed_forward_pair_count:
            raise RecomputeRouteValidationError(
                "compared_pair_count exceeds completed_forward_pair_count"
            )
        if self.mismatch_layer_count > self.compared_layer_count:
            raise RecomputeRouteValidationError(
                "mismatch_layer_count exceeds compared_layer_count"
            )
        if self.mismatch_token_count > self.compared_token_count:
            raise RecomputeRouteValidationError(
                "mismatch_token_count exceeds compared_token_count"
            )
        if self.mismatch_slot_count > self.compared_slot_count:
            raise RecomputeRouteValidationError(
                "mismatch_slot_count exceeds compared_slot_count"
            )

    @property
    def unrecomputed_pair_count(self) -> int:
        return self.forward_pair_count - self.completed_forward_pair_count

    @property
    def exact_match_fraction(self) -> float | None:
        if self.compared_pair_count == 0:
            return None
        return self.exact_pair_count / self.compared_pair_count

    @property
    def has_failure(self) -> bool:
        return bool(
            self.mismatch_pair_count
            or self.invalid_pair_count
            or self.unrecomputed_pair_count
            or self.rank_observation_inconsistent
        )

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in _SUMMARY_COUNT_FIELDS}
        result.update(
            {
                "exact_match_fraction": self.exact_match_fraction,
                "has_failure": self.has_failure,
                "schema_version": self.schema_version,
                "unrecomputed_pair_count": self.unrecomputed_pair_count,
            }
        )
        return result

    def to_metrics(self) -> dict[str, float]:
        metrics = {
            "route_recompute_compared_pairs": float(self.compared_pair_count),
            "route_recompute_completed_forward_pairs": float(
                self.completed_forward_pair_count
            ),
            "route_recompute_forward_pairs": float(self.forward_pair_count),
            "route_recompute_invalid_pairs": float(self.invalid_pair_count),
            "route_recompute_mismatch_layers": float(self.mismatch_layer_count),
            "route_recompute_mismatch_pairs": float(self.mismatch_pair_count),
            "route_recompute_mismatch_slots": float(self.mismatch_slot_count),
            "route_recompute_mismatch_tokens": float(self.mismatch_token_count),
            "route_recompute_rank_inconsistent": float(
                self.rank_observation_inconsistent
            ),
            "route_recompute_unrecomputed_pairs": float(self.unrecomputed_pair_count),
        }
        if self.exact_match_fraction is not None:
            metrics["forward_recompute_route_match"] = self.exact_match_fraction
        return metrics

    def _count_tuple(self) -> tuple[int, ...]:
        return tuple(getattr(self, name) for name in _SUMMARY_COUNT_FIELDS)

    @classmethod
    def _from_count_tuple(cls, counts: Sequence[int]) -> RouteRecomputeSummaryV0:
        if len(counts) != len(_SUMMARY_COUNT_FIELDS):
            raise RecomputeRouteValidationError(
                "route recompute summary count vector has the wrong length"
            )
        return cls(**dict(zip(_SUMMARY_COUNT_FIELDS, counts)))

    @classmethod
    def merge(
        cls, summaries: Sequence[RouteRecomputeSummaryV0]
    ) -> RouteRecomputeSummaryV0:
        """Merge rank-local summaries and flag incomplete rank geometry."""
        if isinstance(summaries, (str, bytes)) or not isinstance(summaries, Sequence):
            raise RecomputeRouteValidationError("summaries must be a sequence")
        summaries = tuple(summaries)
        if not summaries or any(
            not isinstance(summary, RouteRecomputeSummaryV0) for summary in summaries
        ):
            raise RecomputeRouteValidationError(
                "summaries must contain RouteRecomputeSummaryV0 values"
            )
        counts = {
            name: sum(getattr(summary, name) for summary in summaries)
            for name in _SUMMARY_COUNT_FIELDS
            if name != "rank_observation_inconsistent"
        }
        geometries = {
            (
                summary.forward_pair_count,
                summary.completed_forward_pair_count,
                summary.compared_layer_count,
            )
            for summary in summaries
        }
        counts["rank_observation_inconsistent"] = int(
            any(summary.rank_observation_inconsistent for summary in summaries)
            or len(geometries) != 1
        )
        return cls(**counts)


class RouteRecomputeDiagnosticsV0:
    """Thread-safe bounded state shared by checkpointed modules in one batch."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = {name: 0 for name in _SUMMARY_COUNT_FIELDS}
        self._last_report: RecomputeRoutePairReportV0 | None = None
        self._last_invalid_reason: str | None = None

    def reset(self) -> None:
        with self._lock:
            self._counts = {name: 0 for name in _SUMMARY_COUNT_FIELDS}
            self._last_report = None
            self._last_invalid_reason = None

    def record_forward_pair(self) -> None:
        with self._lock:
            self._counts["forward_pair_count"] += 1

    def record_report(self, report: RecomputeRoutePairReportV0) -> None:
        if not isinstance(report, RecomputeRoutePairReportV0):
            raise RecomputeRouteValidationError(
                "report must be RecomputeRoutePairReportV0"
            )
        with self._lock:
            self._counts["completed_forward_pair_count"] += 1
            self._counts["compared_pair_count"] += 1
            self._counts["exact_pair_count"] += int(report.exact_match)
            self._counts["mismatch_pair_count"] += int(not report.exact_match)
            self._counts["compared_layer_count"] += report.layer_count
            self._counts["compared_token_count"] += report.token_count
            self._counts["compared_slot_count"] += report.slot_count
            self._counts["mismatch_layer_count"] += report.mismatch_layer_count
            self._counts["mismatch_token_count"] += report.mismatch_token_count
            self._counts["mismatch_slot_count"] += report.mismatch_slot_count
            self._last_report = report

    def record_invalid_pair(
        self, reason: str, *, completes_forward_pair: bool = False
    ) -> None:
        if not isinstance(reason, str) or not reason:
            raise RecomputeRouteValidationError("invalid-pair reason must be text")
        with self._lock:
            self._counts["invalid_pair_count"] += 1
            self._counts["completed_forward_pair_count"] += int(completes_forward_pair)
            self._last_invalid_reason = reason[:1024]

    def snapshot(self) -> RouteRecomputeSummaryV0:
        with self._lock:
            return RouteRecomputeSummaryV0(**self._counts)

    @property
    def last_report(self) -> RecomputeRoutePairReportV0 | None:
        with self._lock:
            return self._last_report

    @property
    def last_invalid_reason(self) -> str | None:
        with self._lock:
            return self._last_invalid_reason


def _extract_route_layers(output: Any) -> tuple[Tensor, ...]:
    if not isinstance(output, Mapping):
        return ()
    stats = output.get("router_stats")
    if stats is None:
        return ()
    if isinstance(stats, Mapping):
        entries = (stats,)
    elif isinstance(stats, Sequence) and not isinstance(stats, (str, bytes)):
        entries = tuple(stats)
    else:
        raise RecomputeRouteValidationError(
            "router_stats must be a mapping or a sequence of mappings"
        )
    layers = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or "topk_indices" not in entry:
            raise RecomputeRouteValidationError(
                f"router_stats[{index}] must contain topk_indices"
            )
        layers.append(
            _normalize_route_tensor(
                f"router_stats[{index}].topk_indices", entry["topk_indices"]
            )
        )
    return tuple(layers)


class RouteCheckpointPairV0:
    """Per-checkpoint observer correlated by PyTorch's two phase contexts."""

    def __init__(self, diagnostics: RouteRecomputeDiagnosticsV0) -> None:
        if not isinstance(diagnostics, RouteRecomputeDiagnosticsV0):
            raise RecomputeRouteValidationError(
                "diagnostics must be RouteRecomputeDiagnosticsV0"
            )
        self._diagnostics = diagnostics
        self._phase: str | None = None
        self._forward_routes: tuple[Tensor, ...] | None = None
        self._forward_observed = False
        self._forward_counted = False
        self._forward_completed = False
        self._invalid = False
        self._completed = False

    def _record_invalid(self, reason: str) -> None:
        if not self._invalid:
            completes_forward_pair = (
                self._forward_counted and not self._forward_completed
            )
            self._diagnostics.record_invalid_pair(
                reason,
                completes_forward_pair=completes_forward_pair,
            )
            self._forward_completed = self._forward_completed or completes_forward_pair
            self._invalid = True

    @contextmanager
    def _phase_context(self, phase: str) -> Iterator[None]:
        if self._phase is not None:
            self._record_invalid("route checkpoint phase contexts overlapped")
        previous = self._phase
        self._phase = phase
        try:
            yield
        finally:
            self._phase = previous

    def context_fn(
        self,
    ) -> tuple[AbstractContextManager[None], AbstractContextManager[None]]:
        """Return original-forward and recompute contexts for checkpoint()."""
        return (
            self._phase_context("forward"),
            self._phase_context("recompute"),
        )

    def observe(self, output: Any) -> None:
        """Observe one module output without retaining its autograd tensors."""
        if self._completed:
            self._record_invalid("route checkpoint pair produced extra observations")
            return
        try:
            layers = _extract_route_layers(output)
        except RecomputeRouteValidationError as exc:
            if self._phase == "forward" and not self._forward_observed:
                self._diagnostics.record_forward_pair()
                self._forward_observed = True
                self._forward_counted = True
            self._record_invalid(str(exc))
            if self._phase == "recompute":
                self._completed = True
            return

        if self._phase == "forward":
            if self._forward_observed:
                self._record_invalid("checkpoint pair observed forward more than once")
                return
            self._forward_observed = True
            self._forward_routes = layers
            if layers:
                self._diagnostics.record_forward_pair()
                self._forward_counted = True
            return

        if self._phase != "recompute":
            self._record_invalid("route observation occurred outside checkpoint phases")
            return
        self._completed = True
        if self._invalid:
            return
        if not self._forward_observed or self._forward_routes is None:
            self._record_invalid(
                "recompute route arrived without a forward observation"
            )
            return
        if not self._forward_routes and not layers:
            self._forward_routes = None
            return
        if not self._forward_routes or not layers:
            self._record_invalid(
                "forward and recompute disagree on whether routed layers were present"
            )
            self._forward_routes = None
            return
        try:
            report = _compare_normalized_routes(self._forward_routes, layers)
        except RecomputeRouteValidationError as exc:
            self._record_invalid(str(exc))
            self._forward_routes = None
            return
        self._forward_routes = None
        self._diagnostics.record_report(report)
        self._forward_completed = True


def synchronize_route_recompute_summary(
    summary: RouteRecomputeSummaryV0,
    *,
    device: torch.device | str | None = None,
) -> RouteRecomputeSummaryV0:
    """Return the same merged summary on every initialized distributed rank."""
    if not isinstance(summary, RouteRecomputeSummaryV0):
        raise RecomputeRouteValidationError("summary must be RouteRecomputeSummaryV0")
    if not dist.is_available() or not dist.is_initialized():
        return summary
    if device is None:
        backend = str(dist.get_backend()).lower()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if "nccl" in backend
            else torch.device("cpu")
        )
    count_tensor = torch.tensor(
        summary._count_tuple(), dtype=torch.int64, device=device
    )
    gathered = [torch.empty_like(count_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, count_tensor)
    rank_summaries = tuple(
        RouteRecomputeSummaryV0._from_count_tuple(item.cpu().tolist())
        for item in gathered
    )
    return RouteRecomputeSummaryV0.merge(rank_summaries)
