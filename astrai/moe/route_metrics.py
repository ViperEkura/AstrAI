"""Pure alignment diagnostics for compatible versioned MoE route traces."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from astrai.moe.route_trace import (
    RouteTraceV0,
    RouteTraceValidationError,
    require_semantically_aligned_traces,
)


@dataclass(frozen=True)
class RouteAlignmentMetricsV0:
    """Aggregated behavior/current route alignment without changing policy."""

    behavior_policy_version: int
    current_policy_version: int
    valid_token_count: int
    compared_token_layer_count: int
    ordered_topk_match_fraction: float
    unordered_topk_overlap_fraction: float
    route_set_flip_fraction: float
    selected_weight_l1_mean: float | None
    behavior_margin_mean: float | None
    behavior_fragile_fraction: float | None
    fragile_margin_threshold: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible scalar diagnostics."""
        return asdict(self)


def compute_topk_margin(router_scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """Compute top-k versus top-(k+1) margin in FP32.

    The explicit FP32 conversion gives BF16/FP16 callers one documented
    diagnostic reference. It does not choose or override expert IDs.
    """
    if not isinstance(router_scores, torch.Tensor):
        raise RouteTraceValidationError("router_scores must be a torch.Tensor")
    if not router_scores.is_floating_point() or router_scores.ndim < 1:
        raise RouteTraceValidationError("router_scores must be a floating tensor")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise RouteTraceValidationError("top_k must be a positive integer")
    if top_k >= router_scores.shape[-1]:
        raise RouteTraceValidationError(
            "top-k margin requires at least one unselected expert"
        )
    if not bool(torch.isfinite(router_scores).all().item()):
        raise RouteTraceValidationError("router_scores must be finite")
    values = torch.topk(router_scores.float(), top_k + 1, dim=-1, sorted=True).values
    return values[..., top_k - 1] - values[..., top_k]


def _valid_token_layer_mask(trace: RouteTraceV0) -> torch.Tensor:
    if trace.valid_mask is None:
        token_mask = torch.ones(
            trace.token_layout.token_count,
            dtype=torch.bool,
            device=trace.topk_ids.device,
        )
    else:
        token_mask = trace.valid_mask
    return token_mask[:, None].expand(-1, trace.router_schema.num_moe_layers)


def _selected_weight_l1(
    behavior: RouteTraceV0, current: RouteTraceV0
) -> torch.Tensor | None:
    if behavior.selected_weights is None or current.selected_weights is None:
        return None
    behavior_ids = behavior.topk_ids.to(dtype=torch.int64)
    current_ids = current.topk_ids.to(dtype=torch.int64)
    behavior_weights = behavior.selected_weights.float()
    current_weights = current.selected_weights.float()
    same_expert = behavior_ids.unsqueeze(-1) == current_ids.unsqueeze(-2)
    current_for_behavior = (
        same_expert.to(dtype=current_weights.dtype) * current_weights.unsqueeze(-2)
    ).sum(dim=-1)
    behavior_mass_delta = (behavior_weights - current_for_behavior).abs().sum(dim=-1)
    unmatched_current = ~same_expert.any(dim=-2)
    unmatched_current_mass = (current_weights * unmatched_current).sum(dim=-1)
    return behavior_mass_delta + unmatched_current_mass


def compare_route_traces(
    behavior: RouteTraceV0,
    current: RouteTraceV0,
    *,
    fragile_margin_threshold: float | None = None,
) -> RouteAlignmentMetricsV0:
    """Compare exact ordered IDs, expert sets, selected mass, and margin.

    Identity versions may differ because the purpose is to measure drift, but
    router semantics, token layout, validity mask, and device must match
    exactly. Missing optional tensors produce explicit ``None`` metrics rather
    than fabricated zero drift.
    """
    require_semantically_aligned_traces(behavior, current)
    if fragile_margin_threshold is not None:
        if (
            isinstance(fragile_margin_threshold, bool)
            or not isinstance(fragile_margin_threshold, (int, float))
            or not math.isfinite(fragile_margin_threshold)
            or fragile_margin_threshold < 0
        ):
            raise RouteTraceValidationError(
                "fragile_margin_threshold must be a finite non-negative number"
            )
        fragile_margin_threshold = float(fragile_margin_threshold)

    valid_positions = _valid_token_layer_mask(behavior)
    compared_positions = int(valid_positions.sum().item())
    if compared_positions == 0:
        raise RouteTraceValidationError(
            "route metrics require at least one valid token-layer"
        )

    behavior_ids = behavior.topk_ids.to(dtype=torch.int64)
    current_ids = current.topk_ids.to(dtype=torch.int64)
    ordered_match = (behavior_ids == current_ids).all(dim=-1)
    same_expert = behavior_ids.unsqueeze(-1) == current_ids.unsqueeze(-2)
    overlap = same_expert.any(dim=-1).float().mean(dim=-1)
    ordered_fraction = float(ordered_match[valid_positions].float().mean().item())
    overlap_fraction = float(overlap[valid_positions].mean().item())

    weight_l1 = _selected_weight_l1(behavior, current)
    weight_l1_mean = (
        float(weight_l1[valid_positions].mean().item())
        if weight_l1 is not None
        else None
    )

    margin_mean = None
    fragile_fraction = None
    if behavior.topk_margin is not None:
        valid_margins = behavior.topk_margin.float()[valid_positions]
        margin_mean = float(valid_margins.mean().item())
        if fragile_margin_threshold is not None:
            fragile_fraction = float(
                (valid_margins < fragile_margin_threshold).float().mean().item()
            )
    elif fragile_margin_threshold is not None:
        raise RouteTraceValidationError(
            "fragile margin diagnostics require behavior topk_margin"
        )

    return RouteAlignmentMetricsV0(
        behavior_policy_version=behavior.identity.policy_version,
        current_policy_version=current.identity.policy_version,
        valid_token_count=behavior.valid_token_count,
        compared_token_layer_count=compared_positions,
        ordered_topk_match_fraction=ordered_fraction,
        unordered_topk_overlap_fraction=overlap_fraction,
        route_set_flip_fraction=1.0 - overlap_fraction,
        selected_weight_l1_mean=weight_l1_mean,
        behavior_margin_mean=margin_mean,
        behavior_fragile_fraction=fragile_fraction,
        fragile_margin_threshold=fragile_margin_threshold,
    )
