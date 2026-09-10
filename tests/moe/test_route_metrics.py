"""Tests for pure route-alignment and near-tie diagnostics."""

from dataclasses import replace

import pytest
import torch

from astrai.moe import (
    PaddingLayout,
    RouteIdentityV0,
    RouterSchemaV0,
    RouteTokenLayoutV0,
    RouteTraceLevel,
    RouteTraceV0,
    RouteTraceValidationError,
    SelectedWeightSemantics,
    TokenSpanKind,
    canonical_json_digest,
    compare_route_traces,
    compute_topk_margin,
    pack_route_ids,
)


def _metric_trace(
    ids, weights=None, margins=None, *, policy_version=0, valid_mask=None
):
    padding_layout = (
        PaddingLayout.EXPLICIT_MASK if valid_mask is not None else PaddingLayout.NONE
    )
    schema = RouterSchemaV0(
        num_moe_layers=1,
        num_experts=8,
        top_k=2,
        expert_parallel_world_size=1,
        score_function="softmax-fp32-before-topk",
        score_dtype="float32",
        expert_id_ordering="score-descending-stable-index",
        selected_weight_semantics=SelectedWeightSemantics.SELECTED_RENORMALIZED,
        token_ordering="single-sequence-token-major",
        padding_layout=padding_layout,
        backend="unit-test",
        kernel_semantics_version="torch-topk-v1",
        parallel_layout_hash=canonical_json_digest({"placement": "local"}),
    )
    identity = RouteIdentityV0(
        policy_version=policy_version,
        model_revision=f"model-{policy_version}",
        router_state_version=policy_version,
        checkpoint_revision=f"checkpoint-{policy_version}",
        router_schema_hash=schema.fingerprint,
    )
    token_layout = RouteTokenLayoutV0(
        sample_id="sample",
        sequence_token_count=2,
        prompt_token_count=0,
        token_offset=0,
        token_count=2,
        span_kind=TokenSpanKind.FULL_SEQUENCE,
    )
    level = (
        RouteTraceLevel.IDS_WEIGHTS_MARGIN
        if weights is not None and margins is not None
        else RouteTraceLevel.IDS
    )
    mask_tensor = (
        torch.tensor(valid_mask, dtype=torch.bool) if valid_mask is not None else None
    )
    topk_ids = pack_route_ids(torch.tensor(ids).reshape(2, 1, 2), schema.num_experts)
    if mask_tensor is not None:
        topk_ids[~mask_tensor] = 0
    return RouteTraceV0(
        identity=identity,
        router_schema=schema,
        token_layout=token_layout,
        level=level,
        topk_ids=topk_ids,
        selected_weights=(
            torch.tensor(weights, dtype=torch.float16).reshape(2, 1, 2)
            if weights is not None
            else None
        ),
        topk_margin=(
            torch.tensor(margins, dtype=torch.float16).reshape(2, 1)
            if margins is not None
            else None
        ),
        valid_mask=mask_tensor,
    )


def test_alignment_metrics_distinguish_order_set_mass_and_fragility():
    behavior = _metric_trace(
        [[0, 1], [2, 3]],
        [[0.7, 0.3], [0.6, 0.4]],
        [0.01, 0.20],
        policy_version=4,
    )
    current = _metric_trace(
        [[1, 0], [2, 4]],
        [[0.3, 0.7], [0.5, 0.5]],
        [0.02, 0.30],
        policy_version=5,
    )

    metrics = compare_route_traces(behavior, current, fragile_margin_threshold=0.05)

    assert metrics.behavior_policy_version == 4
    assert metrics.current_policy_version == 5
    assert metrics.valid_token_count == 2
    assert metrics.compared_token_layer_count == 2
    assert metrics.ordered_topk_match_fraction == 0.0
    assert metrics.unordered_topk_overlap_fraction == pytest.approx(0.75)
    assert metrics.route_set_flip_fraction == pytest.approx(0.25)
    assert metrics.selected_weight_l1_mean == pytest.approx(0.5, abs=1e-3)
    assert metrics.behavior_margin_mean == pytest.approx(0.105, abs=1e-3)
    assert metrics.behavior_fragile_fraction == pytest.approx(0.5)
    assert metrics.to_dict()["fragile_margin_threshold"] == 0.05


def test_ids_only_metrics_do_not_fabricate_gate_or_margin_values():
    behavior = _metric_trace([[0, 1], [2, 3]], policy_version=0)
    current = _metric_trace([[0, 1], [2, 4]], policy_version=1)

    metrics = compare_route_traces(behavior, current)

    assert metrics.ordered_topk_match_fraction == 0.5
    assert metrics.unordered_topk_overlap_fraction == 0.75
    assert metrics.selected_weight_l1_mean is None
    assert metrics.behavior_margin_mean is None
    assert metrics.behavior_fragile_fraction is None
    with pytest.raises(RouteTraceValidationError, match="require behavior topk_margin"):
        compare_route_traces(behavior, current, fragile_margin_threshold=0.1)


def test_metrics_require_exact_schema_layout_mask_and_device_semantics():
    behavior = _metric_trace([[0, 1], [2, 3]], valid_mask=[True, False])
    current = _metric_trace([[0, 1], [2, 3]], policy_version=1, valid_mask=[True, True])
    with pytest.raises(RouteTraceValidationError, match="validity masks"):
        compare_route_traces(behavior, current)

    schema = replace(current.router_schema, backend="different")
    identity = replace(current.identity, router_schema_hash=schema.fingerprint)
    mismatched_schema = replace(current, router_schema=schema, identity=identity)
    with pytest.raises(RouteTraceValidationError, match="router schema"):
        compare_route_traces(behavior, mismatched_schema)


def test_metrics_reject_empty_valid_population_and_bad_threshold():
    behavior = _metric_trace([[0, 0], [0, 0]], valid_mask=[False, False])
    current = _metric_trace(
        [[0, 0], [0, 0]], policy_version=1, valid_mask=[False, False]
    )
    with pytest.raises(RouteTraceValidationError, match="at least one valid"):
        compare_route_traces(behavior, current)
    valid = _metric_trace([[0, 1], [2, 3]])
    with pytest.raises(RouteTraceValidationError, match="finite non-negative"):
        compare_route_traces(valid, valid, fragile_margin_threshold=float("nan"))


def test_bfloat16_near_tie_margin_uses_explicit_fp32_reference():
    scores = torch.tensor(
        [[1.0, 0.984375, 0.5], [1.0, 1.0, 0.25]], dtype=torch.bfloat16
    )

    margin = compute_topk_margin(scores, top_k=1)

    torch.testing.assert_close(margin, compute_topk_margin(scores.float(), top_k=1))
    assert margin.dtype is torch.float32
    assert margin.tolist() == pytest.approx([0.015625, 0.0])


@pytest.mark.parametrize("top_k", [0, 3, True])
def test_margin_rejects_invalid_top_k(top_k):
    with pytest.raises(RouteTraceValidationError, match="top_k|top-k"):
        compute_topk_margin(torch.ones(2, 3), top_k)


def test_margin_rejects_nonfinite_scores():
    scores = torch.tensor([[1.0, float("nan"), 0.0]])
    with pytest.raises(RouteTraceValidationError, match="finite"):
        compute_topk_margin(scores, 1)
