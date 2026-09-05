"""Tests for forward/checkpoint-recompute route diagnostics."""

from dataclasses import replace

import pytest
import torch

from astrai.moe import (
    ROUTE_RECOMPUTE_SCHEMA_VERSION,
    RecomputeRouteValidationError,
    RouteCheckpointPairV0,
    RouteRecomputeDiagnosticsV0,
    RouteRecomputeSummaryV0,
    compare_recompute_routes,
    synchronize_route_recompute_summary,
)
from astrai.parallel import get_rank, get_world_size, spawn_parallel_fn


def _route_layers():
    return (
        torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int64),
        torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int64),
    )


def test_exact_report_is_content_addressed_and_dtype_normalized():
    forward = _route_layers()
    recompute = tuple(layer.to(torch.int32) for layer in forward)

    report = compare_recompute_routes(forward, recompute)

    assert report.schema_version == ROUTE_RECOMPUTE_SCHEMA_VERSION
    assert report.exact_match is True
    assert report.forward_route_hash == report.recompute_route_hash
    assert report.layer_count == 2
    assert report.token_count == 5
    assert report.slot_count == 10
    assert report.mismatch_layer_count == 0
    assert report.mismatch_token_count == 0
    assert report.mismatch_slot_count == 0
    assert report.to_dict()["exact_match"] is True


def test_report_counts_ordered_slot_token_and_layer_mismatches():
    forward = list(_route_layers())
    recompute = [layer.clone() for layer in forward]
    recompute[0][1, 0] = 6
    recompute[1][0, 1] = recompute[1][0, 1].flip(0)

    report = compare_recompute_routes(forward, recompute)

    assert report.exact_match is False
    assert report.forward_route_hash != report.recompute_route_hash
    assert report.mismatch_layer_count == 2
    assert report.mismatch_token_count == 2
    assert report.mismatch_slot_count == 3


@pytest.mark.parametrize(
    ("forward", "recompute", "message"),
    [
        (_route_layers(), _route_layers()[:1], "layer counts"),
        (
            (torch.tensor([[0, 1]], dtype=torch.int64),),
            (torch.tensor([[[0, 1]]], dtype=torch.int64),),
            "shapes differ",
        ),
        (
            (torch.tensor([[0.0, 1.0]]),),
            (torch.tensor([[0, 1]]),),
            "integer dtype",
        ),
        (
            (torch.tensor([[-1, 1]]),),
            (torch.tensor([[0, 1]]),),
            "negative expert ID",
        ),
        (
            (torch.tensor([[1, 1]]),),
            (torch.tensor([[0, 1]]),),
            "duplicate experts",
        ),
        ((), (), "at least one routed layer"),
    ],
)
def test_compare_rejects_ambiguous_or_malformed_routes(forward, recompute, message):
    with pytest.raises(RecomputeRouteValidationError, match=message):
        compare_recompute_routes(forward, recompute)


def test_report_and_summary_constructors_fail_closed():
    report = compare_recompute_routes(_route_layers(), _route_layers())
    with pytest.raises(RecomputeRouteValidationError, match="unsupported"):
        replace(report, schema_version=1)
    with pytest.raises(RecomputeRouteValidationError, match="hashes"):
        replace(report, recompute_route_hash="0" * 64)
    with pytest.raises(RecomputeRouteValidationError, match="must equal"):
        RouteRecomputeSummaryV0(compared_pair_count=1)
    with pytest.raises(RecomputeRouteValidationError, match="exceeds forward"):
        RouteRecomputeSummaryV0(completed_forward_pair_count=1)
    with pytest.raises(RecomputeRouteValidationError, match="zero or one"):
        RouteRecomputeSummaryV0(rank_observation_inconsistent=2)


def test_checkpoint_pair_records_mismatch_without_retaining_route_tensors():
    diagnostics = RouteRecomputeDiagnosticsV0()
    pair = RouteCheckpointPairV0(diagnostics)
    forward_context, recompute_context = pair.context_fn()
    forward = _route_layers()[0]
    recompute = forward.clone()
    recompute[1, 0] = 6

    with forward_context:
        pair.observe({"router_stats": {"topk_indices": forward}})
    with recompute_context:
        pair.observe({"router_stats": {"topk_indices": recompute}})

    summary = diagnostics.snapshot()
    assert summary.forward_pair_count == 1
    assert summary.completed_forward_pair_count == 1
    assert summary.compared_pair_count == 1
    assert summary.mismatch_pair_count == 1
    assert summary.mismatch_token_count == 1
    assert summary.unrecomputed_pair_count == 0
    assert summary.has_failure is True
    assert diagnostics.last_report is not None
    assert diagnostics.last_report.mismatch_slot_count == 1


def test_checkpoint_pair_ignores_dense_output_and_reports_incomplete_routes():
    diagnostics = RouteRecomputeDiagnosticsV0()
    dense_pair = RouteCheckpointPairV0(diagnostics)
    forward_context, recompute_context = dense_pair.context_fn()
    with forward_context:
        dense_pair.observe({"router_stats": None})
    with recompute_context:
        dense_pair.observe({"router_stats": None})
    assert diagnostics.snapshot() == RouteRecomputeSummaryV0()

    routed_pair = RouteCheckpointPairV0(diagnostics)
    forward_context, _ = routed_pair.context_fn()
    with forward_context:
        routed_pair.observe({"router_stats": {"topk_indices": _route_layers()[0]}})
    summary = diagnostics.snapshot()
    assert summary.unrecomputed_pair_count == 1
    assert summary.has_failure is True

    diagnostics.reset()
    invalid_pair = RouteCheckpointPairV0(diagnostics)
    forward_context, _ = invalid_pair.context_fn()
    with forward_context:
        invalid_pair.observe({"router_stats": {}})
    summary = diagnostics.snapshot()
    assert summary.forward_pair_count == 1
    assert summary.completed_forward_pair_count == 1
    assert summary.invalid_pair_count == 1
    assert summary.unrecomputed_pair_count == 0
    assert diagnostics.last_invalid_reason == (
        "router_stats[0] must contain topk_indices"
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_summary_merge_marks_only_inconsistent_rank_geometry(world_size):
    exact = RouteRecomputeSummaryV0(
        forward_pair_count=1,
        completed_forward_pair_count=1,
        compared_pair_count=1,
        exact_pair_count=1,
        compared_layer_count=2,
        compared_token_count=5,
        compared_slot_count=10,
    )
    summaries = [exact for _ in range(world_size)]
    merged = RouteRecomputeSummaryV0.merge(summaries)
    assert merged.forward_pair_count == world_size
    assert merged.compared_pair_count == world_size
    assert merged.rank_observation_inconsistent == 0

    summaries[-1] = RouteRecomputeSummaryV0()
    merged = RouteRecomputeSummaryV0.merge(summaries)
    assert merged.rank_observation_inconsistent == 1
    assert merged.has_failure is True


def _distributed_summary_worker():
    rank = get_rank()
    world_size = get_world_size()
    forward = _route_layers()[0]
    recompute = forward.clone()
    if rank % 2:
        recompute[0, 0] = 6
    diagnostics = RouteRecomputeDiagnosticsV0()
    diagnostics.record_forward_pair()
    diagnostics.record_report(compare_recompute_routes(forward, recompute))

    merged = synchronize_route_recompute_summary(diagnostics.snapshot(), device="cpu")

    assert merged.forward_pair_count == world_size
    assert merged.compared_pair_count == world_size
    assert merged.exact_pair_count == world_size - world_size // 2
    assert merged.mismatch_pair_count == world_size // 2
    assert merged.mismatch_token_count == world_size // 2
    assert merged.rank_observation_inconsistent == 0

    uneven_local = RouteRecomputeSummaryV0() if rank == 0 else diagnostics.snapshot()
    uneven_merged = synchronize_route_recompute_summary(uneven_local, device="cpu")
    assert uneven_merged.forward_pair_count == world_size - 1
    assert uneven_merged.compared_pair_count == world_size - 1
    assert uneven_merged.rank_observation_inconsistent == 1
    assert uneven_merged.has_failure is True


@pytest.mark.parametrize("world_size", [2, 4])
def test_summary_synchronizes_on_real_gloo_process_groups(world_size):
    spawn_parallel_fn(
        _distributed_summary_worker,
        world_size=world_size,
        backend="gloo",
        device_type="cpu",
    )
