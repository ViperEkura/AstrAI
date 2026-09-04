"""Tests for the rollout route-trace binding microbenchmark."""

import pytest
import torch

from astrai.moe import RouteTraceLevel
from astrai.trainer.route_trace import rollout_route_sample_id
from scripts.benchmark_rollout_route_binding import (
    build_rollout_inputs,
    run_benchmark,
)


@pytest.mark.parametrize("level", list(RouteTraceLevel))
def test_binding_benchmark_reports_explicit_scope_bytes_and_latency(level):
    report = run_benchmark(
        batch_size=2,
        group_size=2,
        prompt_tokens=3,
        response_tokens=4,
        layers=2,
        top_k=2,
        num_experts=8,
        level=level,
        device=torch.device("cpu"),
        warmups=0,
        bind_repeats=1,
        validate_repeats=2,
    )

    assert report["schema_version"] == 1
    assert report["benchmark"] == "astrai-rollout-route-trace-binding-v0"
    assert report["parameters"]["level"] == level.value
    assert report["bytes"]["trace_payloads"] > 0
    assert report["bytes"]["identity_manifest"] > 0
    assert report["latency_us"]["bind_median"] > 0
    assert report["latency_us"]["validate_median"] > 0
    assert len(report["artifact_digest"]) == 64


def test_binding_benchmark_uses_coordinate_derived_sample_ids():
    tensors, traces = build_rollout_inputs(
        batch_size=2,
        group_size=2,
        prompt_tokens=3,
        response_tokens=4,
        layers=2,
        top_k=2,
        num_experts=8,
        level=RouteTraceLevel.IDS,
        device=torch.device("cpu"),
    )

    assert tensors["responses"].shape == (2, 2, 4)
    assert traces[1][1].token_layout.sample_id == rollout_route_sample_id(
        "rollout-route-binding-benchmark", 1, 1
    )


def test_binding_benchmark_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="dimensions"):
        build_rollout_inputs(
            batch_size=1,
            group_size=1,
            prompt_tokens=1,
            response_tokens=1,
            layers=1,
            top_k=3,
            num_experts=2,
            level=RouteTraceLevel.IDS,
            device=torch.device("cpu"),
        )
