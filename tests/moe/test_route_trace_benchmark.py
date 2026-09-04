"""Tests for deterministic RouteTraceV0 codec benchmark setup and reporting."""

import pytest
import torch

from astrai.moe import RouteTraceLevel
from scripts.benchmark_route_trace_codec import build_trace, run_benchmark


@pytest.mark.parametrize("level", list(RouteTraceLevel))
def test_codec_benchmark_reports_scope_bytes_and_latency(level):
    report = run_benchmark(
        tokens=4,
        layers=2,
        top_k=2,
        num_experts=8,
        level=level,
        device=torch.device("cpu"),
        warmups=0,
        repeats=2,
    )

    assert report["schema_version"] == 1
    assert report["benchmark"] == "astrai-route-trace-codec-v0"
    assert report["parameters"]["level"] == level.value
    assert report["bytes"]["wire"] > report["bytes"]["logical_payload"]
    assert report["bytes"]["wire_per_token"] > 0
    assert report["latency_us"]["serialize_median"] > 0
    assert report["latency_us"]["deserialize_median"] > 0
    assert len(report["artifact_digest"]) == 64


def test_codec_benchmark_builds_compact_ids_for_large_expert_count():
    trace = build_trace(
        tokens=2,
        layers=1,
        top_k=2,
        num_experts=257,
        level=RouteTraceLevel.IDS,
        device=torch.device("cpu"),
    )

    assert trace.topk_ids.dtype is torch.uint16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tokens": 0, "layers": 1, "top_k": 1, "num_experts": 2},
        {"tokens": 1, "layers": 1, "top_k": 3, "num_experts": 2},
    ],
)
def test_codec_benchmark_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        build_trace(
            **kwargs,
            level=RouteTraceLevel.IDS,
            device=torch.device("cpu"),
        )
