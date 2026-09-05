"""Tests for the rollout route shard transport microbenchmark."""

import pytest

from astrai.moe import RouteTraceLevel
from scripts.benchmark_rollout_route_transport import run_benchmark


def test_transport_benchmark_reports_lossless_bounded_shards():
    report = run_benchmark(
        batch_size=2,
        group_size=2,
        prompt_tokens=2,
        response_tokens=4,
        layers=2,
        top_k=2,
        num_experts=8,
        level=RouteTraceLevel.IDS,
        max_shard_bytes=1024 * 1024,
        max_items_per_shard=1,
        warmups=0,
        repeats=1,
    )

    assert report["schema_version"] == 1
    assert report["shard_count"] == 4
    assert report["bytes"]["largest_shard"] <= 1024 * 1024
    assert report["bytes"]["transport_total"] > report["bytes"]["trace_payloads"]
    assert report["latency_us"]["build"]["median"] > 0
    assert report["latency_us"]["verify_all"]["median"] > 0
    assert report["latency_us"]["assemble"]["median"] > 0
    assert (
        sum(report["rank_assignment"]["2"]["per_rank_encoded_nbytes"])
        == report["bytes"]["encoded_shards"]
    )


def test_transport_benchmark_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="top_k"):
        run_benchmark(
            batch_size=1,
            group_size=1,
            prompt_tokens=1,
            response_tokens=1,
            layers=1,
            top_k=3,
            num_experts=2,
            level=RouteTraceLevel.IDS,
            max_shard_bytes=1024,
            max_items_per_shard=1,
            warmups=0,
            repeats=1,
        )
