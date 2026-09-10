"""Tests for the MoE checkpoint route-validation microbenchmark."""

import pytest
import torch

from scripts.benchmark_moe_recompute_validation import run_benchmark


def test_recompute_benchmark_reports_route_and_gradient_parity():
    report = run_benchmark(
        tokens=4,
        dim=8,
        dim_ffn=16,
        num_experts=4,
        top_k=2,
        device=torch.device("cpu"),
        warmups=0,
        repeats=1,
    )

    assert report["schema_version"] == 1
    assert report["benchmark"] == "astrai-moe-checkpoint-route-validation-v0"
    assert report["route_pair"]["exact_match"] is True
    assert report["route_summary"]["has_failure"] is False
    assert report["gradient_parity"]["allclose"] is True
    assert report["gradient_parity"]["max_abs_diff"] == pytest.approx(0.0)
    assert report["output_max_abs_diff"] == pytest.approx(0.0)
    assert report["latency_us"]["checkpoint_only_median"] > 0
    assert report["latency_us"]["route_validation_median"] > 0


def test_recompute_benchmark_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="top_k"):
        run_benchmark(
            tokens=1,
            dim=2,
            dim_ffn=4,
            num_experts=2,
            top_k=3,
            device=torch.device("cpu"),
            warmups=0,
            repeats=1,
        )


def test_recompute_benchmark_handles_unselected_expert_gradients():
    report = run_benchmark(
        tokens=1,
        dim=4,
        dim_ffn=8,
        num_experts=8,
        top_k=1,
        device=torch.device("cpu"),
        warmups=0,
        repeats=1,
    )

    parity = report["gradient_parity"]
    assert parity["allclose"] is True
    assert parity["absent_tensor_count"] > 0
    assert parity["presence_mismatch_count"] == 0
    assert (
        parity["compared_tensor_count"] + parity["absent_tensor_count"]
        == parity["tensor_count"]
    )
