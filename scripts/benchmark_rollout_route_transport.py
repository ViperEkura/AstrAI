"""Measure bounded rollout RouteTrace shard framing and reconstruction.

This starts from an already bound synthetic rollout route batch. It measures
CPU byte framing, independent verification, and complete reconstruction. It
does not measure a particular network, model capture, generation, replay,
optimizer work, or end-to-end training throughput.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from astrai.moe import RouteTraceLevel
from astrai.trainer.route_trace import RolloutRouteTraceBatchV0
from astrai.trainer.route_trace_transport import RolloutRouteTraceTransportV0
from scripts.benchmark_rollout_route_binding import build_rollout_inputs


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
    }


def run_benchmark(
    *,
    batch_size: int,
    group_size: int,
    prompt_tokens: int,
    response_tokens: int,
    layers: int,
    top_k: int,
    num_experts: int,
    level: RouteTraceLevel,
    max_shard_bytes: int,
    max_items_per_shard: int,
    warmups: int,
    repeats: int,
) -> dict:
    """Return deterministic byte accounting and CPU codec latency."""
    dimensions = (
        batch_size,
        group_size,
        prompt_tokens,
        response_tokens,
        layers,
        top_k,
        num_experts,
        max_shard_bytes,
        max_items_per_shard,
        repeats,
    )
    if min(dimensions) < 1 or top_k > num_experts:
        raise ValueError("dimensions must be positive and top_k <= num_experts")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if not isinstance(level, RouteTraceLevel):
        raise ValueError("level must be RouteTraceLevel")

    rollout_id = "rollout-route-transport-benchmark"
    tensors, traces = build_rollout_inputs(
        batch_size=batch_size,
        group_size=group_size,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        layers=layers,
        top_k=top_k,
        num_experts=num_experts,
        level=level,
        device=torch.device("cpu"),
        rollout_id=rollout_id,
    )
    batch = RolloutRouteTraceBatchV0.bind(
        rollout_id=rollout_id,
        policy_version=17,
        traces=traces,
        **tensors,
    )

    def build() -> RolloutRouteTraceTransportV0:
        return RolloutRouteTraceTransportV0.from_batch(
            batch,
            max_shard_bytes=max_shard_bytes,
            max_items_per_shard=max_items_per_shard,
        )

    transport = build()
    for _ in range(warmups):
        for shard_index, payload in enumerate(transport.shard_payloads):
            transport.manifest.verify_shard(shard_index, payload)
        transport.assemble()

    build_us = []
    verify_all_us = []
    assemble_us = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        candidate = build()
        build_us.append((time.perf_counter_ns() - started) / 1000)
        if candidate.manifest != transport.manifest:
            raise RuntimeError("route shard manifest is not deterministic")
        if candidate.shard_payloads != transport.shard_payloads:
            raise RuntimeError("route shard frames are not deterministic")

        started = time.perf_counter_ns()
        for shard_index, payload in enumerate(transport.shard_payloads):
            transport.manifest.verify_shard(shard_index, payload)
        verify_all_us.append((time.perf_counter_ns() - started) / 1000)

        started = time.perf_counter_ns()
        restored = transport.assemble()
        assemble_us.append((time.perf_counter_ns() - started) / 1000)
        if restored.artifact_digest != batch.artifact_digest:
            raise RuntimeError("route shard reconstruction changed batch identity")

    manifest_nbytes = len(transport.manifest.dumps())
    encoded_shard_nbytes = transport.manifest.encoded_nbytes
    metadata_nbytes = manifest_nbytes + encoded_shard_nbytes - batch.payload_nbytes
    assignment = {}
    for world_size in (2, 4):
        per_rank = [
            sum(
                len(payload)
                for _, payload in transport.payloads_for_rank(rank, world_size)
            )
            for rank in range(world_size)
        ]
        assignment[str(world_size)] = {
            "per_rank_encoded_nbytes": per_rank,
            "peak_rank_encoded_nbytes": max(per_rank),
        }

    return {
        "artifact_digest": batch.artifact_digest,
        "benchmark": "astrai-rollout-route-trace-sharded-transport-v0",
        "bytes": {
            "encoded_shards": encoded_shard_nbytes,
            "largest_shard": max(
                descriptor.encoded_nbytes for descriptor in transport.manifest.shards
            ),
            "manifest": manifest_nbytes,
            "metadata": metadata_nbytes,
            "metadata_overhead_fraction": metadata_nbytes / batch.payload_nbytes,
            "trace_payloads": batch.payload_nbytes,
            "transport_total": transport.transport_nbytes,
        },
        "latency_us": {
            "assemble": _latency_summary(assemble_us),
            "build": _latency_summary(build_us),
            "verify_all": _latency_summary(verify_all_us),
        },
        "parameters": {
            "batch_size": batch_size,
            "group_size": group_size,
            "layers": layers,
            "level": level.value,
            "max_items_per_shard": max_items_per_shard,
            "max_shard_bytes": max_shard_bytes,
            "num_experts": num_experts,
            "prompt_tokens": prompt_tokens,
            "repeats": repeats,
            "response_tokens": response_tokens,
            "top_k": top_k,
            "warmups": warmups,
        },
        "rank_assignment": assignment,
        "schema_version": 1,
        "scope": (
            "CPU framing/verification/reconstruction only; excludes model capture, "
            "network transport, generation, replay, scoring, and training"
        ),
        "shard_count": len(transport.shard_payloads),
        "torch_version": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--group-size", type=_positive_int, default=8)
    parser.add_argument("--prompt-tokens", type=_positive_int, default=512)
    parser.add_argument("--response-tokens", type=_positive_int, default=1024)
    parser.add_argument("--layers", type=_positive_int, default=40)
    parser.add_argument("--top-k", type=_positive_int, default=22)
    parser.add_argument("--num-experts", type=_positive_int, default=512)
    parser.add_argument(
        "--level",
        choices=[level.value for level in RouteTraceLevel],
        default=RouteTraceLevel.IDS.value,
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=_positive_int,
        default=16 * 1024 * 1024,
    )
    parser.add_argument("--max-items-per-shard", type=_positive_int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.top_k > args.num_experts:
        parser.error("--top-k must not exceed --num-experts")

    report = run_benchmark(
        batch_size=args.batch_size,
        group_size=args.group_size,
        prompt_tokens=args.prompt_tokens,
        response_tokens=args.response_tokens,
        layers=args.layers,
        top_k=args.top_k,
        num_experts=args.num_experts,
        level=RouteTraceLevel(args.level),
        max_shard_bytes=args.max_shard_bytes,
        max_items_per_shard=args.max_items_per_shard,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
