"""Measure RouteTraceV0 codec cost without changing model execution.

This is a serialization microbenchmark, not a capture, replay, training, or
end-to-end performance benchmark. It emits enough geometry and byte accounting
to compare compact ID tiers on a specific host.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

from astrai.moe import (
    PaddingLayout,
    RouteIdentityV0,
    RouterSchemaV0,
    RouteTokenLayoutV0,
    RouteTraceCodecV0,
    RouteTraceLevel,
    RouteTraceV0,
    SelectedWeightSemantics,
    TokenSpanKind,
    canonical_json_digest,
    pack_route_ids,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_trace(
    *,
    tokens: int,
    layers: int,
    top_k: int,
    num_experts: int,
    level: RouteTraceLevel,
    device: torch.device,
) -> RouteTraceV0:
    """Build one deterministic, valid codec benchmark artifact."""
    if min(tokens, layers, top_k, num_experts) < 1 or top_k > num_experts:
        raise ValueError(
            "tokens/layers/top_k/num_experts must be positive and top_k <= num_experts"
        )
    schema = RouterSchemaV0(
        num_moe_layers=layers,
        num_experts=num_experts,
        top_k=top_k,
        expert_parallel_world_size=1,
        score_function="synthetic-softmax-fp32-before-topk",
        score_dtype="float32",
        expert_id_ordering="score-descending-stable-index",
        selected_weight_semantics=SelectedWeightSemantics.SELECTED_RENORMALIZED,
        token_ordering="single-sequence-token-major",
        padding_layout=PaddingLayout.NONE,
        backend="codec-benchmark",
        kernel_semantics_version="synthetic-v0",
        parallel_layout_hash=canonical_json_digest({"placement": "all-experts-local"}),
    )
    identity = RouteIdentityV0(
        policy_version=0,
        model_revision="synthetic-model",
        router_state_version=0,
        checkpoint_revision="synthetic-checkpoint",
        router_schema_hash=schema.fingerprint,
    )
    token_layout = RouteTokenLayoutV0(
        sample_id="codec-benchmark-sample",
        sequence_token_count=tokens,
        prompt_token_count=0,
        token_offset=0,
        token_count=tokens,
        span_kind=TokenSpanKind.FULL_SEQUENCE,
    )
    row = torch.arange(top_k, dtype=torch.int64)
    row_offsets = torch.arange(tokens * layers, dtype=torch.int64).reshape(
        tokens, layers, 1
    )
    topk_ids = pack_route_ids((row + row_offsets) % num_experts, num_experts).to(device)
    selected_weights = None
    topk_margin = None
    if level is RouteTraceLevel.IDS_WEIGHTS_MARGIN:
        selected_weights = torch.full(
            (tokens, layers, top_k),
            1.0 / top_k,
            dtype=torch.float16,
            device=device,
        )
        topk_margin = torch.full(
            (tokens, layers), 0.125, dtype=torch.float16, device=device
        )
    return RouteTraceV0(
        identity=identity,
        router_schema=schema,
        token_layout=token_layout,
        level=level,
        topk_ids=topk_ids,
        selected_weights=selected_weights,
        topk_margin=topk_margin,
    )


def run_benchmark(
    *,
    tokens: int,
    layers: int,
    top_k: int,
    num_experts: int,
    level: RouteTraceLevel,
    device: torch.device,
    warmups: int,
    repeats: int,
) -> dict:
    """Return deterministic byte accounting and measured codec latency."""
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    trace = build_trace(
        tokens=tokens,
        layers=layers,
        top_k=top_k,
        num_experts=num_experts,
        level=level,
        device=device,
    )
    for _ in range(warmups):
        RouteTraceCodecV0.loads(RouteTraceCodecV0.dumps(trace))

    serialize_us = []
    deserialize_us = []
    blob = b""
    restored = None
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter_ns()
        blob = RouteTraceCodecV0.dumps(trace)
        _synchronize(device)
        encoded = time.perf_counter_ns()
        restored = RouteTraceCodecV0.loads(blob)
        finished = time.perf_counter_ns()
        serialize_us.append((encoded - started) / 1000)
        deserialize_us.append((finished - encoded) / 1000)
    assert restored is not None
    if restored.artifact_digest != trace.artifact_digest:
        raise RuntimeError("codec benchmark round-trip changed artifact identity")

    return {
        "schema_version": 1,
        "benchmark": "astrai-route-trace-codec-v0",
        "scope": "codec-only; excludes model capture, transport, replay, and training",
        "device": str(device),
        "torch_version": torch.__version__,
        "parameters": {
            "tokens": tokens,
            "layers": layers,
            "top_k": top_k,
            "num_experts": num_experts,
            "level": level.value,
            "warmups": warmups,
            "repeats": repeats,
        },
        "bytes": {
            "logical_payload": trace.payload_nbytes,
            "wire": len(blob),
            "wire_per_token": len(blob) / tokens,
            "int64_ids_only_baseline": tokens * layers * top_k * 8,
        },
        "latency_us": {
            "serialize_median": statistics.median(serialize_us),
            "serialize_p95": _percentile(serialize_us, 0.95),
            "deserialize_median": statistics.median(deserialize_us),
            "deserialize_p95": _percentile(deserialize_us, 0.95),
        },
        "artifact_digest": trace.artifact_digest,
    }


def main() -> None:
    """Run the codec benchmark and emit one JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=_positive_int, default=1024)
    parser.add_argument("--layers", type=_positive_int, default=16)
    parser.add_argument("--top-k", type=_positive_int, default=2)
    parser.add_argument("--num-experts", type=_positive_int, default=64)
    parser.add_argument(
        "--level", choices=[level.value for level in RouteTraceLevel], default="ids"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=_positive_int, default=20)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.top_k > args.num_experts:
        parser.error("--top-k must not exceed --num-experts")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires an available CUDA device")
    report = run_benchmark(
        tokens=args.tokens,
        layers=args.layers,
        top_k=args.top_k,
        num_experts=args.num_experts,
        level=RouteTraceLevel(args.level),
        device=torch.device(args.device),
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
