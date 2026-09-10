"""Measure immutable rollout-to-RouteTraceV0 binding and validation cost.

This microbenchmark starts from already materialized synthetic route tensors.
It does not measure model capture, generation, transport, replay, reward
scoring, optimizer work, or end-to-end training throughput.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from astrai.moe import RouteTraceLevel, RouteTraceV0, TokenSpanKind
from astrai.trainer.route_trace import (
    RolloutRouteTraceBatchV0,
    rollout_route_sample_id,
)
from scripts.benchmark_route_trace_codec import build_trace


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_rollout_inputs(
    *,
    batch_size: int,
    group_size: int,
    prompt_tokens: int,
    response_tokens: int,
    layers: int,
    top_k: int,
    num_experts: int,
    level: RouteTraceLevel,
    device: torch.device,
    rollout_id: str = "rollout-route-binding-benchmark",
) -> tuple[dict[str, torch.Tensor], list[list[RouteTraceV0]]]:
    """Build deterministic, fully valid tokens and a B-by-G trace grid."""
    dimensions = (
        batch_size,
        group_size,
        prompt_tokens,
        response_tokens,
        layers,
        top_k,
        num_experts,
    )
    if min(dimensions) < 1 or top_k > num_experts:
        raise ValueError("dimensions must be positive and top_k <= num_experts")

    prompts = (
        torch.arange(1, prompt_tokens + 1, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .clone()
    )
    responses = (
        torch.arange(1, response_tokens + 1, dtype=torch.long, device=device)
        .reshape(1, 1, -1)
        .expand(batch_size, group_size, -1)
        .clone()
    )
    response_offsets = torch.arange(
        batch_size * group_size, dtype=torch.long, device=device
    ).reshape(batch_size, group_size, 1)
    responses.add_(response_offsets * response_tokens)
    prompt_mask = torch.ones_like(prompts, dtype=torch.bool)
    response_mask = torch.ones_like(responses, dtype=torch.bool)
    logprobs_old = torch.full_like(responses, -0.5, dtype=torch.float32)

    base = build_trace(
        tokens=response_tokens,
        layers=layers,
        top_k=top_k,
        num_experts=num_experts,
        level=level,
        device=device,
    )
    identity = replace(
        base.identity,
        policy_version=17,
        model_revision="synthetic-model-17",
        router_state_version=17,
        checkpoint_revision="synthetic-checkpoint-17",
    )
    traces = []
    for batch_index in range(batch_size):
        row = []
        for group_index in range(group_size):
            row.append(
                RouteTraceV0(
                    identity=identity,
                    router_schema=base.router_schema,
                    token_layout=replace(
                        base.token_layout,
                        sample_id=rollout_route_sample_id(
                            rollout_id, batch_index, group_index
                        ),
                        sequence_token_count=prompt_tokens + response_tokens,
                        prompt_token_count=prompt_tokens,
                        token_offset=prompt_tokens,
                        span_kind=TokenSpanKind.RESPONSE,
                    ),
                    level=base.level,
                    topk_ids=base.topk_ids,
                    selected_weights=base.selected_weights,
                    topk_margin=base.topk_margin,
                )
            )
        traces.append(row)
    return {
        "prompts": prompts,
        "prompt_mask": prompt_mask,
        "responses": responses,
        "response_mask": response_mask,
        "logprobs_old": logprobs_old,
    }, traces


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
    device: torch.device,
    warmups: int,
    bind_repeats: int,
    validate_repeats: int,
) -> dict:
    """Return byte accounting and bind/validation latency statistics."""
    if warmups < 0 or min(bind_repeats, validate_repeats) < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    rollout_id = "rollout-route-binding-benchmark"
    tensors, traces = build_rollout_inputs(
        batch_size=batch_size,
        group_size=group_size,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        layers=layers,
        top_k=top_k,
        num_experts=num_experts,
        level=level,
        device=device,
        rollout_id=rollout_id,
    )

    def bind() -> RolloutRouteTraceBatchV0:
        return RolloutRouteTraceBatchV0.bind(
            rollout_id=rollout_id,
            policy_version=17,
            traces=traces,
            **tensors,
        )

    bound = bind()
    for _ in range(warmups):
        bound.validate_against(policy_version=17, **tensors)

    bind_us = []
    for _ in range(bind_repeats):
        _synchronize(device)
        started = time.perf_counter_ns()
        candidate = bind()
        _synchronize(device)
        bind_us.append((time.perf_counter_ns() - started) / 1000)
        if candidate.artifact_digest != bound.artifact_digest:
            raise RuntimeError("rollout binding is not deterministic")

    validate_us = []
    for _ in range(validate_repeats):
        _synchronize(device)
        started = time.perf_counter_ns()
        bound.validate_against(policy_version=17, **tensors)
        _synchronize(device)
        validate_us.append((time.perf_counter_ns() - started) / 1000)

    manifest_bytes = len(
        json.dumps(
            bound.manifest(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    return {
        "schema_version": 1,
        "benchmark": "astrai-rollout-route-trace-binding-v0",
        "scope": (
            "binding/validation only; excludes capture, generation, transport, "
            "replay, scoring, and training"
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "parameters": {
            "batch_size": batch_size,
            "group_size": group_size,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "layers": layers,
            "top_k": top_k,
            "num_experts": num_experts,
            "level": level.value,
            "warmups": warmups,
            "bind_repeats": bind_repeats,
            "validate_repeats": validate_repeats,
        },
        "bytes": {
            "trace_payloads": bound.payload_nbytes,
            "identity_manifest": manifest_bytes,
            "trace_payload_per_response": bound.payload_nbytes
            / (batch_size * group_size),
        },
        "latency_us": {
            "bind_median": statistics.median(bind_us),
            "bind_p95": _percentile(bind_us, 0.95),
            "validate_median": statistics.median(validate_us),
            "validate_p95": _percentile(validate_us, 0.95),
        },
        "artifact_digest": bound.artifact_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--group-size", type=_positive_int, default=2)
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--bind-repeats", type=_positive_int, default=5)
    parser.add_argument("--validate-repeats", type=_positive_int, default=50)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.top_k > args.num_experts:
        parser.error("--top-k must not exceed --num-experts")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires an available CUDA device")

    report = run_benchmark(
        batch_size=args.batch_size,
        group_size=args.group_size,
        prompt_tokens=args.prompt_tokens,
        response_tokens=args.response_tokens,
        layers=args.layers,
        top_k=args.top_k,
        num_experts=args.num_experts,
        level=RouteTraceLevel(args.level),
        device=torch.device(args.device),
        warmups=args.warmups,
        bind_repeats=args.bind_repeats,
        validate_repeats=args.validate_repeats,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
