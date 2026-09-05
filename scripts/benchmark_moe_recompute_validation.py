"""Measure opt-in checkpoint route validation and verify gradient parity.

This microbenchmark compares the existing non-reentrant activation-checkpoint
path with the same path plus route observation. It uses one local DeepSeekMoE
module and makes no distributed-training or end-to-end throughput claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path

import torch

from astrai.model.components.mlp import DeepSeekMoE
from astrai.trainer.train_callback import GradientCheckpointingCallback


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


def _make_model(
    *,
    dim: int,
    dim_ffn: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
) -> DeepSeekMoE:
    model = DeepSeekMoE(
        dim=dim,
        dim_ffn=dim_ffn,
        n_routed_experts=num_experts,
        n_shared_experts=1,
        n_activated_experts=top_k,
        n_layers=2,
    )
    model.apply(
        lambda module: (
            module.reset_parameters() if hasattr(module, "reset_parameters") else None
        )
    )
    return model.to(device=device)


def _run_once(model, callback, source, device):
    model.zero_grad(set_to_none=True)
    callback.route_diagnostics.reset()
    value = source.detach().clone().requires_grad_(True)
    _synchronize(device)
    started = time.perf_counter_ns()
    output = model(value)
    loss = output["hidden_states"].float().square().mean()
    loss = loss + 0.01 * output["aux_loss"].float()
    loss.backward()
    _synchronize(device)
    elapsed_us = (time.perf_counter_ns() - started) / 1000
    gradients = {
        "input": None if value.grad is None else value.grad.detach().cpu().clone()
    }
    gradients.update(
        {
            name: (
                None
                if parameter.grad is None
                else parameter.grad.detach().cpu().clone()
            )
            for name, parameter in model.named_parameters()
        }
    )
    return elapsed_us, output["hidden_states"].detach().cpu(), gradients


def _gradient_parity(reference, candidate, *, rtol: float, atol: float):
    if reference.keys() != candidate.keys():
        raise RuntimeError("gradient sets have different parameter names")
    max_abs_diff = 0.0
    allclose = True
    compared_tensor_count = 0
    absent_tensor_count = 0
    presence_mismatch_count = 0
    for name in reference:
        expected = reference[name]
        actual = candidate[name]
        if expected is None or actual is None:
            if expected is None and actual is None:
                absent_tensor_count += 1
            else:
                presence_mismatch_count += 1
                allclose = False
            continue
        compared_tensor_count += 1
        expected = expected.float()
        actual = actual.float()
        if expected.shape != actual.shape:
            raise RuntimeError(f"gradient shape changed for {name}")
        if expected.numel():
            max_abs_diff = max(
                max_abs_diff, float((expected - actual).abs().max().item())
            )
        allclose = allclose and torch.allclose(expected, actual, rtol=rtol, atol=atol)
    return {
        "absent_tensor_count": absent_tensor_count,
        "allclose": allclose,
        "atol": atol,
        "compared_tensor_count": compared_tensor_count,
        "max_abs_diff": max_abs_diff,
        "presence_mismatch_count": presence_mismatch_count,
        "rtol": rtol,
        "tensor_count": len(reference),
    }


def run_benchmark(
    *,
    tokens: int,
    dim: int,
    dim_ffn: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    warmups: int,
    repeats: int,
) -> dict:
    """Return checkpoint latency, route identity, and gradient parity."""
    if min(tokens, dim, dim_ffn, num_experts, top_k, repeats) < 1:
        raise ValueError("dimensions and repeats must be positive")
    if top_k > num_experts:
        raise ValueError("top_k must not exceed num_experts")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")

    torch.manual_seed(20260903)
    reference_model = _make_model(
        dim=dim,
        dim_ffn=dim_ffn,
        num_experts=num_experts,
        top_k=top_k,
        device=device,
    ).train()
    validated_model = copy.deepcopy(reference_model).train()
    source = torch.randn(tokens, dim, dtype=torch.float32, device=device)
    reference_callback = GradientCheckpointingCallback(
        modules=[DeepSeekMoE], route_validation="off"
    )
    validated_callback = GradientCheckpointingCallback(
        modules=[DeepSeekMoE], route_validation="record"
    )
    reference_callback._enable(reference_model)
    validated_callback._enable(validated_model)

    reference_times = []
    validated_times = []
    reference_output = None
    validated_output = None
    reference_gradients = None
    validated_gradients = None
    try:
        for _ in range(warmups):
            _run_once(reference_model, reference_callback, source, device)
            _run_once(validated_model, validated_callback, source, device)
        for repeat in range(repeats):
            modes = (
                (
                    (reference_model, reference_callback, reference_times),
                    (validated_model, validated_callback, validated_times),
                )
                if repeat % 2 == 0
                else (
                    (validated_model, validated_callback, validated_times),
                    (reference_model, reference_callback, reference_times),
                )
            )
            for model, callback, timings in modes:
                elapsed, output, gradients = _run_once(model, callback, source, device)
                timings.append(elapsed)
                if callback is reference_callback:
                    reference_output = output
                    reference_gradients = gradients
                else:
                    validated_output = output
                    validated_gradients = gradients
    finally:
        reference_callback._disable(reference_model)
        validated_callback._disable(validated_model)

    if (
        reference_output is None
        or validated_output is None
        or reference_gradients is None
        or validated_gradients is None
    ):
        raise RuntimeError("benchmark did not produce comparison outputs")
    route_summary = validated_callback.route_diagnostics.snapshot()
    route_report = validated_callback.route_diagnostics.last_report
    if route_report is None or route_summary.has_failure:
        raise RuntimeError("validated checkpoint did not reproduce its forward route")

    reference_median = statistics.median(reference_times)
    validated_median = statistics.median(validated_times)
    output_max_abs_diff = float(
        (reference_output.float() - validated_output.float()).abs().max().item()
    )
    return {
        "benchmark": "astrai-moe-checkpoint-route-validation-v0",
        "device": str(device),
        "gradient_parity": _gradient_parity(
            reference_gradients,
            validated_gradients,
            rtol=1e-5,
            atol=1e-6,
        ),
        "latency_us": {
            "checkpoint_only_median": reference_median,
            "checkpoint_only_p95": _percentile(reference_times, 0.95),
            "route_validation_median": validated_median,
            "route_validation_p95": _percentile(validated_times, 0.95),
            "validation_overhead_percent": 100
            * (validated_median - reference_median)
            / reference_median,
            "validation_ratio": validated_median / reference_median,
        },
        "output_max_abs_diff": output_max_abs_diff,
        "parameters": {
            "dim": dim,
            "dim_ffn": dim_ffn,
            "num_experts": num_experts,
            "repeats": repeats,
            "tokens": tokens,
            "top_k": top_k,
            "warmups": warmups,
        },
        "route_pair": route_report.to_dict(),
        "route_summary": route_summary.to_dict(),
        "schema_version": 1,
        "scope": (
            "one local MoE forward/backward; excludes distributed training, "
            "route replay, optimizer work, and end-to-end throughput"
        ),
        "torch_version": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=_positive_int, default=128)
    parser.add_argument("--dim", type=_positive_int, default=128)
    parser.add_argument("--dim-ffn", type=_positive_int, default=256)
    parser.add_argument("--num-experts", type=_positive_int, default=8)
    parser.add_argument("--top-k", type=_positive_int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=_positive_int, default=10)
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
        dim=args.dim,
        dim_ffn=args.dim_ffn,
        num_experts=args.num_experts,
        top_k=args.top_k,
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
