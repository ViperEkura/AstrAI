"""Benchmark Muon NS or grouped matrix optimizer steps on checkpoint shapes."""

import argparse
import json
import statistics
from collections import Counter
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.optim._muon import _zeropower_via_newtonschulz

from astrai.extension.backend.newton_schulz import newton_schulz
from astrai.extension.kernel.symmetric import is_available
from astrai.extension.policy.symmetric import configure
from astrai.optim.muon_adamw import MuonAdamW

COEFFICIENTS = (3.4445, -4.775, 2.0315)


def checkpoint_shapes(path):
    path = Path(path)
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    if not files:
        raise ValueError("no safetensors weights found")
    counts = Counter()
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as weights:
            for name in weights.keys():
                shape = tuple(weights.get_slice(name).get_shape())
                if len(shape) == 2 and not any(
                    item in name for item in ("norm", "bias", "embed", "lm_head")
                ):
                    counts[shape] += 1
    if not counts:
        raise ValueError("no Muon matrix parameters found")
    return counts


def capture(function):
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5):
            output = function()
    return graph, output


def _step(optimizer, parameters):
    optimizer.step()
    return parameters


def step_functions(gradients):
    functions = {}
    for name, options in {
        "torch": {},
        "reuse": {"reuse_ns_buffers": True},
        "kernels": {
            "use_ns_kernels": True,
            "ns_batch_size": gradients.size(0),
        },
    }.items():
        model = nn.Module()
        parameters = []
        for index, gradient in enumerate(gradients):
            parameter = nn.Parameter(gradient.clone())
            parameter.grad = gradient.clone()
            model.register_parameter("matrix_" + str(index), parameter)
            parameters.append(parameter)
        optimizer = MuonAdamW(model, **options)
        functions[name] = partial(_step, optimizer.muon, tuple(parameters))
    return functions


def check_outputs(reference, candidate):
    pairs = (
        zip(reference, candidate)
        if isinstance(reference, tuple)
        else [(reference, candidate)]
    )
    max_abs = 0.0
    difference_squared = 0.0
    reference_squared = 0.0
    for baseline, result in pairs:
        # Validate in row slices so FP32 error temporaries remain bounded even
        # when several large optimizer models and their Graph pools coexist.
        for row in range(0, baseline.size(-2), 32):
            baseline_float = baseline[..., row : row + 32, :].float()
            candidate_slice = result[..., row : row + 32, :]
            difference = candidate_slice.float() - baseline_float
            if not torch.isfinite(candidate_slice).all():
                raise RuntimeError("candidate contains nonfinite values")
            max_abs = max(max_abs, difference.abs().max().item())
            difference_squared += difference.square().sum().item()
            reference_squared += baseline_float.square().sum().item()
    relative_l2 = (
        difference_squared / max(reference_squared, torch.finfo(torch.float32).tiny)
    ) ** 0.5
    if relative_l2 > 0.01:
        raise RuntimeError("candidate exceeds the numerical error budget")
    return max_abs, relative_l2


def measure_group(shape, batch_size, repetitions, mode, scope, dtype):
    torch.manual_seed(17)
    gradients = torch.randn((batch_size, *shape), device="cuda", dtype=dtype)
    if scope == "step":
        functions = step_functions(gradients)
    else:
        gradient = gradients[0]
        functions = {
            "torch": partial(
                _zeropower_via_newtonschulz, gradient.clone(), COEFFICIENTS, 5, 1e-7
            ),
            "reuse": partial(newton_schulz, gradient.clone(), COEFFICIENTS, 5, 1e-7),
            "kernels": partial(
                newton_schulz,
                gradient.clone(),
                COEFFICIENTS,
                5,
                1e-7,
                backend="auto",
            ),
        }
    # Advance every implementation equally before capturing optimizer state.
    outputs = {key: function() for key, function in functions.items()}
    max_abs, relative_l2 = check_outputs(outputs["reuse"], outputs["kernels"])
    graph_errors = {}
    torch.cuda.empty_cache()
    if mode == "graph":
        captures = {key: capture(function) for key, function in functions.items()}
        runners = {key: value[0].replay for key, value in captures.items()}
        # Keep closures and captured outputs alive: Graph does not own external inputs.
        for runner in runners.values():
            runner()
        graph_max_abs, graph_relative_l2 = check_outputs(
            captures["reuse"][1], captures["kernels"][1]
        )
        graph_errors = {
            "graph_max_abs_vs_reuse": graph_max_abs,
            "graph_relative_l2_vs_reuse": graph_relative_l2,
        }
        calls = 5
    else:
        for function in functions.values():
            for _ in range(5):
                function()
        runners = functions
        calls = 1
    samples = {key: [] for key in functions}
    for _ in range(repetitions):
        for key in ("torch", "reuse", "kernels", "kernels", "reuse", "torch"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runners[key]()
            end.record()
            end.synchronize()
            samples[key].append(start.elapsed_time(end) / calls)
    times = {key: statistics.median(values) for key, values in samples.items()}
    return {
        "batch_size": batch_size,
        "ms": times,
        "speedup_vs_torch": times["torch"] / times["kernels"],
        "speedup_vs_reuse": times["reuse"] / times["kernels"],
        "max_abs_vs_reuse": max_abs,
        "relative_l2_vs_reuse": relative_l2,
        **graph_errors,
    }


def measure_shape(shape, count, repetitions, mode, scope, dtype, batch_size=1):
    if count < 1 or batch_size < 1:
        raise ValueError("matrix count and batch size must be positive")
    if scope == "ns" and batch_size != 1:
        raise ValueError("NS scope requires batch size 1 for the 2D Torch reference")
    full_groups, remainder = divmod(count, batch_size)
    group_sizes = []
    if full_groups:
        group_sizes.append((batch_size, full_groups))
    if remainder:
        group_sizes.append((remainder, 1))
    groups = []
    for size, group_count in group_sizes:
        group = measure_group(shape, size, repetitions, mode, scope, dtype)
        group["group_count"] = group_count
        groups.append(group)
        torch.cuda.empty_cache()
    totals = {
        key: sum(group["group_count"] * group["ms"][key] for group in groups)
        for key in ("torch", "reuse", "kernels")
    }
    return {
        "shape": list(shape),
        "count": count,
        "group_count": sum(group["group_count"] for group in groups),
        "groups": groups,
        # Retain the per-matrix field for consumers of the original benchmark output.
        "ms": {key: value / count for key, value in totals.items()},
        "weighted_ms": totals,
        "speedup_vs_torch": totals["torch"] / totals["kernels"],
        "speedup_vs_reuse": totals["reuse"] / totals["kernels"],
        "max_abs_vs_reuse": max(group["max_abs_vs_reuse"] for group in groups),
        "relative_l2_vs_reuse": max(group["relative_l2_vs_reuse"] for group in groups),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Read matrix shapes from safetensors headers.")
    parser.add_argument("--rows", type=int, default=1536)
    parser.add_argument("--cols", type=int, default=6912)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument(
        "--plan", type=Path, help="Load measured symmetric dispatch rows."
    )
    parser.add_argument("--scope", choices=("ns", "step"), default="ns")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Parameters per measured step group; NS scope requires 1.",
    )
    args = parser.parse_args()
    if min(args.repetitions, args.rows, args.cols, args.batch_size) < 1:
        parser.error("dimensions, repetitions, and batch size must be positive")
    if args.scope == "ns" and args.batch_size != 1:
        parser.error("NS scope requires --batch-size 1 for the 2D Torch reference")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if args.plan:
        configure(json.loads(args.plan.read_text()))
    if not torch.cuda.is_available() or not is_available():
        parser.error("Muon NS CUDA extension is required")
    counts = (
        checkpoint_shapes(args.model)
        if args.model
        else Counter({(args.rows, args.cols): args.batch_size})
    )
    rows = []
    for shape, count in sorted(counts.items()):
        rows.append(
            measure_shape(
                shape,
                count,
                args.repetitions,
                args.mode,
                args.scope,
                dtype,
                args.batch_size,
            )
        )
        torch.cuda.empty_cache()
    totals = {
        key: sum(row["weighted_ms"][key] for row in rows)
        for key in ("torch", "reuse", "kernels")
    }
    print(
        json.dumps(
            {
                "mode": args.mode,
                "input_dtype": str(dtype),
                "compute_dtype": "bfloat16",
                "batch_size": args.batch_size,
                "rows": rows,
                "weighted_ms": totals,
                "weighted_speedup_vs_torch": totals["torch"] / totals["kernels"],
                "weighted_speedup_vs_reuse": totals["reuse"] / totals["kernels"],
                "timing_units": {
                    "groups.ms": "Measured milliseconds per optimizer group or NS call.",
                    "rows.ms": "Weighted milliseconds divided by the matrix count.",
                    "weighted_ms": "Full groups plus a separately measured partial group.",
                },
                "scope": (
                    "Weighted grouped matrix Muon step estimate; each full and partial group is measured with its actual parameter count. Includes momentum/parameter updates; excludes AdamW and model execution."
                    if args.scope == "step"
                    else "Weighted single-matrix NS estimate; excludes momentum, parameter updates, and AdamW."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
