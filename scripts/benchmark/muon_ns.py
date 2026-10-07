"""Benchmark Muon NS or matrix optimizer steps on individual/checkpoint shapes."""

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
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5):
            function()
    return graph


def _step(optimizer, parameter):
    optimizer.step()
    return parameter


def step_functions(gradient):
    functions = {}
    for name, options in {
        "torch": {},
        "reuse": {"reuse_ns_buffers": True},
        "kernels": {"use_ns_kernels": True},
    }.items():
        model = nn.Module()
        model.register_parameter("matrix", nn.Parameter(gradient.clone()))
        model.matrix.grad = gradient.clone()
        optimizer = MuonAdamW(model, **options)
        functions[name] = partial(_step, optimizer.muon, model.matrix)
    return functions


def measure_shape(shape, count, repetitions, mode, scope, dtype):
    torch.manual_seed(17)
    gradient = torch.randn(shape, device="cuda", dtype=dtype)
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
    if scope == "step":
        functions = step_functions(gradient)
    reference = functions["reuse"]()
    candidate = functions["kernels"]()
    difference = candidate.float() - reference.float()
    max_abs = difference.abs().max().item()
    relative_l2 = (difference.norm() / reference.float().norm()).item()
    if not torch.isfinite(candidate).all() or relative_l2 > 0.01:
        raise RuntimeError("candidate exceeds the numerical error budget")

    if mode == "graph":
        graphs = {key: capture(function) for key, function in functions.items()}
        runners = {key: graph.replay for key, graph in graphs.items()}
        # Keep function closures alive: Graph does not own external input storage.
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
        "shape": list(shape),
        "count": count,
        "ms": times,
        "speedup_vs_torch": times["torch"] / times["kernels"],
        "speedup_vs_reuse": times["reuse"] / times["kernels"],
        "max_abs_vs_reuse": max_abs,
        "relative_l2_vs_reuse": relative_l2,
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
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    if args.plan:
        configure(json.loads(args.plan.read_text()))
    if not torch.cuda.is_available() or not is_available():
        parser.error("Muon NS CUDA extension is required")
    if args.repetitions < 1 or args.rows < 1 or args.cols < 1:
        parser.error("dimensions and repetitions must be positive")
    counts = (
        checkpoint_shapes(args.model)
        if args.model
        else Counter({(args.rows, args.cols): 1})
    )
    rows = []
    for shape, count in sorted(counts.items()):
        rows.append(
            measure_shape(shape, count, args.repetitions, args.mode, args.scope, dtype)
        )
        torch.cuda.empty_cache()
    totals = {
        key: sum(row["count"] * row["ms"][key] for row in rows)
        for key in ("torch", "reuse", "kernels")
    }
    print(
        json.dumps(
            {
                "mode": args.mode,
                "input_dtype": str(dtype),
                "compute_dtype": "bfloat16",
                "rows": rows,
                "weighted_ms": totals,
                "weighted_speedup_vs_torch": totals["torch"] / totals["kernels"],
                "weighted_speedup_vs_reuse": totals["reuse"] / totals["kernels"],
                "scope": (
                    "Weighted matrix Muon step estimate; includes momentum/parameter updates, excludes AdamW and model execution."
                    if args.scope == "step"
                    else "Weighted NS estimate; excludes momentum, parameter updates, and AdamW."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
