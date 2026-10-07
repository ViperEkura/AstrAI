"""Sweep symmetric BLAS tiles and export measured dispatch rows.

Like the GEMM tile sweep, candidates bypass automatic planning, are checked
against Torch, and are measured in interleaved order. CUDA Graph replay removes
Python dispatch gaps. Exported rows can be passed to policy.symmetric.configure.
"""

import argparse
import json
import statistics
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch

from astrai.extension.kernel.symmetric import is_available, symm_out, syrk_out, tiles


def capture(function: Callable[[], None]) -> torch.cuda.CUDAGraph:
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            function()
    return graph


def sweep(
    operation: str,
    shape: Tuple[int, int],
    repetitions: int,
    rasters: List[int],
    alpha: float,
    beta: float,
    min_speedup: float,
    input_layout: str,
    output_layout: str,
    batch_size: int = 1,
) -> Dict[str, Any]:
    torch.manual_seed(31)
    prefix = (batch_size,) if batch_size > 1 else ()
    x = torch.randn(
        (*prefix, *(shape if input_layout == "row" else shape[::-1])),
        device="cuda",
        dtype=torch.bfloat16,
    )
    if input_layout == "column":
        x = x.transpose(-2, -1)
    x.div_(x.norm())
    symmetric = torch.randn(
        (*prefix, shape[0], shape[0]), device=x.device, dtype=x.dtype
    )
    symmetric = (symmetric + symmetric.transpose(-2, -1)) / 2
    symmetric.div_(symmetric.norm())
    output_shape = (shape[0], shape[0]) if operation == "syrk" else shape
    output = torch.empty(
        (*prefix, *(output_shape if output_layout == "row" else output_shape[::-1])),
        device=x.device,
        dtype=x.dtype,
    )
    if output_layout == "column":
        output = output.transpose(-2, -1)
    addend = torch.randn_like(output)
    if operation == "syrk":
        addend = (addend + addend.transpose(-2, -1)) / 2
    addend.div_(addend.norm())
    lhs, rhs = (x, x.transpose(-2, -1)) if operation == "syrk" else (symmetric, x)
    addmm = torch.addmm if batch_size == 1 else torch.baddbmm
    reference = addmm(addend, lhs, rhs, alpha=alpha, beta=beta)
    baseline = partial(addmm, addend, lhs, rhs, alpha=alpha, beta=beta, out=output)
    baseline_graph = capture(baseline)
    properties = torch.cuda.get_device_properties(x.device)
    candidates = []
    for tile in tiles(operation):
        for raster in [1] if operation == "syrk" else rasters:
            record = dict(tile=tile["name"], raster=raster, geometry=tile)
            if input_layout not in tile["input_layouts"]:
                record["skip"] = "unsupported input layout"
                candidates.append(record)
                continue
            if tile["shared_memory"] > properties.shared_memory_per_block_optin:
                record["skip"] = "shared memory exceeds device limit"
                candidates.append(record)
                continue
            kernel = syrk_out if operation == "syrk" else partial(symm_out, symmetric)
            function = partial(
                kernel,
                x,
                output,
                addend=addend,
                alpha=alpha,
                beta=beta,
                tile=tile["name"],
                **({"raster": raster} if operation == "symm" else {}),
            )
            function()
            difference = output.float() - reference.float()
            relative = difference.norm().item() / max(
                reference.float().norm().item(), 1e-30
            )
            record["relative_l2"] = relative
            record["max_abs"] = difference.abs().max().item()
            if not torch.isfinite(output).all() or relative > 0.01:
                raise RuntimeError(f"incorrect candidate {record}")
            if operation == "syrk" and not torch.equal(
                output, output.transpose(-2, -1)
            ):
                raise RuntimeError(f"candidate is not symmetric: {tile['name']}")
            graph = capture(function)
            samples = {"torch": [], "cuda": []}
            for _ in range(repetitions):
                for key, target in (
                    ("torch", baseline_graph),
                    ("cuda", graph),
                    ("cuda", graph),
                    ("torch", baseline_graph),
                ):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    target.replay()
                    end.record()
                    end.synchronize()
                    samples[key].append(start.elapsed_time(end) / 10)
            record["ms"] = {
                key: statistics.median(values) for key, values in samples.items()
            }
            record["speedup"] = record["ms"]["torch"] / record["ms"]["cuda"]
            candidates.append(record)
    valid = [row for row in candidates if "skip" not in row]
    best = max(valid, key=lambda row: row["speedup"])
    major, minor = torch.cuda.get_device_capability(x.device)
    measured = dict(
        operation=operation,
        cc=major * 10 + minor,
        rows=shape[0],
        cols=shape[1],
        addend=beta != 0,
        batch_size=batch_size,
        input_layout=input_layout,
        output_layout=output_layout,
        backend="torch",
    )
    if best["speedup"] >= min_speedup:
        measured.update(backend="cuda", tile=best["tile"], raster=best["raster"])
    return dict(
        shape=list(shape),
        batch_size=batch_size,
        operation=operation,
        alpha=alpha,
        beta=beta,
        candidates=candidates,
        plan=measured,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("syrk", "symm"), required=True)
    parser.add_argument("--shapes", default="256:1536,1536:1536,1536:6912")
    parser.add_argument("--rasters", default="0,1,2,-1,-2")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--input-layout", choices=("row", "column"), default="row")
    parser.add_argument("--output-layout", choices=("row", "column"), default="row")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=1.02)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not is_available():
        parser.error("symmetric CUDA extension is required")
    if args.list:
        print(json.dumps(tiles(args.operation), indent=2))
        return
    shapes = [
        tuple(int(value) for value in shape.split(":"))
        for shape in args.shapes.split(",")
    ]
    if (
        not 1 <= args.batch_size <= 65535
        or args.repetitions < 1
        or any(
            len(shape) != 2 or min(shape) < 64 or any(value % 64 for value in shape)
            for shape in shapes
        )
    ):
        parser.error("positive repetitions and dimensions divisible by 64 are required")
    rasters = [int(value) for value in args.rasters.split(",")]
    if not rasters or any(abs(value) > 32 for value in rasters):
        parser.error("rasters must be in [-32, 32]")
    rows = []
    for shape in shapes:
        row = sweep(
            args.operation,
            shape,
            args.repetitions,
            rasters,
            args.alpha,
            args.beta,
            args.min_speedup,
            args.input_layout,
            args.output_layout,
            args.batch_size,
        )
        rows.append(row)
        print(json.dumps(dict(shape=row["shape"], plan=row["plan"])), flush=True)
        torch.cuda.empty_cache()
    report = dict(
        method="interleaved ABBA CUDA Graph, ten calls per replay",
        dtype="bfloat16",
        rows=rows,
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.plan_output:
        args.plan_output.write_text(
            json.dumps([row["plan"] for row in rows], indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
