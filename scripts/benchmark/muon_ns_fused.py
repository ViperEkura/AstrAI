"""Compare the fused Gram polynomial kernel with ATen GEMM in one NS entry."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from astrai.extension.kernel.muon_ns import is_available
from astrai.extension.loader import get_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--profile", action="store_true", help="emit NVTX ranges for nsys"
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not is_available():
        parser.error("muon_ns CUDA extension is unavailable")
    if min(args.rows, args.cols, args.ns_steps, args.repetitions, args.rounds) < 1:
        parser.error("shape, steps, repetitions and rounds must be positive")

    torch.manual_seed(23)
    module = get_module("muon_ns")
    grad = torch.randn((args.rows, args.cols), device="cuda", dtype=torch.float32)
    coefficients = (3.4445, -4.775, 2.0315)

    def run(fused: bool) -> torch.Tensor:
        return module.muon_ns(grad, coefficients, args.ns_steps, 1e-7, fused)

    for _ in range(5):
        run(False)
        run(True)
    torch.cuda.synchronize()

    expected, actual = run(False), run(True)
    parity = {
        "max_abs": (expected.float() - actual.float()).abs().max().item(),
        "different_elements": torch.count_nonzero(expected != actual).item(),
    }
    if args.profile:
        for name, fused in (("muon_unfused", False), ("muon_fused", True)):
            torch.cuda.nvtx.range_push(name)
            run(fused)
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        print(json.dumps({"shape": [args.rows, args.cols], "parity": parity}, indent=2))
        return

    timings = {False: [], True: []}
    for fused in (False, True, True, False) * args.rounds:
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repetitions):
            run(fused)
        stop.record()
        stop.synchronize()
        timings[fused].append(start.elapsed_time(stop) / args.repetitions)

    def summarize(samples: list[float]) -> dict[str, float]:
        ordered = sorted(samples)
        return {
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
            "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        }

    result = {
        "gpu": torch.cuda.get_device_name(),
        "shape": [args.rows, args.cols],
        "ns_steps": args.ns_steps,
        "repetitions": args.repetitions,
        "rounds": args.rounds,
        "parity": parity,
        "aten_gemm": summarize(timings[False]),
        "fused_polynomial": summarize(timings[True]),
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
