"""The four GEMM operand layouts, measured through the production dispatch.

The layout is the ``(trans_a, trans_b)`` flag pair over the fixed math
``C[M,N] = A[M,K] @ B[N,K]^T`` — the fused-linear form every mode shares:

  NT  A [M][K] K-contig  B [N][K] K-contig   the fused-linear orientation
  NN  A [M][K] K-contig  B [K][N] N-contig   crosswise 1 (B direct-load)
  TT  A [K][M] M-contig  B [N][K] K-contig   crosswise 1 (A direct-load)
  TN  A [K][M] M-contig  B [K][N] N-contig   crosswise 2 (wgrad orientation)

Operands are pre-transposed outside the timed region: the question is the
storage the kernel sees, not what a caller would pay to build it. Cells run
round-robin (A-B-..-B-A per trial) so clock and thermal drift spread over
every cell instead of favoring whichever runs last.

``--planner-ab`` answers the follow-up: NT dispatches to the builtin
(measured) plan rows while the crosswise shapes have no rows and fall to
the model planner, so is NT's edge the storage or the tuned rows? Pinning
NT to the model planner too (the switch rides inside the timed call, ~us
against a ms kernel) says which.

Run with the venv python from the repo root, e.g.::

    .venv/bin/python csrc/bench/benchmark_layouts.py -m 16384
    .venv/bin/python csrc/bench/benchmark_layouts.py --planner-ab
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrai.extension.ops.gemm import probe, quant_gemm, set_planner  # noqa: E402

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0

# (name, trans_a, trans_b)
LAYOUTS = (
    ("NT", False, True),
    ("NN", False, False),
    ("TT", True, True),
    ("TN", True, False),
)
DTYPES = ("bf16", "fp8")

# The astrai_1b projections (n x k); m comes from -m.
SHAPES = (
    ("square_1536", 1536, 1536),
    ("qkv_6144", 6144, 1536),
    ("mlp_up_6912", 6912, 1536),
    ("mlp_down_6912", 1536, 6912),
)


def quant_fp8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric e4m3 quantization (the F8A8 training pairing)."""
    amax = t.float().abs().amax()
    scale = (amax.clamp_min(1e-12) / FP8_MAX).float()
    return (t.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


def dtype_pair(kind: str) -> tuple[torch.dtype, torch.dtype]:
    return (FP8, FP8) if kind == "fp8" else (torch.bfloat16, torch.bfloat16)


def build_cell(a_bf, b_bf, ta, tb, kind, ref):
    """One (layout, dtype) cell: operand storage + the callable to time."""
    if kind == "bf16":
        a, b, a_scale = a_bf, b_bf, None
    else:
        a8, sa = quant_fp8(a_bf)
        b8, sb = quant_fp8(b_bf)
        a, b, a_scale = a8, b8, (sa * sb).reshape(1)
    a_op = a.t().contiguous() if ta else a
    b_op = b if tb else b.t().contiguous()
    op = lambda: quant_gemm(  # noqa: E731
        a_op, b_op, a_scale=a_scale, trans_a=ta, trans_b=tb
    )
    return op, ref


def time_op(op, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        op()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def report(shape, order, samples, flops):
    for key in order:
        med = statistics.median(samples[key])
        lo, hi = min(samples[key]), max(samples[key])
        print(
            f"{shape:14s} {key[1]:6s} {key[2]:4s} {med:9.4f} "
            f"{flops / (med * 1e-3) / 1e12:9.1f}   [{lo:.4f}..{hi:.4f}]"
        )


@click.command()
@click.option(
    "-m",
    type=int,
    default=16384,
    show_default=True,
    help="activation rows (pretrain micro-batch 8x window 2048).",
)
@click.option("--shapes", default="", help="comma list of SHAPES names to keep.")
@click.option("--warmup", type=int, default=5, show_default=True)
@click.option("--iterations", type=int, default=30, show_default=True)
@click.option(
    "--trials",
    type=int,
    default=3,
    show_default=True,
    help="round-robin passes; each pass times every cell twice.",
)
@click.option(
    "--planner-ab",
    is_flag=True,
    help="NT x {builtin-wins, model} to split layout from row coverage.",
)
def main(m, shapes, warmup, iterations, trials, planner_ab):
    keep = {s.strip() for s in shapes.split(",") if s.strip()}
    print(
        f"device={torch.cuda.get_device_name(0)} m={m} warmup={warmup} "
        f"iters={iterations} trials={trials}"
    )

    if planner_ab:
        n, k = 6144, 1536
        torch.manual_seed(11)
        a_bf = (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
        b_bf = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
        a8, sa = quant_fp8(a_bf)
        b8, sb = quant_fp8(b_bf)
        scale = (sa * sb).reshape(1)
        cells = {}
        for lname, ta, tb in (("NT", False, True), ("TT", True, True)):
            for kind, a, b, sc in (("bf16", a_bf, b_bf, None), ("fp8", a8, b8, scale)):
                a_op = a.t().contiguous() if ta else a
                b_op = b if tb else b.t().contiguous()
                for mode in ("hybrid", "model"):

                    def op(a_op=a_op, b_op=b_op, sc=sc, ta=ta, tb=tb, mode=mode):
                        set_planner(mode)
                        return quant_gemm(
                            a_op, b_op, a_scale=sc, trans_a=ta, trans_b=tb
                        )

                    cells[(lname, kind, mode)] = op
        print(
            "probe:",
            {
                lname: probe(m, n, k, *dtype_pair(kind), trans_a=ta, trans_b=tb)
                for lname, ta, tb in (("NT", False, True), ("TT", True, True))
                for kind in DTYPES
            },
        )
    else:
        cells = {}
        for name, n, k in SHAPES:
            if keep and name not in keep:
                continue
            torch.manual_seed(11)
            a_bf = (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
            b_bf = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
            a8, sa = quant_fp8(a_bf)
            b8, sb = quant_fp8(b_bf)
            refs = {
                "bf16": a_bf.float() @ b_bf.float().t(),
                "fp8": (a8.float() * sa) @ (b8.float() * sb).t(),
            }
            del a8, b8
            for lname, ta, tb in LAYOUTS:
                for kind in DTYPES:
                    op, ref = build_cell(a_bf, b_bf, ta, tb, kind, refs[kind])
                    # Correctness before timing: a layout that changes the
                    # math silently (square shapes hide a flipped operand in
                    # the shape check) must never reach the timing loop.
                    rel = ((op().float() - ref).abs().max() / ref.abs().max()).item()
                    assert rel < 0.02, f"{name} {lname} {kind}: rel={rel}"
                    cells[(name, lname, kind)] = op
            print(
                f"# {name} n={n} k={k}: "
                + ", ".join(
                    f"{l}/{d}={probe(m, n, k, *dtype_pair(d), trans_a=ta, trans_b=tb)['source']}"
                    for l, ta, tb in LAYOUTS
                    for d in DTYPES
                )
            )
        del a_bf, b_bf, refs

    for op in cells.values():
        for _ in range(warmup):
            op()
    torch.cuda.synchronize()

    samples = {key: [] for key in cells}
    order = list(cells)
    for _ in range(trials):
        for key in (*order, *reversed(order)):
            samples[key].append(time_op(cells[key], iterations))

    header = f"{'shape':14s} {'layout':6s} {'dtype':4s} {'ms':>9s} {'TFLOP/s':>9s}"
    print(header + "   [min..max]")
    print("-" * (len(header) + 24))
    if planner_ab:
        n, k = 6144, 1536
        for key in order:
            med = statistics.median(samples[key])
            lo, hi = min(samples[key]), max(samples[key])
            print(
                f"{'qkv_6144':14s} {key[0]:6s} {key[1] + '/' + key[2]:14s} "
                f"{med:9.4f} {2.0 * m * n * k / (med * 1e-3) / 1e12:9.1f}   "
                f"[{lo:.4f}..{hi:.4f}]"
            )
        set_planner("")
    else:
        for name, n, k in SHAPES:
            if keep and name not in keep:
                continue
            keys = [key for key in order if key[0] == name]
            report(name, keys, samples, 2.0 * m * n * k)


if __name__ == "__main__":
    main()
