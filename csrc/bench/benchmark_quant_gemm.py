"""Benchmark the quantized-GEMM family against the bf16 ``F.linear`` baseline.

Every dtype pairing the gemm dispatch instantiates, as an activation-kind
x weight-kind grid: W16A16 (bf16 x bf16), W8A16 (bf16 activations against
int8 or fp8 e4m3/e5m2 weights, per-channel scales), W8A8 (int8 x int8,
per-row activations), and the symmetric-fp8 training pair (matching
formats, per-tensor activations). All modes run the NT orientation the
linear path uses (activation ``[M][K]``, weight ``[N][K]``); quantize
passes are excluded from the timed GEMM — they price the kernel, not the
policy. Agreement columns report the max error against the dequantized
reference.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import click
import torch
import torch.nn.functional as F

from astrai.extension import is_available
from astrai.extension.ops.gemm import quant_gemm
from astrai.extension.quantize import quantize_act_int8, quantize_weight_int8

# GEMM shapes as (N, K) weight mats; M comes from --m-values.
GEMM_SHAPES = (
    ("astrai_1b_square", 1536, 1536),
    ("astrai_1b_qkv", 1536 * 4, 1536),
    ("llama2_7b_qkv", 4096, 4096),
    ("llama2_7b_up_gate", 11008, 4096),
    ("llama2_7b_down", 4096, 11008),
    ("llama3_70b_up_gate", 28672, 8192),
)

# The fp8 formats and their max finite magnitudes.
FP8_FORMATS = (
    ("f8e4m3", torch.float8_e4m3fn, 448.0),
    ("f8e5m2", torch.float8_e5m2, 57344.0),
)

# Every dtype pairing the gemm dispatch instantiates (find_gemm_dispatch in
# csrc/kernels/gemm/gemm.cu): each row names the cell, then the activation
# and weight kinds. Asymmetric low-bit mixes — int8 x fp8, mismatched fp8
# formats, quantized acts against bf16 weights — have no kernel and no row.
GEMM_COMBOS = (
    ("w16a16", "bf16", "bf16"),
    ("w8a16", "bf16", "int8"),
    ("w8a16_f8e4m3", "bf16", "f8e4m3"),
    ("w8a16_f8e5m2", "bf16", "f8e5m2"),
    ("w8a8", "int8", "int8"),
    ("f8a8_e4m3", "f8e4m3", "f8e4m3"),
    ("f8a8_e5m2", "f8e5m2", "f8e5m2"),
)
OP_ORDER = ("bf16", *(label for label, _, _ in GEMM_COMBOS))


def parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise click.BadParameter("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise click.BadParameter("values must be positive integers")
    return values


def parse_shape(value: str) -> tuple[str, int, int]:
    parts = value.split(":")
    if len(parts) != 3 or not parts[0]:
        raise click.BadParameter("shape must use NAME:ROWS:COLS")
    try:
        rows, cols = (int(item) for item in parts[1:])
    except ValueError as exc:
        raise click.BadParameter("ROWS:COLS must be integers") from exc
    if rows <= 0 or cols <= 0:
        raise click.BadParameter("ROWS and COLS must be positive")
    return parts[0], rows, cols


def time_operation(operation: Callable[[], torch.Tensor], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def measure_operations(
    operations: dict[str, Callable[[], torch.Tensor]],
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, list[float]]:
    for operation in operations.values():
        for _ in range(warmup):
            operation()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in operations}
    order = tuple(operations)
    # A-B-C-C-B-A order balances cache, clock, and temperature drift.
    for _ in range(trials):
        for name in (*order, *reversed(order)):
            samples[name].append(time_operation(operations[name], iterations))
    return samples


def quantize_fp8(
    t: torch.Tensor, dtype: torch.dtype, max_val: float, per_channel: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric fp8 quantization onto the format's finite range.

    Weights go per-channel (``[N]`` scales, the W-F8A16 pairing);
    activations per-tensor (one scalar, the F8A8 training pairing). The
    kernel applies the inverse scales in its epilogue.
    """
    amax = t.float().abs().amax(dim=-1 if per_channel else None, keepdim=True)
    scale = amax.clamp_min(1e-12) / max_val
    t_q = (t.float() / scale).clamp(-max_val, max_val).to(dtype)
    return t_q, (scale.squeeze(-1) if per_channel else scale).float()


def make_combo_op(
    a: torch.Tensor,
    a_scale: torch.Tensor | None,
    b: torch.Tensor,
    b_scale: torch.Tensor | None,
) -> Callable[[], torch.Tensor]:
    """Bind one combo cell into a zero-arg op (lambdas in loops bind late)."""
    return lambda: quant_gemm(a, b, a_scale=a_scale, b_scale=b_scale)


def dequantize(t: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """Undo a quantize pair for the F.linear reference."""
    if scale is None:
        return t
    if scale.ndim == 1:
        scale = scale.unsqueeze(-1)
    return (t.float() * scale).to(torch.bfloat16)


def benchmark_gemm(
    name: str,
    n: int,
    k: int,
    m: int,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, object]:
    x = (torch.randn(m, k, device="cuda") * 0.05).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
    acts: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {
        "bf16": (x, None),
        "int8": quantize_act_int8(x),
    }
    weights: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {
        "bf16": (w, None),
        "int8": quantize_weight_int8(w),
    }
    for label, dtype, max_val in FP8_FORMATS:
        acts[label] = quantize_fp8(x, dtype, max_val, per_channel=False)
        weights[label] = quantize_fp8(w, dtype, max_val, per_channel=True)

    operations: dict[str, Callable[[], torch.Tensor]] = {"bf16": lambda: F.linear(x, w)}
    ref = {}
    for label, a_kind, b_kind in GEMM_COMBOS:
        operations[label] = make_combo_op(*acts[a_kind], *weights[b_kind])
        ref[label] = F.linear(
            dequantize(*acts[a_kind]), dequantize(*weights[b_kind])
        ).float()

    samples = measure_operations(
        operations, warmup=warmup, iterations=iterations, trials=trials
    )
    med = {key: statistics.median(vals) for key, vals in samples.items()}
    flops = 2.0 * m * n * k
    tfs = {key: flops / (ms * 1e-3) / 1e12 for key, ms in med.items()}
    err = {
        label: (operations[label]().float() - ref[label]).abs().max().item()
        for label, _, _ in GEMM_COMBOS
    }
    row = [name, f"{m}x{n}x{k}"]
    row += [f"{med[op]:.4f}" for op in OP_ORDER]
    row += [f"{tfs[op]:.1f}" for op in OP_ORDER]
    row += [f"{med['bf16'] / med[op]:.2f}x" for op in OP_ORDER[1:]]
    row += [f"{err[op]:.4f}" for op in OP_ORDER[1:]]
    print(",".join(row))
    return {
        "shape": name,
        "m": m,
        "n": n,
        "k": k,
        "median_ms": med,
        "tflops": tfs,
        "speedup_vs_bf16": {
            key: med["bf16"] / ms for key, ms in med.items() if key != "bf16"
        },
        "max_err": err,
    }


@click.command()
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional JSON evidence path (kept out of the repository).",
)
@click.option(
    "--m-values",
    default="512,2048,4096",
    show_default=True,
    callback=lambda _c, _p, v: parse_positive_ints(v),
)
@click.option(
    "--shape",
    "shape_values",
    multiple=True,
    help="Filter defaults by name or add NAME:N:K.",
)
@click.option("--warmup", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--iterations", type=click.IntRange(min=1), default=50, show_default=True)
@click.option("--trials", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def benchmark_command(
    output: Path | None,
    m_values: tuple[int, ...],
    shape_values: tuple[str, ...],
    warmup: int,
    iterations: int,
    trials: int,
    seed: int,
) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    if not is_available("gemm"):
        raise click.ClickException("the built gemm kernel is required")

    bare_names = {value for value in shape_values if ":" not in value}
    known = {shape[0] for shape in GEMM_SHAPES}
    unknown = sorted(bare_names - known)
    if unknown:
        raise click.BadParameter(f"unknown default shape names: {', '.join(unknown)}")
    specs = [parse_shape(value) for value in shape_values if ":" in value]
    if shape_values:
        by_name = {s[0]: s for s in GEMM_SHAPES if s[0] in bare_names}
        by_name.update({s[0]: s for s in specs})
        shapes = list(by_name.values())
    else:
        shapes = list(GEMM_SHAPES)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    header = (
        ["shape", "mxn_xk"]
        + [f"{op}_ms" for op in OP_ORDER]
        + [f"{op}_tflops" for op in OP_ORDER]
        + [f"{op}_vs_bf16" for op in OP_ORDER[1:]]
        + [f"err_{op}" for op in OP_ORDER[1:]]
    )
    print(",".join(header))
    results = []
    with torch.inference_mode():
        for name, n, k in shapes:
            for m in m_values:
                results.append(
                    benchmark_gemm(
                        name,
                        n,
                        k,
                        m,
                        warmup=warmup,
                        iterations=iterations,
                        trials=trials,
                    )
                )
            torch.cuda.empty_cache()

    if output is not None:
        props = torch.cuda.get_device_properties(0)
        payload = {
            "metadata": {
                "gpu_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "settings": {
                "warmup": warmup,
                "iterations": iterations,
                "trials": trials,
                "seed": seed,
                "order": "A-B-C-C-B-A",
                "m_values": list(m_values),
            },
            "results": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")


if __name__ == "__main__":
    benchmark_command()
