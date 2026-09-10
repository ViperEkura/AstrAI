"""Benchmark the fused rotary-embedding kernel against the torch fallback.

The baseline is the complex-multiply fallback from
``astrai.extension.backend.rotary``. Layouts mirror the production call
shapes: packed 3D [tokens, n_heads, head_dim] and dense 4D
[batch, seq_len, n_heads, head_dim]; positions are random integers so every
row exercises a distinct cos/sin gather.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import click
import torch
import torch.nn.functional as F

from astrai.extension import is_available
from astrai.extension.ops import rotary_emb


@dataclass(frozen=True)
class RotaryCase:
    name: str
    layout: str
    batch: int
    seq_len: int
    heads: int
    head_dim: int


DEFAULT_CASES = (
    RotaryCase("decode_bs1", "packed", 1, 1, 32, 128),
    RotaryCase("decode_bs32", "packed", 32, 1, 32, 128),
    RotaryCase("prefill_4k_llama7b", "packed", 1, 4096, 32, 128),
    RotaryCase("prefill_4k_llama70b", "packed", 1, 4096, 64, 128),
    RotaryCase("train_8x2k_llama7b", "dense", 8, 2048, 32, 128),
    RotaryCase("train_4x1k_d64", "dense", 4, 1024, 32, 64),
)


def parse_case(value: str) -> RotaryCase:
    parts = value.split(":")
    if len(parts) != 6 or not parts[0]:
        raise click.BadParameter("case must use NAME:LAYOUT:BATCH:SEQ:HEADS:HEAD_DIM")
    name, layout, batch, seq_len, heads, head_dim = parts
    try:
        fields = (int(batch), int(seq_len), int(heads), int(head_dim))
    except ValueError as exc:
        raise click.BadParameter("fields must be integers") from exc
    if layout not in ("packed", "dense"):
        raise click.BadParameter("layout must be 'packed' or 'dense'")
    if any(field <= 0 for field in fields) or head_dim % 2:
        raise click.BadParameter("fields must be positive; HEAD_DIM even")
    return RotaryCase(name, layout, fields[0], fields[1], fields[2], fields[3])


def time_operation(operation: Callable[[], torch.Tensor], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
    }


def torch_apply(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """The complex-multiply fallback (mirrors backend.rotary._torch_apply)."""
    cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]
    dtype = x.dtype
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_)
    freqs_cis_complex = torch.complex(cos, sin).unsqueeze(-2)
    x_rotated = x_complex * freqs_cis_complex
    return torch.view_as_real(x_rotated).flatten(-2).to(dtype)


def build_freqs(head_dim: int, positions: torch.Tensor) -> torch.Tensor:
    """[cos, sin] pairs for the given positions, laid out [..., head_dim/2, 2]."""
    theta = 10000.0 ** (
        -torch.arange(0, head_dim, 2, dtype=torch.float64, device=positions.device)
        / head_dim
    )
    freqs = positions.double().unsqueeze(-1) * theta
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1).float()


def benchmark_case(
    case: RotaryCase, *, warmup: int, iterations: int, trials: int
) -> dict[str, object]:
    if case.layout == "packed":
        tokens = case.batch * case.seq_len
        x = torch.randn(
            tokens, case.heads, case.head_dim, device="cuda", dtype=torch.bfloat16
        )
        positions = torch.randint(0, 65536, (tokens,), device="cuda")
    else:
        x = torch.randn(
            case.batch,
            case.seq_len,
            case.heads,
            case.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        positions = torch.randint(0, 65536, (case.batch, case.seq_len), device="cuda")
    freqs_cis = build_freqs(case.head_dim, positions)
    operations: dict[str, Callable[[], torch.Tensor]] = {
        "torch": lambda: torch_apply(x, freqs_cis),
        "cuda": lambda: rotary_emb(x, freqs_cis),
    }
    for operation in operations.values():
        for _ in range(warmup):
            operation()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in operations}
    order = tuple(operations)
    # A-B-B-A order balances cache, clock, and temperature drift.
    for _ in range(trials):
        for name in (*order, *reversed(order)):
            samples[name].append(time_operation(operations[name], iterations))

    with torch.no_grad():
        expected = operations["torch"]().float()
        actual = operations["cuda"]().float()
    difference = actual - expected
    io_bytes = (
        2 * x.numel() * x.element_size() + freqs_cis.numel() * freqs_cis.element_size()
    )

    result: dict[str, object] = {
        "case": case.name,
        "layout": case.layout,
        "heads": case.heads,
        "head_dim": case.head_dim,
        "estimated_io_bytes": io_bytes,
        "max_abs_error": float(difference.abs().max()),
        "cosine_similarity": float(
            F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        ),
    }
    for name, samples_ms in samples.items():
        latency = summarize(samples_ms)
        result[name] = {
            "effective_bandwidth_gbps": io_bytes / (latency["median_ms"] / 1000) / 1e9,
            **latency,
        }
    speedup = (result["torch"]["median_ms"] / result["cuda"]["median_ms"] - 1.0) * 100.0
    print(
        f"{case.name},{case.layout},{result['torch']['median_ms']:.5f},"
        f"{result['cuda']['median_ms']:.5f},{speedup:+.1f}%,"
        f"{result['max_abs_error']:.5f}"
    )
    return result


@click.command(help=__doc__)
@click.option("--output", type=click.Path(path_type=Path), help="Optional JSON output.")
@click.option(
    "--case",
    "case_values",
    multiple=True,
    help="Filter defaults by bare name, or add/override with "
    "NAME:LAYOUT:BATCH:SEQ:HEADS:HEAD_DIM.",
)
@click.option("--warmup", type=click.IntRange(min=1), default=10, show_default=True)
@click.option(
    "--iterations", type=click.IntRange(min=1), default=100, show_default=True
)
@click.option("--trials", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def benchmark_command(
    output: Path | None,
    case_values: tuple[str, ...],
    warmup: int,
    iterations: int,
    trials: int,
    seed: int,
) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    if not is_available("rotary_emb"):
        raise click.ClickException("the built rotary_emb kernel is required")

    # A bare name filters the matching default; a full spec overrides or appends.
    chosen: dict[str, RotaryCase] = {}
    for value in case_values:
        case = (
            next((c for c in DEFAULT_CASES if c.name == value), None)
            if ":" not in value
            else parse_case(value)
        )
        if case is None:
            raise click.BadParameter(f"unknown default case {value!r}")
        chosen[case.name] = case
    cases = tuple(chosen.values()) or DEFAULT_CASES

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("case,layout,torch_ms,cuda_ms,speedup,max_abs")
    results = []
    with torch.inference_mode():
        for case in cases:
            results.append(
                benchmark_case(
                    case, warmup=warmup, iterations=iterations, trials=trials
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
                "order": "A-B-B-A",
            },
            "results": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    benchmark_command()
