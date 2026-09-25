"""Benchmark the Gated DeltaNet reference operators against each other and against attention.

Suites (--suite): train, decode, all.

``train`` times one Gated DeltaNet layer's chunked forward/backward against the per-token
recurrent reference and against a GQA layer using the installed attention
backend, at several sequence lengths. ``decode`` times a single-token
``decode_step`` against softmax decoding over a context of the same length.

The Gated DeltaNet layer here is pure PyTorch: it exists to be correct and to be the
reference the chunked operator is checked against, so the per-token path is
expected to lose badly on wall clock — the reusable number is the ratio between
the two Gated DeltaNet paths, which is what a fused kernel would have to beat.

Ordering is A-B-B-A within each trial to balance clock and cache drift, and the
reported latency is the median across trials. Agreement between the chunked and
per-token paths is printed alongside so a speed reading is never taken from a
run that silently diverged.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import click
import torch
import torch.nn.functional as F

from astrai.model.components.attention import GDN, GQA
from astrai.model.components.gdn_ops import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)
from astrai.model.components.rope import RotaryEmbedding


@dataclass(frozen=True)
class GdnGeom:
    """Layer geometry: GQA and Gated DeltaNet are compared at equal hidden size."""

    name: str
    hidden: int
    q_heads: int
    kv_heads: int
    head_dim: int


DEFAULT_GEOM = GdnGeom("sm120_h1024", 1024, 8, 2, 128)
TRAIN_SEQS = (512, 2048, 8192)
DECODE_CONTEXTS = (2048, 8192)


def parse_geom(value: str) -> GdnGeom:
    parts = value.split(":")
    if len(parts) != 5 or not parts[0]:
        raise click.BadParameter(
            "geometry must use NAME:HIDDEN:Q_HEADS:KV_HEADS:HEAD_DIM"
        )
    try:
        hidden, q_heads, kv_heads, head_dim = (int(item) for item in parts[1:])
    except ValueError as exc:
        raise click.BadParameter(
            "HIDDEN/Q_HEADS/KV_HEADS/HEAD_DIM must be integers"
        ) from exc
    if min(hidden, q_heads, kv_heads, head_dim) <= 0 or q_heads % kv_heads:
        raise click.BadParameter("positive dimensions with Q_HEADS % KV_HEADS == 0")
    return GdnGeom(parts[0], hidden, q_heads, kv_heads, head_dim)


def build_layers(geom: GdnGeom, device: str, dtype: torch.dtype):
    """One GQA layer and one Gated DeltaNet layer of the same hidden size."""
    gqa = (
        GQA(
            dim=geom.hidden,
            n_heads=geom.q_heads,
            n_kv_heads=geom.kv_heads,
            use_qk_norm=False,
            norm_eps=1e-6,
            use_gated_attention=False,
            layer_id=0,
        )
        .to(device)
        .to(dtype)
    )
    gdn = (
        GDN(
            dim=geom.hidden,
            n_heads=geom.q_heads,
            gdn_num_key_heads=geom.q_heads,
            gdn_num_value_heads=geom.q_heads,
            gdn_key_head_dim=geom.head_dim,
            gdn_value_head_dim=geom.head_dim,
            n_layers=1,
        )
        .to(device)
        .to(dtype)
    )
    return gqa, gdn


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
    # A-B-B-A order balances cache, clock, and temperature drift.
    for _ in range(trials):
        for name in (*order, *reversed(order)):
            samples[name].append(time_operation(operations[name], iterations))
    return samples


def peak_memory_mb(operation: Callable[[], torch.Tensor]) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    torch.cuda.empty_cache()
    return peak / 2**20


def benchmark_train(
    geom: GdnGeom,
    seq_len: int,
    chunk_size: int,
    *,
    warmup: int,
    iterations: int,
    trials: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, object]:
    gqa, gdn = build_layers(geom, device, dtype)
    x = torch.randn(1, seq_len, geom.hidden, device=device, dtype=dtype)
    freqs = RotaryEmbedding(geom.head_dim, seq_len).to(device)(x)

    # The operators are compared on their own inputs so the reading is about the
    # algorithm, not about the layer's projections. Detached from the layer's
    # graph: forward timing needs no gradient, and the backward phase gets its
    # own leaves so each pass builds only the operator's own graph.
    q, k, v = (tensor.detach() for tensor in gdn._project_qkv(x))
    g, beta = (tensor.detach() for tensor in gdn._gates(x))

    def attn_op():
        return gqa(x, freqs, is_causal=True)

    def chunked_op():
        return chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=chunk_size)[0]

    def recurrent_op():
        return recurrent_gated_delta_rule(q, k, v, g, beta)[0]

    # The per-token reference is 1-2 orders of magnitude slower than everything
    # else here. Interleaving it would leave the GPU in a hot state for whichever
    # candidate runs next and inflate its reading (measured: attention 0.44 ms
    # clean vs 1.29 ms when interleaved with the recurrent op), so it is timed on
    # its own and never mixed into the A-B-B-A group.
    forward = {
        "attn(gqa)": attn_op,
        "gated_deltanet_chunked": chunked_op,
    }
    samples = measure_operations(
        forward, warmup=warmup, iterations=iterations, trials=trials
    )
    samples["gated_deltanet_recurrent"] = measure_operations(
        {"gated_deltanet_recurrent": recurrent_op},
        warmup=warmup,
        iterations=iterations,
        trials=trials,
    )["gated_deltanet_recurrent"]
    result: dict[str, object] = {
        "suite": "train",
        "geometry": geom.name,
        "seq_len": seq_len,
        "chunk_size": chunk_size,
        "forward": {name: summarize(values) for name, values in samples.items()},
        "agreement": {
            "chunked_vs_recurrent_max_abs": float(
                (chunked_op() - recurrent_op()).abs().max()
            ),
            "output_scale": float(chunked_op().abs().max()),
            "dtype": str(dtype),
        },
        "peak_mb": {
            "attn(gqa)": peak_memory_mb(attn_op),
            "gated_deltanet_chunked": peak_memory_mb(chunked_op),
            "gated_deltanet_recurrent": peak_memory_mb(recurrent_op),
        },
    }
    result["speedup"] = {
        "chunked_over_recurrent": result["forward"]["gated_deltanet_recurrent"][
            "median_ms"
        ]
        / result["forward"]["gated_deltanet_chunked"]["median_ms"],
        "attn_over_chunked": result["forward"]["gated_deltanet_chunked"]["median_ms"]
        / result["forward"]["attn(gqa)"]["median_ms"],
    }

    leaves = [tensor.clone().requires_grad_(True) for tensor in (q, k, v, g, beta)]

    def backward(name: str, operation: Callable[[], torch.Tensor]):
        def run():
            for tensor in leaves:
                tensor.grad = None
            operation().square().mean().backward()

        return name, run

    q_g, k_g, v_g, g_g, beta_g = leaves

    # gated_deltanet_fused is deliberately absent: the Triton launches are not
    # autograd-aware, so its backward would silently drop the kernel's
    # contribution instead of failing. gated_deltanet_ops.chunk_gated_delta_rule is the only
    # trainable path until the backward kernels are ported.
    grad_samples = measure_operations(
        {
            "attn(gqa)": backward("attn(gqa)", lambda: gqa(x, freqs, is_causal=True))[
                1
            ],
            "gated_deltanet_chunked": backward(
                "gated_deltanet_chunked",
                lambda: chunk_gated_delta_rule(q_g, k_g, v_g, g_g, beta_g)[0],
            )[1],
        },
        warmup=warmup,
        iterations=iterations,
        trials=trials,
    )
    grad_samples.update(
        measure_operations(
            {
                "gated_deltanet_recurrent": backward(
                    "gated_deltanet_recurrent",
                    lambda: recurrent_gated_delta_rule(q_g, k_g, v_g, g_g, beta_g)[0],
                )[1]
            },
            warmup=warmup,
            iterations=iterations,
            trials=trials,
        )
    )
    result["forward_backward"] = {
        name: summarize(values) for name, values in grad_samples.items()
    }
    result["forward_backward"]["gated_deltanet_fused"] = (
        None  # forward-only: no backward pass
    )
    result["forward_backward"]["gated_deltanet_fast"] = None
    print(
        f"train,{geom.name},{seq_len},{result['forward']['attn(gqa)']['median_ms']:.3f},"
        f"{result['forward']['gated_deltanet_chunked']['median_ms']:.3f},"
        f"{result['forward']['gated_deltanet_recurrent']['median_ms']:.2f},"
        f"{result['speedup']['chunked_over_recurrent']:.1f}x,"
        f"{result['speedup']['attn_over_chunked']:.2f}x,"
        f"{result['agreement']['chunked_vs_recurrent_max_abs']:.2e}"
    )
    del gqa, gdn, x, freqs, q, k, v, g, beta
    torch.cuda.empty_cache()
    return result


def benchmark_decode(
    geom: GdnGeom,
    context: int,
    *,
    warmup: int,
    iterations: int,
    trials: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, object]:
    """One decode step each: Gated DeltaNet steps its fixed state, attention attends the context."""
    gqa, gdn = build_layers(geom, device, dtype)
    token = torch.randn(1, 1, geom.hidden, device=device, dtype=dtype)
    freqs = RotaryEmbedding(geom.head_dim, context + 1).to(device)(token)

    with torch.no_grad():
        _, state = gdn.prefill(token)
        # Softmax decode reads a context of the same length from a stand-in
        # cache; the GQA head expansion is data preparation, done once outside
        # the timed region so both sides time a single step.
        k_ctx = torch.randn(
            1, context, geom.kv_heads, geom.head_dim, device=device, dtype=dtype
        )
        v_ctx = torch.randn_like(k_ctx)
        k_exp = k_ctx.repeat_interleave(geom.q_heads // geom.kv_heads, dim=2)
        v_exp = v_ctx.repeat_interleave(geom.q_heads // geom.kv_heads, dim=2)
        q_step = torch.randn(
            1, 1, geom.q_heads, geom.head_dim, device=device, dtype=dtype
        ).transpose(1, 2)

    def gated_deltanet_step():
        with torch.no_grad():
            out, _ = gdn.decode_step(token, state)
        return out

    def attn_step():
        with torch.no_grad():
            return F.scaled_dot_product_attention(
                q_step, k_exp.transpose(1, 2), v_exp.transpose(1, 2)
            ).transpose(1, 2)

    operations = {"attn(sdpa)": attn_step, "gated_deltanet_step": gated_deltanet_step}
    samples = measure_operations(
        operations, warmup=warmup, iterations=iterations, trials=trials
    )
    result: dict[str, object] = {
        "suite": "decode",
        "geometry": geom.name,
        "context": context,
        "step": {name: summarize(values) for name, values in samples.items()},
        "state_bytes": {
            "gated_deltanet_recurrent": state.recurrent.numel()
            * state.recurrent.element_size(),
            "gated_deltanet_conv": state.conv.numel() * state.conv.element_size(),
            "attn_kv_fp16": 2 * k_ctx.numel() * k_ctx.element_size(),
        },
    }
    print(
        f"decode,{geom.name},{context},{result['step']['attn(sdpa)']['median_ms']:.4f},"
        f"{result['step']['gated_deltanet_step']['median_ms']:.4f},"
        f"{result['state_bytes']['gated_deltanet_recurrent']},"
        f"{result['state_bytes']['attn_kv_fp16']}"
    )
    del gqa, gdn, token, k_ctx, v_ctx, k_exp, v_exp
    torch.cuda.empty_cache()
    return result


@click.command(help=__doc__)
@click.option("--output", type=click.Path(path_type=Path), help="Optional JSON output.")
@click.option(
    "--suite",
    "suites",
    type=click.Choice(("train", "decode", "all")),
    multiple=True,
    default=("all",),
    show_default=True,
)
@click.option(
    "--geometry",
    "geometry_values",
    multiple=True,
    help="NAME:HIDDEN:Q_HEADS:KV_HEADS:HEAD_DIM. Defaults to sm120_h1024.",
)
@click.option("--chunk-size", type=click.IntRange(min=1), default=64, show_default=True)
# The per-token reference takes seconds per call at long sequences, so the
# defaults stay at one iteration; each operation is still sampled four times
# (two trials of A-B-B-A).
@click.option("--warmup", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--iterations", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--trials", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--dtype",
    "dtype_name",
    type=click.Choice(("bfloat16", "float32")),
    default="bfloat16",
    show_default=True,
)
def benchmark_command(
    output: Optional[Path],
    suites: tuple[str, ...],
    geometry_values: tuple[str, ...],
    chunk_size: int,
    warmup: int,
    iterations: int,
    trials: int,
    seed: int,
    dtype_name: str,
) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    device = "cuda"
    dtype = getattr(torch, dtype_name)
    geometries = tuple(parse_geom(value) for value in geometry_values) or (
        DEFAULT_GEOM,
    )
    selected = ("train", "decode") if "all" in suites else tuple(dict.fromkeys(suites))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(
        "suite,geometry,seq,attn_ms,chunked_ms,recurrent_ms,"
        "chunked_over_recurrent,attn_over_chunked,chunked_vs_recurrent_max_abs"
    )
    results = []
    for geom in geometries:
        if "train" in selected:
            for seq_len in TRAIN_SEQS:
                results.append(
                    benchmark_train(
                        geom,
                        seq_len,
                        chunk_size,
                        warmup=warmup,
                        iterations=iterations,
                        trials=trials,
                        device=device,
                        dtype=dtype,
                    )
                )
        if "decode" in selected:
            for context in DECODE_CONTEXTS:
                results.append(
                    benchmark_decode(
                        geom,
                        context,
                        warmup=warmup,
                        iterations=iterations,
                        trials=trials,
                        device=device,
                        dtype=dtype,
                    )
                )

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
                "dtype": dtype_name,
                "chunk_size": chunk_size,
                "suites": list(selected),
                "geometries": [asdict(geom) for geom in geometries],
            },
            "results": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    benchmark_command()
