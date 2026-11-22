"""Benchmark dense and cached attention for prefill and decode against torch SDPA.

Suites (--suite): decode, prefill, paged_decode, paged_prefill, all. The
torch side times one SDPA call per step over dense tensors: GQA expansion,
page-table gathers, padding, and masks are built once outside the timed
region, and masked calls prefer the cuDNN backend (the default masked path
is the slow math backend). The kernel's timed work still includes its fused
paged reads and current-token K/V append. Agreement is checked against the
same reference.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import click
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from astrai.extension import is_available
from astrai.extension.runtime.loader import get_module
from astrai.inference.worker.workspace import MAX_SPLITS
from astrai.performance import measure_operations, summarize


@dataclass(frozen=True)
class GqaConfig:
    """One model family's attention geometry (llama-style GQA)."""

    name: str
    hq: int
    hkv: int
    head_dim: int


DEFAULT_CONFIGS = (GqaConfig("astrai_2b", 16, 4, 128),)

# (batch, per-request context length); context includes the token being
# decoded (kv_len = context, the last slot written in-kernel).
DECODE_CASES = ((1, 2048), (1, 4096), (8, 4096), (32, 2048), (64, 1024))
# (batch, q_len); prefill from scratch so kv_len == q_len.
PREFILL_CASES = ((1, 2048), (1, 4096), (4, 1024), (8, 512))


def parse_config(value: str) -> GqaConfig:
    parts = value.split(":")
    if len(parts) != 4 or not parts[0]:
        raise click.BadParameter("config must use NAME:HQ:HKV:HEAD_DIM")
    try:
        hq, hkv, head_dim = (int(item) for item in parts[1:])
    except ValueError as exc:
        raise click.BadParameter("HQ/HKV/HEAD_DIM must be integers") from exc
    if hq <= 0 or hkv <= 0 or head_dim <= 0 or hq % hkv or head_dim % 32:
        raise click.BadParameter(
            "HQ/HKV positive with HQ % HKV == 0; HEAD_DIM % 32 == 0"
        )
    return GqaConfig(parts[0], hq, hkv, head_dim)


def repeat_kv_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand [*, n_kv_heads, head_dim] to [*, n_kv_heads * n_rep, head_dim]
    with the backend's grouping (kv head = q head // n_rep)."""
    if n_rep == 1:
        return x
    n_heads, head_dim = x.shape[-2:]
    return (
        x.unsqueeze(-2)
        .expand(*x.shape[:-2], n_heads, n_rep, head_dim)
        .reshape(*x.shape[:-2], n_heads * n_rep, head_dim)
    )


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
    """SDPA over blhd tensors: [batch, seq, heads, head_dim] -> blhd."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **kwargs
    )
    return out.transpose(1, 2)


def prefer_cudnn_sdpa(
    call: Callable[[], torch.Tensor],
) -> Callable[[], torch.Tensor]:
    """Return a closure running the masked SDPA ``call`` on cuDNN attention
    when the backend accepts the bool mask, else torch's default (the
    default masked path falls back to the much slower math backend)."""

    def with_cudnn() -> torch.Tensor:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return call()

    try:
        with_cudnn()
    except Exception:
        return call
    return with_cudnn


def ragged_lens(batch: int, span: int) -> List[int]:
    """Deterministic mixed lengths spanning [span // 2, span]."""
    if batch == 1:
        return [span]
    step = max(span // 2 // (batch - 1), 1)
    return [span - (batch - 1 - i) * step for i in range(batch)]


def cumsum_indptr(lens: List[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )


@dataclass(frozen=True)
class PagedInputs:
    """Standalone replicas of the PagePool / InferenceWorkspace tensors."""

    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_table: torch.Tensor
    slots: torch.Tensor


def build_paged_inputs(
    config: GqaConfig, kv_lens: List[int], q_lens: Optional[List[int]]
) -> PagedInputs:
    batch = len(kv_lens)
    page_size = 16
    columns = (max(kv_lens) + page_size - 1) // page_size
    table = torch.randperm(batch * columns, device="cuda", dtype=torch.int32).view(
        batch, columns
    )
    pool = torch.randn(
        batch * columns,
        page_size,
        config.hkv,
        config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    positions = torch.arange(max(kv_lens), device="cuda")
    slots = table[:, positions // page_size].long() * page_size + positions % page_size
    return PagedInputs(
        q=torch.randn(
            sum(q_lens) if q_lens is not None else batch,
            config.hq,
            config.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        k_cache=pool,
        v_cache=torch.randn_like(pool),
        page_table=table,
        slots=slots,
    )


def report_result(
    suite: str,
    config: GqaConfig,
    case: Dict[str, int],
    operations: Dict[str, Callable[[], torch.Tensor]],
    io_bytes: int,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> Dict[str, object]:
    samples = measure_operations(
        operations, warmup=warmup, iterations=iterations, trials=trials
    )
    reference = operations["torch"]()
    actual = operations["cuda"]()
    difference = actual.float() - reference.float()
    result: Dict[str, object] = {
        "suite": suite,
        "config": config.name,
        "agreement": {
            "max_abs_error": float(difference.abs().max()),
            "cosine_similarity": float(
                F.cosine_similarity(
                    actual.float().flatten(), reference.float().flatten(), dim=0
                )
            ),
        },
        "estimated_io_bytes": io_bytes,
        **case,
    }
    for name in operations:
        latency = summarize(samples[name])
        result[name] = {
            "effective_bandwidth_gbps": io_bytes / (latency["median_ms"] / 1000) / 1e9,
            **latency,
        }
    speedup = (result["torch"]["median_ms"] / result["cuda"]["median_ms"] - 1.0) * 100.0
    label = f"B={case.get('batch')}" + (
        f" ctx={case['context']}" if "context" in case else f" q={case['q_len']}"
    )
    print(
        f"{suite},{config.name},{label},{result['torch']['median_ms']:.4f},"
        f"{result['cuda']['median_ms']:.4f},{speedup:+.1f}%,"
        f"{result['agreement']['max_abs_error']:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------


def benchmark_decode(
    config: GqaConfig,
    batch: int,
    context: int,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> Dict[str, object]:
    q = torch.randn(
        batch, 1, config.hq, config.head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch,
        context,
        config.hkv,
        config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    # GQA expansion is data preparation, not attention compute — build it
    # once so the timed torch side is a single SDPA launch.
    k_expanded = repeat_kv_heads(k, config.hq // config.hkv)
    v_expanded = repeat_kv_heads(v, config.hq // config.hkv)

    def torch_op() -> torch.Tensor:
        return sdpa(q, k_expanded, v_expanded)

    def cuda_op() -> torch.Tensor:
        return get_module("attention").attention(q, k, v, is_causal=True)

    operations = {"torch": torch_op, "cuda": cuda_op}
    io_bytes = (2 * q.numel() + 2 * k.numel()) * q.element_size()
    return report_result(
        "decode",
        config,
        {"batch": batch, "context": context},
        operations,
        io_bytes,
        warmup=warmup,
        iterations=iterations,
        trials=trials,
    )


def benchmark_prefill(
    config: GqaConfig,
    batch: int,
    q_len: int,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> Dict[str, object]:
    q = torch.randn(
        batch, q_len, config.hq, config.head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch, q_len, config.hkv, config.head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    k_expanded = repeat_kv_heads(k, config.hq // config.hkv)
    v_expanded = repeat_kv_heads(v, config.hq // config.hkv)

    def torch_op() -> torch.Tensor:
        return sdpa(q, k_expanded, v_expanded, is_causal=True)

    def cuda_op() -> torch.Tensor:
        return get_module("attention").attention(q, k, v, is_causal=True)

    operations = {"torch": torch_op, "cuda": cuda_op}
    io_bytes = (2 * q.numel() + 2 * k.numel()) * q.element_size()
    return report_result(
        "prefill",
        config,
        {"batch": batch, "q_len": q_len},
        operations,
        io_bytes,
        warmup=warmup,
        iterations=iterations,
        trials=trials,
    )


def benchmark_paged_decode(
    config: GqaConfig,
    batch: int,
    context: int,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> Dict[str, object]:
    n_rep = config.hq // config.hkv
    kv_lens = [length + 1 for length in ragged_lens(batch, context)]
    inputs = build_paged_inputs(config, kv_lens, None)
    new_k = torch.randn(
        batch, config.hkv, config.head_dim, device="cuda", dtype=torch.bfloat16
    )
    new_v = torch.randn_like(new_k)
    o_part = torch.empty(
        batch,
        config.hq,
        MAX_SPLITS,
        config.head_dim,
        dtype=torch.float32,
        device="cuda",
    )
    ml_part = torch.empty(
        batch, config.hq, MAX_SPLITS, 2, dtype=torch.float32, device="cuda"
    )
    out_buf = torch.empty(
        batch, config.hq, config.head_dim, dtype=torch.bfloat16, device="cuda"
    )

    past = torch.tensor(kv_lens, dtype=torch.int32, device="cuda") - 1
    native = get_module("attention").attention_with_kvcache

    def cuda_op() -> torch.Tensor:
        return native(
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            new_k,
            new_v,
            past,
            inputs.page_table,
            is_causal=True,
            o_part=o_part,
            ml_part=ml_part,
            output=out_buf,
        )

    # Reference-side data preparation happens once, outside the timed
    # closure: append the current-token K/V into the pool (the kernel does
    # this fused inside its launch), gather padded K/V, expand GQA heads.
    max_len = max(kv_lens)
    slots = inputs.slots
    last_slots = inputs.slots[
        torch.arange(batch, device="cuda"), torch.tensor(kv_lens) - 1
    ].long()
    inputs.k_cache.flatten(0, 1)[last_slots] = new_k
    inputs.v_cache.flatten(0, 1)[last_slots] = new_v
    k_expanded = repeat_kv_heads(inputs.k_cache.flatten(0, 1)[slots], n_rep)
    v_expanded = repeat_kv_heads(inputs.v_cache.flatten(0, 1)[slots], n_rep)
    position = torch.arange(max_len, device="cuda")
    lengths = torch.tensor(kv_lens, device="cuda", dtype=torch.long)
    keep_mask = (position[None, :] < lengths[:, None])[:, None, None, :]
    q_batched = inputs.q.unsqueeze(1)

    sdpa_call = prefer_cudnn_sdpa(
        lambda: sdpa(q_batched, k_expanded, v_expanded, attn_mask=keep_mask)
    )

    def torch_op() -> torch.Tensor:
        return sdpa_call().squeeze(1)  # [B, Hq, D]

    operations = {"torch": torch_op, "cuda": cuda_op}
    io_bytes = (
        2 * inputs.q.numel()  # q read + out write
        + 2 * sum(kv_lens) * config.hkv * config.head_dim  # k/v reads
        + 2 * new_k.numel()  # new k/v writes
    ) * inputs.q.element_size()
    return report_result(
        "paged_decode",
        config,
        {"batch": batch, "context": context},
        operations,
        io_bytes,
        warmup=warmup,
        iterations=iterations,
        trials=trials,
    )


def benchmark_paged_prefill(
    config: GqaConfig,
    batch: int,
    q_len: int,
    *,
    warmup: int,
    iterations: int,
    trials: int,
) -> Dict[str, object]:
    n_rep = config.hq // config.hkv
    q_lens = ragged_lens(batch, q_len)
    inputs = build_paged_inputs(config, q_lens, q_lens)
    qo_indptr = cumsum_indptr(q_lens)

    past = torch.zeros(batch, dtype=torch.int32, device="cuda")
    packed_slots = torch.cat([row[:n] for row, n in zip(inputs.slots, q_lens)])
    k_new = inputs.k_cache.flatten(0, 1)[packed_slots]
    v_new = inputs.v_cache.flatten(0, 1)[packed_slots]
    native = get_module("attention").attention_with_kvcache

    def cuda_op() -> torch.Tensor:
        return native(
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            k_new,
            v_new,
            past,
            inputs.page_table,
            qo_indptr,
            is_causal=True,
        )

    # Same once-only preparation: gather padded K/V, expand GQA heads, pad Q,
    # build the causal + validity mask. The timed reference is one SDPA call
    # plus the packed-row unpack.
    max_len = max(q_lens)
    slots = inputs.slots
    k_expanded = repeat_kv_heads(inputs.k_cache.flatten(0, 1)[slots], n_rep)
    v_expanded = repeat_kv_heads(inputs.v_cache.flatten(0, 1)[slots], n_rep)
    position = torch.arange(max_len, device="cuda")
    lengths = torch.tensor(q_lens, device="cuda", dtype=torch.long)
    causal = position[None, :, None] >= position[None, None, :]
    keep = position[None, None, :] < lengths[:, None, None]
    attn_mask = (causal & keep).unsqueeze(1)
    q_padded = torch.zeros(
        batch,
        max_len,
        config.hq,
        config.head_dim,
        device="cuda",
        dtype=inputs.q.dtype,
    )
    for i, length in enumerate(q_lens):
        q_padded[i, :length] = inputs.q[int(qo_indptr[i]) : int(qo_indptr[i + 1])]

    sdpa_call = prefer_cudnn_sdpa(
        lambda: sdpa(q_padded, k_expanded, v_expanded, attn_mask=attn_mask)
    )

    def torch_op() -> torch.Tensor:
        out = sdpa_call()
        return torch.cat([out[i, :length] for i, length in enumerate(q_lens)])

    operations = {"torch": torch_op, "cuda": cuda_op}
    io_bytes = (
        2 * inputs.q.numel() + 2 * sum(q_lens) * config.hkv * config.head_dim
    ) * inputs.q.element_size()
    return report_result(
        "paged_prefill",
        config,
        {"batch": batch, "q_len": q_len},
        operations,
        io_bytes,
        warmup=warmup,
        iterations=iterations,
        trials=trials,
    )


@click.command(help=__doc__)
@click.option("--output", type=click.Path(path_type=Path), help="Optional JSON output.")
@click.option(
    "--suite",
    "suites",
    type=click.Choice(("decode", "prefill", "paged_decode", "paged_prefill", "all")),
    multiple=True,
    default=("all",),
    show_default=True,
)
@click.option(
    "--config",
    "config_values",
    multiple=True,
    help="Filter defaults by bare name, or add/override with NAME:HQ:HKV:HEAD_DIM.",
)
@click.option("--warmup", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--iterations", type=click.IntRange(min=1), default=50, show_default=True)
@click.option("--trials", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def benchmark_command(
    output: Optional[Path],
    suites: Tuple[str, ...],
    config_values: Tuple[str, ...],
    warmup: int,
    iterations: int,
    trials: int,
    seed: int,
) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    kernel_for_suite = {
        "decode": "attention",
        "prefill": "attention",
        "paged_decode": "attention",
        "paged_prefill": "attention",
    }
    selected = (
        tuple(kernel_for_suite) if "all" in suites else tuple(dict.fromkeys(suites))
    )
    missing = [
        kernel_for_suite[suite]
        for suite in selected
        if not is_available(kernel_for_suite[suite])
    ]
    if missing:
        raise click.ClickException(f"built kernels required: {', '.join(missing)}")

    # A bare name filters the matching default; a full spec overrides or appends.
    chosen: Dict[str, GqaConfig] = {}
    for value in config_values:
        config = (
            next((c for c in DEFAULT_CONFIGS if c.name == value), None)
            if ":" not in value
            else parse_config(value)
        )
        if config is None:
            raise click.BadParameter(f"unknown default config {value!r}")
        chosen[config.name] = config
    configs = tuple(chosen.values()) or DEFAULT_CONFIGS

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    runners = {
        "decode": (benchmark_decode, DECODE_CASES),
        "prefill": (benchmark_prefill, PREFILL_CASES),
        "paged_decode": (benchmark_paged_decode, DECODE_CASES),
        "paged_prefill": (benchmark_paged_prefill, PREFILL_CASES),
    }
    print("suite,config,case,torch_ms,cuda_ms,speedup,max_abs")
    results = []
    with torch.inference_mode():
        for suite in selected:
            runner, cases = runners[suite]
            for config in configs:
                for case in cases:
                    results.append(
                        runner(
                            config,
                            *case,
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
                "order": "A-B-B-A",
                "suites": list(selected),
                "configs": [asdict(config) for config in configs],
            },
            "results": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    benchmark_command()
