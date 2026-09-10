"""Benchmark topology candidates for INT8 DLO AllGather and SP all-to-all."""

from __future__ import annotations

import json
import math
import os
import socket
import statistics
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from astrai.parallel.topology import (
    dlo_groups_for_plan,
    optimize_device_order,
    parse_nvidia_topology,
    topology_score,
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    dp_size: int
    sp_size: int
    device_order: tuple[int, ...]
    topology_label_score: float


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires samples")
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(samples_ms: Sequence[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples_ms),
        "p90_ms": percentile(samples_ms, 0.90),
        "p99_ms": percentile(samples_ms, 0.99),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": list(samples_ms),
    }


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _make_groups(
    dlo_groups: Sequence[tuple[int, ...]],
    sp_groups: Sequence[tuple[int, ...]],
) -> dict[tuple[int, ...], dist.ProcessGroup]:
    unique_groups = tuple(dict.fromkeys((*dlo_groups, *sp_groups)))
    return {
        ranks: dist.new_group(ranks=list(ranks), backend="nccl")
        for ranks in unique_groups
    }


def _rank_group(
    rank: int,
    groups: Sequence[tuple[int, ...]],
    handles: dict[tuple[int, ...], dist.ProcessGroup],
) -> tuple[tuple[int, ...], dist.ProcessGroup]:
    group = next(group for group in groups if rank in group)
    return group, handles[group]


def _measure_collective(
    operation: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    trials: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    dist.barrier()

    samples = []
    for _ in range(trials):
        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000 / iterations
        worst_rank = torch.tensor(elapsed_ms, dtype=torch.float64, device=device)
        dist.all_reduce(worst_rank, op=dist.ReduceOp.MAX)
        if dist.get_rank() == 0:
            samples.append(float(worst_rank.item()))
    return samples


def _benchmark_worker(
    rank: int,
    world_size: int,
    port: int,
    cases: tuple[BenchmarkCase, ...],
    dlo_payload_bytes: int,
    sp_payload_bytes: int,
    warmup: int,
    iterations: int,
    trials: int,
    topology_text: str,
    output: str,
    markdown_output: str | None,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    results: list[dict[str, object]] = []
    try:
        for case in cases:
            kind, dlo_groups, sp_groups = dlo_groups_for_plan(
                case.device_order, case.dp_size, case.sp_size
            )
            handles = _make_groups(dlo_groups, sp_groups)
            dlo_ranks, dlo_handle = _rank_group(rank, dlo_groups, handles)
            sp_ranks, sp_handle = _rank_group(rank, sp_groups, handles)

            local_dlo_bytes = dlo_payload_bytes // len(dlo_ranks)
            if local_dlo_bytes * len(dlo_ranks) != dlo_payload_bytes:
                raise ValueError(
                    "DLO payload bytes must be divisible by DLO group size"
                )
            if sp_payload_bytes % len(sp_ranks):
                raise ValueError("SP payload bytes must be divisible by SP group size")

            dlo_input = torch.full(
                (local_dlo_bytes,), rank, dtype=torch.uint8, device=device
            )
            dlo_output = torch.empty(
                (dlo_payload_bytes,), dtype=torch.uint8, device=device
            )
            sp_input = torch.full(
                (sp_payload_bytes,), rank, dtype=torch.uint8, device=device
            )
            sp_output = torch.empty_like(sp_input)

            dlo_all_gather = partial(
                dist.all_gather_into_tensor,
                dlo_output,
                dlo_input,
                group=dlo_handle,
            )
            sp_all_to_all = partial(
                dist.all_to_all_single,
                sp_output,
                sp_input,
                group=sp_handle,
            )

            dlo_samples = _measure_collective(
                dlo_all_gather,
                warmup=warmup,
                iterations=iterations,
                trials=trials,
                device=device,
            )
            sp_samples = _measure_collective(
                sp_all_to_all,
                warmup=warmup,
                iterations=iterations,
                trials=trials,
                device=device,
            )
            if rank == 0:
                results.append(
                    {
                        "name": case.name,
                        "dp_size": case.dp_size,
                        "sp_size": case.sp_size,
                        "device_order": list(case.device_order),
                        "dlo_group_kind": kind,
                        "dlo_groups": [list(group) for group in dlo_groups],
                        "sp_groups": [list(group) for group in sp_groups],
                        "topology_label_score": case.topology_label_score,
                        "dlo_all_gather": summarize(dlo_samples),
                        "sp_all_to_all": summarize(sp_samples),
                    }
                )
            dist.barrier()

        if rank == 0:
            properties = torch.cuda.get_device_properties(device)
            payload = {
                "metadata": {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "gpu_name": properties.name,
                    "gpu_count": world_size,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "nccl_version": ".".join(
                        str(value) for value in torch.cuda.nccl.version()
                    ),
                    "dtype": "uint8-int8-transfer-representative",
                    "dlo_full_payload_bytes": dlo_payload_bytes,
                    "sp_per_rank_payload_bytes": sp_payload_bytes,
                    "warmup": warmup,
                    "iterations": iterations,
                    "trials": trials,
                    "timing_scope": "worst-rank steady-state collective",
                    "topology": topology_text,
                },
                "results": results,
            }
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if markdown_output is not None:
                markdown_path = Path(markdown_output)
                markdown_path.parent.mkdir(parents=True, exist_ok=True)
                markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def render_markdown(payload: dict[str, object]) -> str:
    metadata = payload["metadata"]
    results = payload["results"]
    assert isinstance(metadata, dict)
    assert isinstance(results, list)
    lines = [
        "# Topology-aware DLO DP/SP benchmark",
        "",
        f"- GPU: {metadata['gpu_count']}x {metadata['gpu_name']}",
        f"- Compute capability: {metadata['compute_capability']}",
        (
            f"- PyTorch / CUDA / NCCL: {metadata['torch_version']} / "
            f"{metadata['cuda_version']} / {metadata['nccl_version']}"
        ),
        f"- DLO full INT8 payload: {metadata['dlo_full_payload_bytes']} bytes",
        f"- SP payload per rank: {metadata['sp_per_rank_payload_bytes']} bytes",
        f"- Timing: {metadata['trials']} trials, worst rank per trial",
        "",
        (
            "| Candidate | DP x SP | Device order | DLO group | DLO median (ms) | "
            "DLO p99 (ms) | SP median (ms) | SP p99 (ms) |"
        ),
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        assert isinstance(result, dict)
        dlo = result["dlo_all_gather"]
        sp = result["sp_all_to_all"]
        assert isinstance(dlo, dict)
        assert isinstance(sp, dict)
        lines.append(
            f"| {result['name']} | {result['dp_size']}x{result['sp_size']} | "
            f"{','.join(str(value) for value in result['device_order'])} | "
            f"{result['dlo_group_kind']} | {dlo['median_ms']:.4f} | "
            f"{dlo['p99_ms']:.4f} | {sp['median_ms']:.4f} | {sp['p99_ms']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_cases(
    topology_text: str, dp_sizes: Sequence[int]
) -> tuple[BenchmarkCase, ...]:
    topology = parse_nvidia_topology(topology_text)
    natural = tuple(sorted(topology.devices))
    cases = []
    for dp_size in dp_sizes:
        if len(natural) % dp_size:
            raise click.BadParameter(
                f"DP size {dp_size} does not divide {len(natural)} GPUs"
            )
        sp_size = len(natural) // dp_size
        optimized, optimized_score = optimize_device_order(topology, dp_size, sp_size)
        candidates = (("natural", natural), ("topology", optimized))
        seen = set()
        for name, order in candidates:
            if order in seen:
                continue
            seen.add(order)
            cases.append(
                BenchmarkCase(
                    name=f"dp{dp_size}-sp{sp_size}-{name}",
                    dp_size=dp_size,
                    sp_size=sp_size,
                    device_order=order,
                    topology_label_score=(
                        optimized_score
                        if order == optimized
                        else topology_score(topology, order, dp_size, sp_size)
                    ),
                )
            )
    return tuple(cases)


def parse_dp_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise click.BadParameter("DP sizes must be comma-separated integers") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise click.BadParameter("DP sizes must be positive")
    return sizes


@click.command(help=__doc__)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--markdown-output", type=click.Path(path_type=Path))
@click.option("--dp-sizes", default="1,2,4", show_default=True)
@click.option(
    "--dlo-payload-mib", type=click.IntRange(min=1), default=256, show_default=True
)
@click.option(
    "--sp-payload-mib", type=click.IntRange(min=1), default=64, show_default=True
)
@click.option("--warmup", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--iterations", type=click.IntRange(min=1), default=20, show_default=True)
@click.option("--trials", type=click.IntRange(min=3), default=7, show_default=True)
def main(
    output: Path,
    markdown_output: Path | None,
    dp_sizes: str,
    dlo_payload_mib: int,
    sp_payload_mib: int,
    warmup: int,
    iterations: int,
    trials: int,
) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    topology_text = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    cases = build_cases(topology_text, parse_dp_sizes(dp_sizes))
    world_size = torch.cuda.device_count()
    if world_size != len(parse_nvidia_topology(topology_text).devices):
        raise click.ClickException("visible CUDA devices and topology matrix disagree")
    mp.spawn(
        _benchmark_worker,
        args=(
            world_size,
            _find_free_port(),
            cases,
            dlo_payload_mib * 1024 * 1024,
            sp_payload_mib * 1024 * 1024,
            warmup,
            iterations,
            trials,
            topology_text,
            str(output),
            str(markdown_output) if markdown_output is not None else None,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
