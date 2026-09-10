"""Plan topology-aware DP/SP placement for distributed layerwise offload."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import click

from astrai.parallel.topology import (
    DLOMeasurement,
    parse_nvidia_topology,
    select_dlo_plan,
)


def read_topology(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    result = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def read_measurements(path: Path | None) -> tuple[DLOMeasurement, ...]:
    if path is None:
        return ()
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise click.ClickException("measurement JSON must contain a results list")
    measurements = []
    for item in payload["results"]:
        try:
            measurements.append(
                DLOMeasurement(
                    dp_size=int(item["dp_size"]),
                    sp_size=int(item["sp_size"]),
                    device_order=tuple(int(value) for value in item["device_order"]),
                    dlo_all_gather_ms=float(item["dlo_all_gather"]["median_ms"]),
                    sp_collective_ms=float(item["sp_all_to_all"]["median_ms"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise click.ClickException(f"invalid measurement result: {item!r}") from exc
    return tuple(measurements)


@click.command(help=__doc__)
@click.option(
    "--topology-file",
    type=click.Path(path_type=Path, exists=True, readable=True),
    help="Read captured nvidia-smi topo -m output instead of probing locally.",
)
@click.option(
    "--measurements",
    type=click.Path(path_type=Path, exists=True, readable=True),
    help="Prefer collective timings from benchmark_dlo_topology.py.",
)
@click.option(
    "--concurrent-requests", type=click.IntRange(min=1), default=1, show_default=True
)
@click.option("--dp-size", type=click.IntRange(min=1))
@click.option("--sp-size", type=click.IntRange(min=1))
@click.option("--output", type=click.Path(path_type=Path))
def main(
    topology_file: Path | None,
    measurements: Path | None,
    concurrent_requests: int,
    dp_size: int | None,
    sp_size: int | None,
    output: Path | None,
) -> None:
    if (dp_size is None) != (sp_size is None):
        raise click.UsageError("--dp-size and --sp-size must be provided together")
    topology = parse_nvidia_topology(read_topology(topology_file))
    plan = select_dlo_plan(
        topology,
        concurrent_requests=concurrent_requests,
        dp_size=dp_size,
        sp_size=sp_size,
        measurements=read_measurements(measurements),
    )
    payload = plan.as_dict()
    order = ",".join(str(device) for device in plan.device_order)
    payload["launch"] = {
        "astrai_env": {"ASTRAI_DEVICE_ORDER": order},
        "vllm_omni_env": {"CUDA_VISIBLE_DEVICES": order},
        "vllm_omni_args": [
            "--num-gpus",
            str(len(plan.device_order)),
            "--data-parallel-size",
            str(plan.dp_size),
            "--usp",
            str(plan.sp_size),
            "--ring",
            "1",
            "--enable-distributed-layerwise-offload",
        ],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    click.echo(rendered, nl=False)


if __name__ == "__main__":
    main()
