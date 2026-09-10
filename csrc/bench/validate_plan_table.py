"""Interleaved holdout validation of candidate AOT plan tables.

Every shape is measured under every table back-to-back (shape outer loop,
round-robin across tables so the comparison shares one GPU clock/thermal
state — the same interleaving rule the sweeps use), best-of-rounds per
table. A table is an ASTR_GEMM_TABLE row file; the special names "none"
and "model" run without a row file (ASTR_GEMM_TABLE=-), i.e. the degraded
band rows — the cost model is deleted from the C++, so both aliases mean
the same "no table" state. Reports per-table totals and per-shape deltas
vs the first (baseline) table.

Usage:
    python csrc/bench/validate_plan_table.py \\
        --table old:/autodl-fs/data/plan_table_grid.txt \\
        --table dense:/autodl-fs/data/plan_table_dense_classic.txt \\
        --shapes "qkv:512:4096:4096,up_gate:2048:14336:4096" \\
        --combos w16a16,w8a16 --batch 1 --iterations 30 --trials 3
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click
import torch
from gen_plan_table import COMBOS, make_scale, random_operand

from astrai.extension import is_available
from astrai.extension.ops.gemm import quant_gemm


def parse_holdout_shape(value: str) -> tuple[str, int, int, int]:
    name, m, n, k = value.split(":")
    return name, int(m), int(n), int(k)


def apply_table(name: str, path: str | None) -> None:
    # "-" = explicit AOT off: neither override nor builtin rows, so the
    # planner falls to the degraded bands. "model" and "none" are both
    # that state (the cost model is deleted from the C++); the aliases
    # keep old invocations working.
    if name in ("model", "none"):
        os.environ["ASTR_GEMM_TABLE"] = "-"
    else:
        os.environ["ASTR_GEMM_TABLE"] = path or ""


def validate(
    tables: dict[str, str | None],
    shapes: list[tuple[str, int, int, int]],
    combos: tuple[str, ...],
    batch: int,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[tuple[str, str], dict[str, float]]:
    """(combo, shape name) -> {table: best ms}, tables round-robin interleaved."""
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    if not is_available("gemm"):
        raise click.ClickException(
            "the built gemm kernel is required (rebuild "
            "the extension with CSRC_KERNELS=true first)"
        )

    device = torch.device(torch.cuda.current_device())
    torch.manual_seed(0)
    names = list(tables)
    records: dict[tuple[str, str], dict[str, float]] = {}
    for shape_idx, (name, m, n, k) in enumerate(shapes):
        # Alternate the table order per shape so a slow GPU clock drift
        # does not favor whichever table runs first throughout.
        order = names[:: -1 if shape_idx % 2 else 1]
        for combo in combos:
            act_dtype, weight_dtype = COMBOS[combo]
            a_scale = make_scale(act_dtype, device)
            b_scale = make_scale(weight_dtype, device)
            weight = random_operand((batch, n, k), weight_dtype, device)
            acts = random_operand((batch, m, k), act_dtype, device)

            def run(acts=acts, weight=weight, a_scale=a_scale, b_scale=b_scale):
                return quant_gemm(acts, weight, a_scale, b_scale)

            best = {table: float("inf") for table in names}
            for _ in range(trials):
                for table in order:
                    apply_table(table, tables[table])
                    for _ in range(warmup):
                        run()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(iterations):
                        run()
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - start) / iterations
                    best[table] = min(best[table], ms)
            records[(combo, name)] = best
            for table, ms in best.items():
                print(
                    f"{table:10s} {combo:12s} {name:14s} m{m:5d} n{n:6d} "
                    f"k{k:5d} {ms * 1e3:8.3f} ms",
                    flush=True,
                )
    return records


@click.command()
@click.option(
    "--table",
    "tables",
    multiple=True,
    required=True,
    help="NAME:PATH (or NAME:none / NAME:model) — candidate table to score.",
)
@click.option(
    "--shapes",
    "shape_values",
    multiple=True,
    required=True,
    help="NAME:M:N:K — holdout shape (NT layout).",
)
@click.option(
    "--combos",
    default=",".join(COMBOS),
    show_default=True,
    callback=lambda _c, _p, v: tuple(
        part.strip() for part in v.split(",") if part.strip()
    ),
)
@click.option("--batch", default=1, show_default=True, help="Batch dim b.")
@click.option("--warmup", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--iterations", type=click.IntRange(min=1), default=30, show_default=True)
@click.option("--trials", type=click.IntRange(min=1), default=3, show_default=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional JSON dump of the per-task best-ms records.",
)
def validate_command(
    tables: tuple[str, ...],
    shape_values: tuple[str, ...],
    combos: tuple[str, ...],
    batch: int,
    warmup: int,
    iterations: int,
    trials: int,
    output: Path | None,
) -> None:
    """Holdout: score candidate tables on out-of-grid and llm-ish shapes."""
    table_map: dict[str, str | None] = {}
    for spec in tables:
        name, _, path = spec.partition(":")
        table_map[name] = path if path else None
    shapes = [parse_holdout_shape(value) for value in shape_values]

    baseline = list(table_map)[0]
    records = validate(table_map, shapes, combos, batch, warmup, iterations, trials)

    totals = {table: 0.0 for table in table_map}
    for (_combo, _name), best in records.items():
        for table, ms in best.items():
            totals[table] += ms

    click.echo("\n=== totals (sum of best ms over all shapes) ===")
    for table, total in totals.items():
        delta = (
            0.0
            if table == baseline
            else (total - totals[baseline]) / totals[baseline] * 100.0
        )
        click.echo(f"{table:14s} {total * 1e3:10.1f} ms  {delta:+7.2f}%")
    click.echo("\n=== per-shape deltas vs baseline (%) ===")
    worst: list[tuple[str, str, float]] = []
    for (combo, name), best in records.items():
        base = best[baseline]
        parts = []
        for table in table_map:
            if table == baseline:
                continue
            delta = (best[table] - base) / base * 100.0
            parts.append(f"{table} {delta:+.1f}")
            if delta > 2.0:
                worst.append((table, f"{combo}/{name}", delta))
        click.echo(f"{combo:12s} {name:14s} " + " | ".join(parts))
    if worst:
        click.echo("\n>=2% regressions:")
        for table, where, delta in sorted(worst, key=lambda w: -w[2]):
            click.echo(f"  {table:10s} {where:26s} +{delta:.1f}%")
    else:
        click.echo("\nno >=2% regressions")
    if output is not None:
        output.write_text(json.dumps({f"{c}/{n}": b for (c, n), b in records.items()}))
        click.echo(f"saved records to {output}")


if __name__ == "__main__":
    validate_command()
