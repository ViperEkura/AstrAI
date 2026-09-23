"""Map the dispatch LOGIC over a dense (m, n, k) grid — no kernel launches.

The planner chain is host-only by design (plan_probe in gemm.cuh), so
"which recipe serves this cell" is an analytic question: probing a full
stride grid costs seconds where measuring one costs ~10s/cell (and a
batch-ordered measurement of that size is the thermal-bias trap the
workspace AGENTS.md documents). This tool answers logic questions —
decision diversity, builtin-row coverage, where along m the pick flips,
whether k moves the pick at all — and emits a CSV when the per-cell
decisions are wanted (e.g. diffing two planner modes or two tables).

    python csrc/bench/dispatch_grid.py -m 256:4096:256
    python csrc/bench/dispatch_grid.py -m 256:4096:256 -n 256:4096:512 --combos w16a16 \
        --planner model --csv /tmp/logic_model.csv

The measurement side stays what it always was: the product-grid sweep
(tune_plan_table.py sweep -m/-n/-k), then interleaved A/B for anything that
ships (diff_rows.py). A dense grid here nominates band boundaries; it proves
nothing about performance.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path

import click
import torch

from astrai.extension import ops, plan

COMBOS = {
    "w16a16": (torch.bfloat16, torch.bfloat16),
    "w8a16": (torch.bfloat16, torch.int8),
    "w8a8": (torch.int8, torch.int8),
    "f8a8": (torch.float8_e4m3fn, torch.float8_e4m3fn),
}


def parse_range(value: str) -> list[int]:
    """START:END:STEP (end inclusive) — the same grid format as -m/-n/-k
    everywhere else in the bench scripts."""
    start, stop, step = (int(x) for x in value.split(":"))
    if step <= 0 or stop < start:
        raise click.BadParameter(f"bad range {value!r}: want START:END:STEP")
    return list(range(start, stop + 1, step))


def recipe_name(probe: dict, names: list[str]) -> str:
    return f"{names[probe['cta']][1:]}_s{probe['stages']}_kk{probe['kk']}"


@click.command()
@click.option(
    "-m",
    "--m-grid",
    default="256:4096:256",
    show_default=True,
    callback=lambda _c, _p, v: parse_range(v),
    help="M grid: START:END:STEP, end inclusive.",
)
@click.option(
    "-n",
    "--n-grid",
    default="256:4096:256",
    show_default=True,
    callback=lambda _c, _p, v: parse_range(v),
    help="N grid: START:END:STEP, end inclusive.",
)
@click.option(
    "-k",
    "--k-grid",
    default="256:4096:256",
    show_default=True,
    callback=lambda _c, _p, v: parse_range(v),
    help="K grid: START:END:STEP, end inclusive.",
)
@click.option("--combos", default=",".join(COMBOS), show_default=True)
@click.option(
    "--planner",
    default="hybrid",
    show_default=True,
    type=click.Choice(("hybrid", "model", "table")),
    help="Which chain to map (the shipped default is hybrid).",
)
@click.option("--csv", "csv_path", default=None, type=click.Path(path_type=Path))
def main(
    m_grid: list[int],
    n_grid: list[int],
    k_grid: list[int],
    combos: str,
    planner: str,
    csv_path: Path | None,
):
    # A clean logic map: the shipped builtin rows, no runtime override or
    # injected tier shadowing them (model_capture's --check does the same).
    ops.gemm.set_table("")
    plan.configure(rows="", tier="injected")
    ops.gemm.set_planner(planner)
    names = ops.gemm.get_module("gemm").tile_class_names()

    rows_out: list[dict] | None = [] if csv_path is not None else None
    for combo in (c for c in combos.split(",") if c):
        act, weight = COMBOS[combo]
        decisions: Counter[tuple[str, str]] = Counter()
        # m boundaries where any (n, k) slice flips its pick, and whether k
        # moves the pick at all for a fixed (m, n).
        m_flips: Counter[int] = Counter()
        k_varies = 0
        prev_by_nk: dict[tuple[int, int], tuple[str, str]] = {}
        prev_m: dict[tuple[int, int], int] = {}
        for m in m_grid:
            for n in n_grid:
                pick_by_k: dict[tuple[str, str], int] = {}
                for k in k_grid:
                    d = ops.gemm.probe(m, n, k, act, weight)
                    dec = (d["source"], recipe_name(d, names))
                    decisions[dec] += 1
                    pick_by_k[dec] = k
                    if dec != prev_by_nk.get((n, k)):
                        if (n, k) in prev_by_nk:
                            m_flips[(m + prev_m[(n, k)]) // 2] += 1
                        prev_by_nk[(n, k)] = dec
                    prev_m[(n, k)] = m
                if len(pick_by_k) > 1:
                    k_varies += 1
        cells = len(m_grid) * len(n_grid) * len(k_grid)
        print(f"=== {combo}  planner={planner}  cells={cells}")
        print(f"    distinct decisions: {len(decisions)}")
        for (source, recipe), count in decisions.most_common(8):
            print(f"    {count:6d} ({100 * count / cells:5.1f}%)  {source:8s} {recipe}")
        if len(decisions) > 8:
            print(f"    ... {len(decisions) - 8} more")
        flips = sorted(m_flips)
        if flips:
            per_slice = [m_flips[f] for f in flips]
            print(
                f"    m-flip edges (any n,k slice): {len(flips)} unique,"
                f" median slices/edge {statistics.median(per_slice):.0f},"
                f" max {max(per_slice)}"
            )
            print(f"      {flips}")
        print(
            f"    (m,n) cells whose pick varies with k: "
            f"{k_varies}/{len(m_grid) * len(n_grid)}"
        )
        if rows_out is not None:
            for m in m_grid:
                for n in n_grid:
                    for k in k_grid:
                        d = ops.gemm.probe(m, n, k, act, weight)
                        rows_out.append(
                            {
                                "combo": combo,
                                "m": m,
                                "n": n,
                                "k": k,
                                "source": d["source"],
                                "recipe": recipe_name(d, names),
                                "raster": d["raster"],
                            }
                        )
    ops.gemm.set_planner("")  # restore the shipped default
    if rows_out is not None and csv_path is not None:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
        click.echo(f"wrote {len(rows_out)} rows to {csv_path}")


if __name__ == "__main__":
    main()
