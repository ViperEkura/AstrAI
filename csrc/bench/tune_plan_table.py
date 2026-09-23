"""Plan-table tuning pipeline: sweep candidates, validate holdouts, install.

One CLI, three stages (each subcommand is what one of the old standalone
scripts did):

    sweep     measure every candidate recipe per shape and emit a row file
    validate  interleaved holdout comparison of candidate tables
    run       sweep -> validate -> install into the autotuner cache (the
              default flow; ``--skip-validate`` sweeps and installs only)

Stages run as separate processes (``run`` re-invokes this file), so each
stage gets its own GPU clock/thermal state — the isolation the three
standalone scripts had.

Row tables are served at runtime through ``ops.gemm.set_table`` (no
rebuild, no environment variable); an emitted row file can also be pasted
into csrc/kernels/gemm/plan_table.h's GENERATED block, which does require a
rebuild. The special ``model`` candidate measures every row tier off (the
degraded rows) — the reference of the min-gain mode.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import click
import torch

from astrai.extension import is_available, ops
from astrai.extension.loader import get_module
from astrai.extension.ops.gemm import quant_gemm
from astrai.extension.plan import Tile, device_signature


@click.group(help=__doc__)
def cli() -> None:
    """Plan-table tuning pipeline."""


# Combo name -> (activation dtype, weight dtype). Scales: int8 operands
# require their dequant scale, fp8 accept one optionally (a [1] per-tensor
# scalar), bf16 rejects one. Matches the csrc dispatch (gemm.cu).
COMBOS: dict[str, tuple[torch.dtype, torch.dtype]] = {
    "w16a16": (torch.bfloat16, torch.bfloat16),
    "w8a16": (torch.bfloat16, torch.int8),
    "w8a8": (torch.int8, torch.int8),
    "f8a8_e4m3": (torch.float8_e4m3fn, torch.float8_e4m3fn),
    "f8a8_e5m2": (torch.float8_e5m2, torch.float8_e5m2),
}

# GemmPerfClass ids (gemm.cuh): W16A16 / W8A16 / W8A8 / F8A8.
PERF_CLASS: dict[str, int] = {
    "w16a16": 0,
    "w8a16": 1,
    "w8a8": 2,
    "f8a8_e4m3": 3,
    "f8a8_e5m2": 3,
}

# ---------------------------------------------------------------------------
# The candidate vocabulary, read from the launch ladders.
#
# A recipe is one tile: its CTA class, its ring depth and its k-tile depth,
# written as a row file's (cta, stages, kK) — which is exactly the dispatch
# key dispatch_tile matches on. The vocabulary is single-sourced from the
# compiled binding (ops.gemm.tile_vocabulary -> gemm.cuh gemm_recipes_for):
# the extension returns, per staging pair, every dispatch key the ladders
# carry in dispatch (manifest) order, deduped by its own first-match rule —
# the same rule dispatch_tile applies — so this file cannot disagree with
# policy.cuh's manifests. Names come from the vocabulary record itself
# (``Tile.name``, whose spelling the bench emits on the C++ side), which is
# how dataset recipe strings join without a second mapping — and without a
# second spelling of the format.
# ---------------------------------------------------------------------------

# Operand widths per GemmPerfClass id: W16A16 / W8A16 / W8A8 / F8A8.
PERF_WIDTH: dict[int, tuple[int, int]] = {0: (2, 2), 1: (2, 1), 2: (1, 1), 3: (1, 1)}

# Name tokens in dataset rows and run headers (the structural spelling the
# bench writes; the vocabulary itself comes from the binding below).
_FACTS_RE = re.compile(r"Tile_(\d+)x(\d+)x(\d+)_W(\d+)x(\d+)_S(\d+)(?:_Fast)?")
# The shared ladder every staging path carries: six geometries, one per class
# and ring depth. If the binding below cannot see these, the build is stale.
_SHARED = (
    "Tile_64x64x64_W16x32_S2",
    "Tile_64x64x64_W16x32_S3",
    "Tile_128x64x64_W32x32_S2",
    "Tile_128x64x64_W32x32_S3",
    "Tile_128x128x64_W64x32_S2",
    "Tile_128x128x64_W64x32_S3",
)


def tile_facts(name: str) -> tuple[int, int, int]:
    """A tile's dispatch key: (cta class, stages, ring K)."""
    m, n, k, _wm, _wn, stages = _FACTS_RE.fullmatch(name).groups()
    try:
        return _CLASS_OF[(int(m), int(n))], int(stages), int(k)
    except KeyError as exc:  # a geometry TileClass does not name
        raise RuntimeError(
            f"{name}: CTA geometry absent from the tile_vocabulary binding "
            f"— a stale extension build? Rebuild gemm before tuning."
        ) from exc


def _binding_vocabulary():
    """(ladders in dispatch order, geometry -> TileClass ordinal) from the
    extension binding — the same rows the extension's own planner ranks,
    so the vocabulary cannot drift from the compiled manifests."""
    from astrai.extension import ops  # host-only: no kernel, no device

    ladders: dict[str, list[str]] = {
        "TileManifestCross": [],
        "TileManifest": [],
        "TileManifestByte": [],
    }
    class_of: dict[tuple[int, int], int] = {}
    for row in ops.gemm.tile_vocabulary():
        tile = Tile(*row)  # a record that still unpacks like its row
        cw, ba, bb, cta, _stages, _kk, bm, bn, _wm, _wn, _threads, _smem = tile
        name = tile.name
        ladder = (
            "TileManifestCross"
            if cw
            else "TileManifestByte"
            if (ba, bb) == (1, 1)
            else "TileManifest"
        )
        if name not in ladders[ladder]:
            ladders[ladder].append(name)
        class_of.setdefault((bm, bn), cta)
    return ladders, class_of


# The binding's rows are deduped on the dispatch key already (first manifest
# match per key — the extension's own rule), so the ladders are the reachable
# sets as-is.
LADDERS, _CLASS_OF = _binding_vocabulary()
REACHABLE: dict[str, tuple[str, ...]] = {
    ladder: tuple(tiles) for ladder, tiles in LADDERS.items()
}
# Every recipe the ladders carry, by name: cta class, ring depth, k-tile depth.
RECIPES: dict[str, tuple[int, int, int]] = {
    tile: tile_facts(tile) for tiles in REACHABLE.values() for tile in tiles
}
_missing = [t for t in _SHARED if t not in RECIPES]
if _missing:
    raise RuntimeError(
        f"vocabulary read from the tile_vocabulary binding missed {_missing}: "
        f"a stale extension build? Rebuild gemm before tuning (the binding "
        f"and policy.cuh ship in the same library)."
    )


def ladder_for_widths(ba: int, bb: int) -> str:
    """The grouping the binding uses (manifest_for in policy.cuh): 2-byte
    and mixed pairs get the congruous ladder, 1-byte pairs the byte one, a
    crosswise problem the shared six. The sweep is NT (crosswise 0), so a
    mixed width pair rides the kK=32 twins now (the 1-byte bus is a
    predicated skip)."""
    if ba == 2 and bb == 2:
        return "TileManifest"
    if ba + bb == 3:
        return "TileManifest"
    if ba == 1 and bb == 1:
        return "TileManifestByte"
    return "TileManifestCross"


def ladder_for(perf_class: int) -> str:
    return ladder_for_widths(*PERF_WIDTH[perf_class])


def order_for(perf_class: int) -> tuple[str, ...]:
    """Tie preference for one dtype class: its ladder's manifest order, with
    the model (no row) last — a tie carries no information, so it resolves the
    way dispatch_tile itself would."""
    return REACHABLE[ladder_for(perf_class)] + ("model",)


def short_name(tile: str) -> str:
    """A tile name without the Tile_ dressing, for run headers."""
    return tile.removeprefix("Tile_").removesuffix("_Fast")


def reachability_report() -> list[str]:
    """No lines since the vocabulary came from the binding: the extension
    dedupes on the dispatch key itself (first manifest match per key), so
    every entry this file can see is reachable by construction."""
    return []


# Ring feasibility mirror (policy.cuh ring_smem_bytes vs the device's
# smem opt-in ceiling): over-budget recipes cannot launch and would be
# silently measured as the model/degraded — filter them out per combo.
# class -> CTA (M, N), the inverse of the same map that names the ordinals
# (policy.cuh's kTileClassCta is the C++ home for these numbers).
CTA_GEOM: dict[int, tuple[int, int]] = {v: k for k, v in _CLASS_OF.items()}
SMEM_OPTIN = 101376  # MaxSharedMemoryPerBlockOptin: 99KB on sm_89 and sm_120
_DTYPE_BYTES = {
    torch.bfloat16: 2,
    torch.int8: 1,
    torch.float8_e4m3fn: 1,
    torch.float8_e5m2: 1,
}
BYTES: dict[str, tuple[int, int]] = {
    combo: (_DTYPE_BYTES[a], _DTYPE_BYTES[b]) for combo, (a, b) in COMBOS.items()
}


def ring_smem_bytes(stages: int, cta: int, kk: int, ba: int, bb: int) -> int:
    bm, bn = CTA_GEOM[cta]
    return (stages + 1) * kk * (bm * ba + bn * bb)


def candidate_recipes(combo: str) -> tuple[str, ...]:
    """Every recipe a row can select for this combo, in preference order, plus
    the model (no table) reference. Ladder membership carries the width and
    depth rules for free — a kK=32 tile is in no 1-byte ladder, the wide CTA in
    no 2-byte one — so the only gate left here is the device's own: the ring
    against the smem opt-in ceiling. What this gets wrong, the tag probe
    catches rather than records."""
    ba, bb = BYTES[combo]
    return tuple(
        recipe
        for recipe in REACHABLE[ladder_for_widths(ba, bb)]
        if ring_smem_bytes(
            RECIPES[recipe][1], RECIPES[recipe][0], RECIPES[recipe][2], ba, bb
        )
        <= SMEM_OPTIN
    ) + ("model",)


_TAG_RE = re.compile(r"\[gemm-plan\] (override|injected|builtin|model|degraded)\b")


def _last_tag(text: str) -> str:
    """The decision source of the launch that just happened (the planner
    logs one decision line per dispatch, then the launch line)."""
    tags = _TAG_RE.findall(text)
    if not tags:
        return "?"
    return "degraded (no table)" if tags[-1] == "degraded" else tags[-1]


def _planned_tag(run) -> str:
    """The decision source of one launch, with the plan log on for that
    launch only (ops.gemm.set_log toggles the C-side flag around it, so its
    fprintf stays out of the timed loop). The tag is read from fd 2 — the
    log is C-level fprintf, not Python's stderr."""
    log = tempfile.TemporaryFile()
    saved = os.dup(2)
    ops.gemm.set_log(True)
    os.dup2(log.fileno(), 2)
    try:
        run()
        torch.cuda.synchronize()
    finally:
        ops.gemm.set_log(False)
        os.dup2(saved, 2)
        os.close(saved)
    # fd 2 shares this file's offset, so the write above left it at the end.
    log.seek(0)
    return _last_tag(log.read().decode(errors="replace"))


def _check_recipe(perf_class: int, recipe: str) -> None:
    """A recipe is legal for a class only if that class's ladder carries it —
    one check covering the width, ring-depth and CTA-geometry rules at once.
    The sweep only ever measures ladder members, so a name outside them means
    the input measurements are not this script's: fail loudly rather than emit
    a row dispatch would silently fail to launch."""
    if recipe == "model":
        return
    if recipe not in REACHABLE[ladder_for(perf_class)]:
        raise ValueError(
            f"recipe {recipe!r} is not in {ladder_for(perf_class)}, the ladder "
            f"perf_class {perf_class} dispatches over — no row naming it could "
            f"launch. Measurements naming one cannot come from this sweep; "
            f"re-measure instead of re-searching that JSON."
        )


def _candidate_rows() -> dict[str, str]:
    """One synthetic open row per candidate, as inline row text — the
    recipe's (cta, stages, kK) on -1/-1 keys, so the row is the only plan
    source and the planner gates it on the ring and the operand widths,
    exactly like an emitted row."""
    return {
        recipe: f"0 0 0 0 -1 -1 {cta} {stages} 0 {kk}"
        for recipe, (cta, stages, kk) in RECIPES.items()
    }


def parse_positive_ints(value: str) -> tuple[int, ...]:
    """START:END:STEP (end inclusive) or a comma list.

    The stride form is what a tuning grid is described in
    (``256:4096:256`` is the 16 values 256 through 4096); the list form
    stays for irregular grids. Every -m/-n/-k option in the bench scripts
    takes this format.
    """
    value = value.strip()
    parts = [p for p in value.split(",") if p.strip()]
    if len(parts) == 1 and ":" in value:
        start, stop, step = (int(x) for x in value.split(":"))
        if step <= 0 or stop < start:
            raise click.BadParameter(f"bad range {value!r}: want START:END:STEP")
        return tuple(range(start, stop + 1, step))
    return tuple(int(part) for part in parts)


def parse_shape(value: str) -> tuple[str, int, int]:
    name, n, k = value.split(":")
    return name, int(n), int(k)


def make_scale(operand_dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    # bf16 operands reject scales; int8 requires its dequant scale, fp8
    # takes one optionally — a [1] per-tensor float32 scalar covers both.
    if operand_dtype == torch.bfloat16:
        return None
    return torch.tensor([0.1], dtype=torch.float32, device=device)


def random_operand(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    base = torch.rand(shape, device=device, dtype=torch.float32) * 0.2 - 0.1
    return base.to(dtype)


def measure(fn, warmup: int, iterations: int, trials: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) / iterations)
    return best


def sweep(
    m_values: tuple[int, ...],
    shapes: list[tuple[str, int, int]],
    combos: tuple[str, ...],
    batch: int,
    warmup: int,
    iterations: int,
    trials: int,
    wanted: tuple[str, ...] = (),
) -> list[dict]:
    """One process, shape outer loop — every candidate measured back-to-back."""
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    if not is_available("gemm"):
        raise click.ClickException(
            "the built gemm kernel is required (rebuild "
            "the extension with CSRC_KERNELS=true first)"
        )

    # Candidates are one-row tables toggled per launch through the plan API
    # (ops.gemm.set_table), so all candidates at a shape share its GPU
    # clock/thermal state. The "model" candidate measures the degraded
    # rows: it is the "this band has no table row" reference of the
    # min-gain mode, not a planner.
    candidate_rows = _candidate_rows()
    tags: dict[tuple[str, str], str] = {}

    device = torch.device(torch.cuda.current_device())
    # The "model" candidate measures the no-row reference: with the row
    # tiers off, whichever planner the process runs (the shipped hybrid
    # default answers with the analytical model; a table-pinned process
    # with the degraded ladder).
    no_row_source = (
        "degraded (no table)" if ops.gemm.state()["planner"] == "table" else "model"
    )
    torch.manual_seed(0)
    results = []
    for combo in combos:
        act_dtype, weight_dtype = COMBOS[combo]
        perf_class = PERF_CLASS[combo]
        a_scale = make_scale(act_dtype, device)
        b_scale = make_scale(weight_dtype, device)
        for _name, n, k in shapes:
            weight = random_operand((batch, n, k), weight_dtype, device)
            for m in m_values:
                acts = random_operand((batch, m, k), act_dtype, device)

                def run(acts=acts, weight=weight, a_scale=a_scale, b_scale=b_scale):
                    return quant_gemm(acts, weight, a_scale, b_scale)

                for recipe in candidate_recipes(combo):
                    if wanted and recipe not in wanted:
                        continue
                    if recipe == "model":
                        # "-" = every row tier off: the no-row reference
                        # (the model under the shipped default).
                        ops.gemm.set_table("-")
                    else:
                        ops.gemm.set_table(candidate_rows[recipe])
                    expected = no_row_source if recipe == "model" else "override"
                    if (combo, recipe) not in tags:
                        tags[(combo, recipe)] = _planned_tag(run)
                    if tags[(combo, recipe)] != expected:
                        # The planner demoted the candidate (ring, width or
                        # depth gate) and served something else: those numbers
                        # are the fallback's, not this recipe's. Naming them
                        # after the recipe is how a sweep reports a
                        # configuration it never ran.
                        print(
                            f"  ! {short_name(recipe):28s} {combo:14s}: planner "
                            f"served {tags[(combo, recipe)]!r}, skipped",
                            flush=True,
                        )
                        continue
                    run()  # steady state for this (shape, recipe)
                    torch.cuda.synchronize()
                    ms = measure(run, warmup, iterations, trials)
                    flops = 2.0 * batch * m * n * k
                    results.append(
                        {
                            "combo": combo,
                            "perf_class": perf_class,
                            "m": m,
                            "n": n,
                            "k": k,
                            "batch": batch,
                            "recipe": recipe,
                            "planned": tags[(combo, recipe)],
                            "ms": ms,
                            "tflops": flops / ms / 1e12,
                        }
                    )
                    print(
                        f"{recipe:8s} {combo:14s} b{batch} m{m:5d} n{n:6d} "
                        f"k{k:5d} {ms * 1e3:8.3f} ms {flops / ms / 1e12:7.1f} "
                        f"TFLOPS",
                        flush=True,
                    )
    ops.gemm.set_table("")
    return results


def band_edges(values: tuple[int, ...]) -> list[tuple[int, int]]:
    """(min, max] bands for sorted unique values; 0 = open upper."""
    uniq = sorted(set(values))
    if len(uniq) <= 1:
        return [(0, 0)]
    mid_pairs = itertools.pairwise(uniq)
    mids = [a + (b - a) // 2 for a, b in mid_pairs]
    edges = [(0, mids[0])]
    edges += [(mids[i], mids[i + 1]) for i in range(len(mids) - 1)]
    edges.append((mids[-1], 0))
    return edges


def build_rows(
    results: list[dict],
    min_gain: float = 0.0,
    full_coverage: bool = False,
) -> list[str]:
    # (perf_class, m, n) -> {recipe: best tflops across the k/batch grid}.
    aggregate: dict[tuple[int, int, int], dict[str, float]] = {}
    m_values: set[int] = set()
    n_values: set[int] = set()
    for point in results:
        expected = (
            "degraded (no table)"
            if point["recipe"] == "model" and point.get("planned") == "degraded"
            else ("model" if point["recipe"] == "model" else "override")
        )
        if "planned" in point and point["planned"] != expected:
            continue  # demoted candidate: those numbers are the fallback's
        # datasets saved before 2026-09-16 spell names with a `_Fast` suffix
        recipe = point["recipe"].removesuffix("_Fast")
        _check_recipe(point["perf_class"], recipe)
        key = (point["perf_class"], point["m"], point["n"])
        over = aggregate.setdefault(key, {})
        tflops = point["tflops"]
        over[recipe] = max(over.get(recipe, 0.0), tflops)
        m_values.add(point["m"])
        n_values.add(point["n"])

    sorted_m = sorted(m_values)
    sorted_n = sorted(n_values)
    m_bands = band_edges(tuple(sorted_m))
    n_bands = band_edges(tuple(sorted_n))

    rows: list[str] = []
    for perf_class in sorted({pt[0] for pt in aggregate}):
        for n_idx, (n_min, n_max) in enumerate(n_bands):
            # Winner per m band at this n band; merge adjacent m runs that
            # pick the same recipe into one row.
            if full_coverage:
                # The table is the only production dispatch: every band
                # gets a row (no min-gain gate, no model fallback), and
                # ties resolve to the class ladder's own order.
                recipes = [
                    _best_forced(aggregate, (perf_class, m, sorted_n[n_idx]))
                    for m in sorted_m
                ]
            else:
                recipes = [
                    _winner(aggregate, (perf_class, m, sorted_n[n_idx]), min_gain)
                    for m in sorted_m
                ]
            run_start = 0
            for i in range(1, len(recipes) + 1):
                if i == len(recipes) or recipes[i] != recipes[run_start]:
                    recipe = recipes[run_start]
                    if recipe != "model":
                        m_min, m_max = m_bands[run_start][0], m_bands[i - 1][1]
                        cta, stages, kk = RECIPES[recipe]
                        rows.append(
                            f"{m_min} {m_max} {n_min} {n_max} {perf_class} 0 "
                            f"{cta} {stages} 0 {kk}"
                        )
                    run_start = i
        if full_coverage:
            # Tail catch-all row per class: shapes outside the grid bands
            # still hit (the table never misses). The default recipe is
            # the one measured at the grid's top-right corner (largest M
            # and N) — shapes beyond the grid are big shapes, where the
            # corner's winner beats the grid's most common (small-shape
            # biased) winner.
            corner = _best_forced(aggregate, (perf_class, sorted_m[-1], sorted_n[-1]))
            cta, stages, kk = RECIPES[corner]
            rows.append(f"0 0 0 0 {perf_class} 0 {cta} {stages} 0 {kk}")
    # A band row that already spans everything (the grid's own merge) makes the
    # catch-all identical; identical rows are one decision, and the old
    # compactor that used to drop them is gone.
    return list(dict.fromkeys(rows))


def emit_cpp_rows(rows: list[str]) -> str:
    """The same rows as plan_table.h initializers, grouped per class.

    Field order is TableRow's own (cta first, then the bands), the cta
    ordinal expands through the binding's class-name table (C++ owns the
    spellings), and each group is labelled with its builtin array so the
    paste is a straight replacement. The header's static_asserts then check
    the class keying, which a hand-edited ordinal would silently get wrong.
    """
    names = get_module("gemm").tile_class_names()
    by_class: dict[int, list[str]] = {}
    for line in rows:
        m_min, m_max, n_min, n_max, perf, crosswise, cta, stages, _raster, kk = (
            line.split()
        )
        by_class.setdefault(int(perf), []).append(
            f"    {{TileClass::{names[int(cta)]}, {m_min}, {m_max}, "
            f"{n_min}, {n_max}, {perf}, {crosswise}, {stages}, 0, {kk}}},"
        )
    classes = ("W16A16", "W8A16", "W8A8", "F8A8")
    out = ["// Generated by csrc/bench/tune_plan_table.py sweep --emit cpp."]
    # std::array form, not a raw C array: the GENERATED block declares
    # std::array, and the nested braces are what std::array's single member
    # needs (a raw-array paste here fails with "too many initializer
    # values", which is a build break, not a fallback).
    for perf in range(4):
        group = by_class.get(perf, [])
        out.append(f"// kBuiltinPlan{classes[perf]}:")
        out.append(
            f"static constexpr std::array<TableRow, {len(group)}> "
            f"kBuiltinPlan{classes[perf]} = {{{{"
        )
        out.extend(group)
        out.append("}};")
    return "\n".join(out) + "\n"


def _best_forced(
    aggregate: dict[tuple[int, int, int], dict[str, float]],
    key: tuple[int, int, int],
) -> str:
    # Best measured recipe; a tie carries no information, so it resolves the
    # way dispatch_tile itself would: the ladder's manifest order.
    order = order_for(key[0])
    over = aggregate.get(key, {})
    best = max((v for r, v in over.items() if r != "model"), default=0.0)
    for recipe in order:
        if recipe == "model":
            continue
        if over.get(recipe, 0.0) == best:
            return recipe
    return order[0]


def _winner(
    aggregate: dict[tuple[int, int, int], dict[str, float]],
    key: tuple[int, int, int],
    min_gain: float = 0.0,
) -> str:
    # A forced recipe only wins when it beats every other option (model
    # included) by at least min_gain (a fraction of its throughput): the
    # sweep's near-ties are measurement noise, and a table row claimed on
    # a tie would flip on every re-run. Ties resolve to "model" (no row).
    over = aggregate.get(key, {})
    if not over:
        return "model"
    best = max(over.values())
    for recipe in order_for(key[0]) + ("model",):
        if over.get(recipe, 0.0) == best:
            winner = recipe
            break
    else:
        return "model"
    if winner == "model" or min_gain <= 0.0:
        return winner
    second = max((v for r, v in over.items() if r != winner), default=0.0)
    if (best - second) < min_gain * best:
        return "model"
    return winner


@cli.command("sweep")
@click.option(
    "-m",
    "--m-values",
    default="512,2048,4096",
    show_default=True,
    callback=lambda _c, _p, v: parse_positive_ints(v),
    help="M grid: START:END:STEP (end inclusive) or a comma list.",
)
@click.option(
    "--shapes",
    "shape_values",
    multiple=True,
    help="NAME:N:K — named weight shapes; give the grid via -n/-k instead "
    "for the product sweep.",
)
@click.option(
    "-n",
    "--n-values",
    default=None,
    callback=lambda _c, _p, v: parse_positive_ints(v) if v else None,
    help="N grid (START:END:STEP, end inclusive); with -k, the sweep grid.",
)
@click.option(
    "-k",
    "--k-values",
    default=None,
    callback=lambda _c, _p, v: parse_positive_ints(v) if v else None,
    help="K grid (START:END:STEP, end inclusive); with -n, the sweep grid.",
)
@click.option("--batch", default=1, show_default=True, help="Batch dim b.")
@click.option(
    "--combos",
    default=",".join(COMBOS),
    show_default=True,
    callback=lambda _c, _p, v: tuple(
        part.strip() for part in v.split(",") if part.strip()
    ),
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Plan-table row file (plan.set_table(path) serves it).",
)
@click.option(
    "--emit",
    type=click.Choice(("rows", "cpp")),
    default="rows",
    show_default=True,
    help="rows: the row-file syntax; cpp: TileClass initializers, "
    "grouped per class, ready to paste into plan_table.h's GENERATED block.",
)
@click.option("--warmup", type=click.IntRange(min=1), default=10, show_default=True)
@click.option("--iterations", type=click.IntRange(min=1), default=50, show_default=True)
@click.option("--trials", type=click.IntRange(min=1), default=3, show_default=True)
@click.option(
    "--min-gain",
    type=click.FloatRange(min=0.0),
    default=1.0,
    show_default=True,
    help="Minimum relative gain (%) a forced recipe must show over the "
    "runner-up to claim a row; smaller leads are treated as ties and keep "
    "the degraded bands. Ignored with --full-coverage (every band gets a row, "
    "ties resolve to the big>narrow>small preference).",
)
@click.option(
    "--full-coverage",
    is_flag=True,
    default=False,
    help="Emit a row for every (class, M/N band): production dispatch "
    "becomes the table alone (dispatch is table-only — no row is ever "
    "skipped), and each class ends with a catch-all row so no shape can "
    "miss the table.",
)
@click.option(
    "--save-results",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Dump the raw per-point measurements to JSON (reproducible search: "
    "rebuild rows from it with --results-json).",
)
@click.option(
    "--results-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Skip the sweep: build rows from a JSON saved by --save-results.",
)
@click.option(
    "--recipes",
    "recipe_filter",
    default=None,
    help="Comma-separated subset of each ladder's recipes to measure "
    "(default: every reachable one). --list-recipes prints the names.",
)
@click.option(
    "--list-recipes",
    is_flag=True,
    default=False,
    help="Print the candidate vocabulary this process reads out of policy.cuh, "
    "with the tiles no row can reach, and exit.",
)
def plan_table_command(
    m_values: tuple[int, ...],
    shape_values: tuple[str, ...],
    n_values: tuple[int, ...] | None,
    k_values: tuple[int, ...] | None,
    batch: int,
    combos: tuple[str, ...],
    output: Path,
    emit: str,
    warmup: int,
    iterations: int,
    trials: int,
    min_gain: float,
    full_coverage: bool,
    save_results: Path | None,
    results_json: Path | None,
    recipe_filter: str | None,
    list_recipes: bool,
) -> None:
    """Sweep every candidate recipe (interleaved) at the M x shape grid."""
    if list_recipes:
        for ladder, tiles in REACHABLE.items():
            click.echo(f"{ladder}:")
            for tile in tiles:
                cta, stages, kk = RECIPES[tile]
                click.echo(f"  {tile:44s} cta{cta} s{stages} k{kk}")
        for line in reachability_report():
            click.echo(f"  unreachable - {line}")
        click.echo("  model  degraded bands (every row tier off), the reference")
        return
    wanted = tuple(
        part.strip() for part in (recipe_filter or "").split(",") if part.strip()
    )
    unknown_recipes = sorted(set(wanted) - set(RECIPES) - {"model"})
    if unknown_recipes:
        raise click.BadParameter(
            f"unknown recipes: {', '.join(unknown_recipes)}; --list-recipes "
            f"prints the vocabulary read from policy.cuh"
        )
    unknown = [combo for combo in combos if combo not in COMBOS]
    if unknown:
        raise click.BadParameter(f"unknown combos: {', '.join(unknown)}")
    if results_json is not None:
        results = json.loads(results_json.read_text())
    else:
        if shape_values:
            shapes = [parse_shape(value) for value in shape_values]
        elif n_values or k_values:
            if not (n_values and k_values):
                raise click.BadParameter(
                    "-n and -k go together — the sweep grid is their product"
                )
            shapes = [(f"n{n}k{k}", n, k) for n in n_values for k in k_values]
        else:
            shapes = [("n4096k4096", 4096, 4096), ("n14336k4096", 14336, 4096)]

        click.echo(
            f"--- interleaved sweep ({len(combos)} combos x {len(shapes)} shapes "
            f"x {len(m_values)} M x b={batch}, recipes measured per shape)"
            + ("; full coverage" if full_coverage else "")
        )
        for line in reachability_report():
            click.echo(f"# unreachable by any row: {line}")
        for combo in combos:
            cands = [r for r in candidate_recipes(combo) if not wanted or r in wanted]
            click.echo(
                f"# {combo}: {len(cands)} candidates -> "
                + ", ".join(short_name(c) for c in cands)
            )
        results = sweep(
            m_values, shapes, combos, batch, warmup, iterations, trials, wanted
        )
        if save_results is not None:
            save_results.write_text(json.dumps(results))
            click.echo(f"saved measurements to {save_results}")

    rows = build_rows(
        results,
        min_gain=min_gain / 100.0,
        full_coverage=full_coverage,
    )
    header = (
        "# AOT dispatch rows: m_min m_max n_min n_max perf_class crosswise "
        "cta stages raster [k]\n"
        "# (min, max] bands, 0 = open; perf_class 0..3 (W16A16/W8A16/W8A8/"
        "F8A8); crosswise 0 = NT; cta 0 small64 / 1 narrow128x64 / 2 big128 / "
        "3 wide128x256; raster 0 "
        "= auto.\n"
        "# k is the row's ring K, optional (an omitted field keeps 64): a "
        "kK=32 row\n"
        "# only survives a dual-2-byte pair, so the sweep drops that "
        "candidate for\n"
        "# every other combo rather than measuring the fallback under its "
        "name.\n"
        "# The sweep times quant_gemm's fused-linear (NT) layout, so every "
        "row carries\n"
        "# crosswise 0: TT/TN shapes miss this table and take the degraded "
        "bands in C++.\n"
        "# Generated by csrc/bench/tune_plan_table.py sweep; tune the grid then "
        "re-run.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if emit == "cpp":
        text = emit_cpp_rows(rows)
        output.write_text(text)
        click.echo(f"wrote {len(rows)} rows to {output} (cpp initializers)")
        click.echo(
            "paste each group into its kBuiltinPlan<CLASS> array between "
            "the GENERATED markers"
        )
    else:
        output.write_text(header + "\n".join(rows) + ("\n" if rows else ""))
        click.echo(f"wrote {len(rows)} rows to {output}")
        click.echo(f"use: plan.set_table({output!r})")


def parse_holdout_shape(value: str) -> tuple[str, int, int, int]:
    name, m, n, k = value.split(":")
    return name, int(m), int(n), int(k)


def apply_table(name: str, path: str | None, hybrid: bool = False) -> None:
    # "model" runs the analytical planner alone; "none" turns every row
    # tier off (the degraded reference) — the aliases keep old
    # invocations working.
    #
    # hybrid=True keeps the shipped planner chain instead of pinning the
    # table-only mode: rows are installed at the override tier and the
    # model still answers where no row matches. That is the configuration
    # production runs, and the only mode in which a DIFF row table (rows
    # that deliberately miss the bands the model wins) can be scored — in
    # table-only mode a miss falls to the degraded ladder, which is a
    # different deployment, not this one.
    if name == "model":
        ops.gemm.set_planner("")
        ops.gemm.set_table("-" if hybrid else "")
        return
    ops.gemm.set_planner("" if hybrid else "table")
    if name == "none":
        # Every row tier off: the degraded-bands reference.
        ops.gemm.set_table("-")
    else:
        ops.gemm.set_table(path or "")


def validate(
    tables: dict[str, str | None],
    shapes: list[tuple[str, int, int, int]],
    combos: tuple[str, ...],
    batch: int,
    warmup: int,
    iterations: int,
    trials: int,
    hybrid: bool = False,
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
                    apply_table(table, tables[table], hybrid)
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


@cli.command("validate")
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
    help="NAME:M:N:K — named holdout shape (NT layout).",
)
@click.option(
    "-m",
    "--m-grid",
    default=None,
    callback=lambda _c, _p, v: parse_positive_ints(v) if v else None,
    help="Holdout M grid (START:END:STEP, end inclusive); with -n/-k.",
)
@click.option(
    "-n",
    "--n-grid",
    default=None,
    callback=lambda _c, _p, v: parse_positive_ints(v) if v else None,
    help="Holdout N grid (START:END:STEP, end inclusive); with -m/-k.",
)
@click.option(
    "-k",
    "--k-grid",
    default=None,
    callback=lambda _c, _p, v: parse_positive_ints(v) if v else None,
    help="Holdout K grid (START:END:STEP, end inclusive); with -m/-n.",
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
    "--mode",
    type=click.Choice(("table", "hybrid")),
    default="table",
    show_default=True,
    help="table pins table-only dispatch; hybrid keeps the shipped "
    "chain (rows -> model) — required for diff row tables.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional JSON dump of the per-task best-ms records.",
)
def validate_command(
    tables: tuple[str, ...],
    shape_values: tuple[str, ...],
    m_grid: tuple[int, ...] | None,
    n_grid: tuple[int, ...] | None,
    k_grid: tuple[int, ...] | None,
    combos: tuple[str, ...],
    batch: int,
    warmup: int,
    iterations: int,
    trials: int,
    mode: str,
    output: Path | None,
) -> None:
    """Holdout: score candidate tables on out-of-grid and llm-ish shapes."""
    table_map: dict[str, str | None] = {}
    for spec in tables:
        name, _, path = spec.partition(":")
        table_map[name] = path if path else None
    shapes = [parse_holdout_shape(value) for value in shape_values]
    if m_grid or n_grid or k_grid:
        if not (m_grid and n_grid and k_grid):
            raise click.BadParameter(
                "-m/-n/-k go together — the holdout grid is their product"
            )
        shapes += [
            (f"m{m}n{n}k{k}", m, n, k) for m in m_grid for n in n_grid for k in k_grid
        ]
    if not shapes:
        raise click.BadParameter("give --shapes NAME:M:N:K and/or the -m/-n/-k grid")

    baseline = list(table_map)[0]
    records = validate(
        table_map,
        shapes,
        combos,
        batch,
        warmup,
        iterations,
        trials,
        hybrid=(mode == "hybrid"),
    )

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


# Repo root: the stage subprocesses run with it as their cwd (the astrai
# package must import from the checkout).
REPO = Path(__file__).resolve().parents[2]

# Holdout grid: decode-thin, decode-wide, prefill, a wide-MLP N, a
# narrow-N large-M point, the mid-K cell where the 2026-09-13 sweep's fp8
# data proved optimistic (it caught a whole class of bad rows), and the
# parity square. All NT (the layout the sweep times).
DEFAULT_HOLDOUT = (
    "decode1:1:4096:4096",
    "decode128:128:4096:4096",
    "prefill:2048:4096:4096",
    "wide_mlp:512:11008:4096",
    "narrow_nm:4096:1024:4096",
    "midk:1024:4096:2048",
    "parity:4096:4096:4096",
)

# Combo name -> plan-table class (the perf_class column rows key on).
_COMBO_CLASS = {"w16a16": 0, "w8a16": 1, "w8a8": 2, "f8a8": 3}


def _class_of_combo(combo: str) -> int:
    return _COMBO_CLASS[combo.split("_", 1)[0]]


def _clamp_m_domain(rows_path: Path, floor: int) -> None:
    """Clamp every row's M band into the swept domain. A full-coverage
    catch-all leaves gen with m_min=0, claiming every M below the sweep's
    grid — the decode shapes there are unmeasured, and the first baseline
    install lost up to 400% on exactly that (2026-09-13: class-3 wide and
    class-0 big catch-alls at m=1). Below the floor the builtin table
    serves, which is what the (min, max] band semantics make natural."""
    out = []
    for line in rows_path.read_text().splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            out.append(line)
            continue
        f = body.split()
        m_min, m_max = int(f[0]), int(f[1])
        if m_max and m_max <= floor:
            continue  # row lies wholly under the swept floor
        if m_min < floor:
            f[0] = str(floor)
        out.append(" ".join(f))
    rows_path.write_text("\n".join(out) + "\n")


def _drop_classes(rows_path: Path, classes: set[int]) -> None:
    """Remove every row keyed for `classes`; their shapes fall through the
    injected file to the class's builtin table at lookup."""
    kept = [
        line
        for line in rows_path.read_text().splitlines()
        if not (line.split("#", 1)[0].strip() and int(line.split()[4]) in classes)
    ]
    rows_path.write_text("\n".join(kept) + "\n")


def _device_facts() -> dict:
    """The device facts the planner prices against, straight from the
    binding (the cache key cannot drift from the autotuner's)."""
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    from astrai.extension.ops.gemm import facts

    return facts()


def _run(cmd: list[str]) -> str:
    click.echo(f"+ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True, cwd=REPO
    )
    if proc.returncode != 0:
        click.echo(proc.stdout)
        click.echo(proc.stderr)
        raise click.ClickException(f"stage failed: {cmd[1]}")
    return proc.stdout + proc.stderr


def _install(rows_path: Path, out_dir: Path, sig: str, argv: list[str]) -> Path:
    assert device_signature(_device_facts()) == sig, "signature drifted mid-run"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{sig}.rows"
    header = (
        f"# baseline rows for {sig}\n"
        f"# generated {date.today():%Y-%m-%d} by csrc/bench/tune_plan_table.py run\n"
        f"# argv: {' '.join(argv)}\n"
    )
    dest.write_text(header + rows_path.read_text())
    return dest


@cli.command("run")
@click.option("--m-values", default=None, help="Passthrough to sweep.")
@click.option(
    "--shapes",
    "shape_values",
    multiple=True,
    help="NAME:N:K passthrough to sweep.",
)
@click.option(
    "--n-values",
    default=None,
    help="Explicit N grid passthrough (used when --shapes is empty).",
)
@click.option(
    "--k-values",
    default=None,
    help="Explicit K grid passthrough (used when --shapes is empty).",
)
@click.option("--combos", default=None, help="Comma list passthrough to sweep.")
@click.option(
    "--batch", default=1, show_default=True, help="Passthrough to both stages."
)
@click.option(
    "--warmup", type=int, default=None, help="Passthrough (per-stage defaults)."
)
@click.option(
    "--iterations", type=int, default=None, help="Passthrough (per-stage defaults)."
)
@click.option(
    "--trials", type=int, default=None, help="Passthrough (per-stage defaults)."
)
@click.option(
    "--validate-shapes",
    "holdout",
    multiple=True,
    help="NAME:M:N:K holdout shapes ADDED to the default regression gate.",
)
@click.option(
    "--skip-validate", is_flag=True, default=False, help="Sweep and install only."
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Install dir (default: the autotuner cache dir — ASTR_GEMM_TUNE_DIR"
    " or ~/.astrai/cache/gemm_plans).",
)
def main(
    m_values: str | None,
    shape_values: tuple[str, ...],
    n_values: str | None,
    k_values: str | None,
    combos: str | None,
    batch: int,
    warmup: int | None,
    iterations: int | None,
    trials: int | None,
    holdout: tuple[str, ...],
    skip_validate: bool,
    out_dir: Path | None,
) -> None:
    facts = _device_facts()
    sig = device_signature(facts)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    click.echo(f"device: {props.name} ({sig})")

    if out_dir is None:
        out_dir = Path(
            os.environ.get("ASTR_GEMM_TUNE_DIR", "~/.astrai/cache/gemm_plans")
        ).expanduser()

    timing = ["--batch", str(batch)]
    for name, value in (
        ("--warmup", warmup),
        ("--iterations", iterations),
        ("--trials", trials),
    ):
        if value is not None:
            timing += [name, str(value)]

    with tempfile.TemporaryDirectory(prefix="gemm_baseline_") as tmp:
        candidate = Path(tmp) / "candidate.rows"
        gen_cmd = [
            sys.executable,
            str(Path(__file__)),
            "sweep",
            "--full-coverage",
            "--output",
            candidate,
        ]
        if m_values:
            gen_cmd += ["--m-values", m_values]
        for shape in shape_values:
            gen_cmd += ["--shapes", shape]
        if n_values:
            gen_cmd += ["--n-values", n_values]
        if k_values:
            gen_cmd += ["--k-values", k_values]
        if combos:
            gen_cmd += ["--combos", combos]
        gen_cmd += timing
        click.echo(_run(gen_cmd).strip().splitlines()[-1])  # the "wrote N rows" line
        # Rows only over the swept M domain (see _clamp_m_domain): the
        # floor is the smallest M this run asked gen to sweep.
        floor = min(
            int(v) for v in (m_values or "512,2048,4096").split(",") if v.strip()
        )
        _clamp_m_domain(candidate, floor - 1)

        if not skip_validate:
            all_holdout = list(dict.fromkeys(DEFAULT_HOLDOUT + tuple(holdout)))
            val_cmd = [
                sys.executable,
                str(Path(__file__)),
                "validate",
                "--table",
                "builtin:",
                "--table",
                f"tuned:{candidate}",
            ]
            for shape in all_holdout:
                val_cmd += ["--shapes", shape]
            if combos:
                val_cmd += ["--combos", combos]
            val_cmd += timing

            def summarize(report: str) -> list[str]:
                regressions = [
                    line for line in report.splitlines() if line.startswith("  tuned ")
                ]
                click.echo(
                    "\n".join(
                        line
                        for line in report.splitlines()
                        if "totals" in line
                        or "no >=" in line
                        or line.startswith("  tuned ")
                    )
                )
                return regressions

            regressions = summarize(_run(val_cmd))
            if regressions:
                # One class's bad rows must not reject the classes that
                # measured clean: drop the regressing classes (their shapes
                # fall through the injected file to the builtin table) and
                # re-gate once. Recipe-level surgery inside a class stays
                # manual — that judgement wants the dispute adjudicated,
                # not a rule.
                bad = {_class_of_combo(line.split()[1]) for line in regressions}
                click.echo(
                    f"regressions in classes {sorted(bad)}; dropping their rows "
                    "and re-validating"
                )
                _drop_classes(candidate, bad)
                regressions = summarize(_run(val_cmd))
            if regressions:
                debug_copy = out_dir / f"{sig}.rejected.rows"
                debug_copy.parent.mkdir(parents=True, exist_ok=True)
                debug_copy.write_text(candidate.read_text())
                raise click.ClickException(
                    f"candidate regressed {len(regressions)} holdout shapes by >=2%; "
                    f"rows kept at {debug_copy} for inspection"
                )

        dest = _install(candidate, out_dir, sig, sys.argv[1:])
    click.echo(f"installed: {dest}")
    click.echo(
        f'serve without rebuild: ops.gemm.set_table("{dest}")\n'
        "(the runtime autotuner picks it up as its cache automatically)"
    )


if __name__ == "__main__":
    cli()
