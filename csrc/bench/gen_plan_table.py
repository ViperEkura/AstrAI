"""Generate AOT plan-table rows from measured shape sweeps.

The candidate recipes are not a list in this file: they are read out of
csrc/kernels/gemm/policy.cuh — the tile aliases and the three launch ladders
that dispatch_tile resolves a row against. A hand-written candidate list
drifts both ways: it keeps recipes the ladders dropped (so the sweep measures
the degraded fallback and files it under that recipe's name) and it omits
tiles they carry (so a tile that wins on the shapes at hand is never
measured — how the wide CTA and the two kK=32 s3 tiles went missing). The
vocabulary here is whatever the ladders say, each recipe named by its tile's
own structural token, with the entries no row can reach reported rather than
quietly ignored.

Every dtype combo is swept at each (M, N, K, batch) grid point under every
candidate its ladder carries; the measured winner per point becomes a
dispatch-table row (csrc/kernels/gemm/plan_table.h). Rows are keyed (M, N)
bands per dtype class and carry the winning recipe's CTA class, ring depth and
k-tile depth, so a conflict across K or batch at one (M, N) resolves to the
recipe with the best tflops.

The candidates are measured back-to-back at each shape in a single process
(shape outer loop, recipe inner loop): each candidate runs as a one-row
ASTR_GEMM_TABLE file toggled per launch (the row source re-reads its env per
call, so the comparison happens under the same GPU clock/thermal state). A
sweep that measured whole recipe batches in separate processes compared the
big CTA (measured first) against the small CTA (measured 30 minutes later)
under different boost states and picked systematically wrong winners.

A candidate whose row the planner demotes (ring, operand width or depth gate)
is served by the degraded bands instead, so a sweep that recorded tflops alone
would report the fallback's number under that recipe's name — how the big CTA's
s3 ring and the wide CTA once came to look measured. Every candidate is probed
first, with the plan log on for a single launch, and a candidate that does not
land on its own row is dropped before any measurement is attributed to it.

Usage (rows to a runtime-override file, no rebuild):
    python csrc/bench/gen_plan_table.py \
        --m-values 512,2048,4096 \
        --shapes "qkv:4096:4096,up_gate:14336:4096" \
        --batch 1 --combos w16a16 --output plan_table.txt

--list-recipes prints the vocabulary this process would sweep (and the tiles no
row can reach); --recipes narrows it. --save-results dumps the raw measurements
to JSON so the bands can be re-cut offline with --results-json instead of
re-measuring.

This script only measures and emits the row file: use it with
ASTR_GEMM_TABLE=plan_table.txt to serve the rows without a rebuild, or paste
them into the compiled-in GENERATED block of plan_table.h by hand (pasting
maps the cta column onto TileClass; the rest is literal).
"""

from __future__ import annotations

import itertools
import json
import os
import re
import tempfile
import time
from pathlib import Path

import click
import torch

from astrai.extension import is_available
from astrai.extension.ops.gemm import quant_gemm

# Combo name -> (activation dtype, weight dtype). Scales: int8 operands
# require their dequant scale, fp8 accept one optionally (a [1] per-tensor
# scalar), bf16 rejects one. Matches the csrc dispatch (gemm.cu).
COMBOS: dict[str, tuple[torch.dtype, torch.dtype]] = {
    "w16a16": (torch.bfloat16, torch.bfloat16),
    "w8a16": (torch.bfloat16, torch.int8),
    "w8a16_f8e4m3": (torch.bfloat16, torch.float8_e4m3fn),
    "w8a16_f8e5m2": (torch.bfloat16, torch.float8_e5m2),
    "w8a8": (torch.int8, torch.int8),
    "f8a8_e4m3": (torch.float8_e4m3fn, torch.float8_e4m3fn),
    "f8a8_e5m2": (torch.float8_e5m2, torch.float8_e5m2),
}

# GemmPerfClass ids (gemm.cuh): W16A16 / W8A16 / W8A8 / F8A8.
PERF_CLASS: dict[str, int] = {
    "w16a16": 0,
    "w8a16": 1,
    "w8a16_f8e4m3": 1,
    "w8a16_f8e5m2": 1,
    "w8a8": 2,
    "f8a8_e4m3": 3,
    "f8a8_e5m2": 3,
}

# ---------------------------------------------------------------------------
# The candidate vocabulary, read from the launch ladders.
#
# A recipe is one tile: its CTA class, its ring depth and its k-tile depth,
# written as a row file's (cta, stages, kK) — which is exactly the dispatch
# key dispatch_tile matches on. The set of recipes is therefore the set of
# tiles the ladders carry, so it is parsed from policy.cuh instead of being
# maintained here. Names are the tile's own structural token
# (Tile_<M>x<N>x<kK>_W<wM>x<wN>_S<stages>[_Fast]), so a recipe name cannot
# describe a tile that is not there.
#
# One consequence worth reading twice: dispatch_tile matches (class, stages,
# kK) and takes the FIRST entry, so two tiles on the same key are one
# reachable recipe and one dead entry — the warp tiling is not part of the
# key. Those dead entries are reported (see UNREACHABLE) rather than measured,
# because a row can never select them whatever the shape.
# ---------------------------------------------------------------------------

# Operand widths per GemmPerfClass id: W16A16 / W8A16 / W8A8 / F8A8.
PERF_WIDTH: dict[int, tuple[int, int]] = {0: (2, 2), 1: (2, 1), 2: (1, 1), 3: (1, 1)}

POLICY_CUH = Path(__file__).resolve().parents[1] / "kernels" / "gemm" / "policy.cuh"
_TILE_RE = re.compile(r"(Tile_\d+x\d+x\d+_W\d+x\d+_S\d+(?:_Fast)?)")
_FACTS_RE = re.compile(r"Tile_(\d+)x(\d+)x(\d+)_W(\d+)x(\d+)_S(\d+)(_Fast)?")
# TileClass ordinals, policy.cuh enum order (what a row's cta column means).
_CLASS_OF = {(64, 64): 0, (128, 64): 1, (128, 128): 2, (128, 256): 3}
# The shared ladder every staging path carries: six geometries, one per class
# and ring depth. If the parse below cannot see these, it is broken.
_SHARED = (
    "Tile_64x64x64_W16x32_S2_Fast",
    "Tile_64x64x64_W16x32_S3_Fast",
    "Tile_128x64x64_W32x32_S2_Fast",
    "Tile_128x64x64_W32x32_S3_Fast",
    "Tile_128x128x64_W64x32_S2_Fast",
    "Tile_128x128x64_W64x32_S3_Fast",
)


def tile_facts(name: str) -> tuple[int, int, int]:
    """A tile's dispatch key: (cta class, stages, ring K)."""
    m, n, k, _wm, _wn, stages, _fast = _FACTS_RE.fullmatch(name).groups()
    try:
        return _CLASS_OF[(int(m), int(n))], int(stages), int(k)
    except KeyError as exc:  # a geometry TileClass does not name
        raise RuntimeError(
            f"{name}: CTA geometry not in the TileClass enum "
            f"(policy.cuh). Add the class there and its ordinal to _CLASS_OF."
        ) from exc


def _load_ladders() -> dict[str, tuple[str, ...]]:
    """Manifest membership per ladder, in manifest order — that order is the
    tie preference, because dispatch_tile takes the first key match."""
    text = POLICY_CUH.read_text()
    if "using TileManifest" not in text:
        raise RuntimeError(f"{POLICY_CUH}: no launch ladders found")
    declared = set(_TILE_RE.findall(text))

    def listed(name: str) -> list[str]:
        start = text.index(f"using {name}")
        end = text.index(";", start)
        return [t for t in _TILE_RE.findall(text[start:end]) if t in declared]

    cross = listed("TileManifestCross =")
    return {
        # manifest_for in policy.cuh picks between exactly these three.
        "TileManifestCross": tuple(cross),
        "TileManifest": tuple(
            cross + [t for t in listed("TileManifest =") if t not in cross]
        ),
        "TileManifestByte": tuple(
            cross + [t for t in listed("TileManifestByte =") if t not in cross]
        ),
    }


def _reachable(tiles: tuple[str, ...]) -> tuple[tuple[str, ...], list[str]]:
    """Split a ladder into what a row can select (the first tile per dispatch
    key) and what it cannot (a second tile on a key already taken)."""
    seen: dict[tuple[int, int, int], str] = {}
    live: list[str] = []
    dead: list[str] = []
    for tile in tiles:
        key = tile_facts(tile)
        if key in seen:
            dead.append(f"{tile} (same (class, stages, kK) as {seen[key]})")
        else:
            seen[key] = tile
            live.append(tile)
    return tuple(live), dead


LADDERS = _load_ladders()
REACHABLE: dict[str, tuple[str, ...]] = {}
UNREACHABLE: dict[str, list[str]] = {}
for _ladder, _tiles in LADDERS.items():
    REACHABLE[_ladder], UNREACHABLE[_ladder] = _reachable(_tiles)
# Every recipe the ladders carry, by name: cta class, ring depth, k-tile depth.
RECIPES: dict[str, tuple[int, int, int]] = {
    tile: tile_facts(tile) for tiles in REACHABLE.values() for tile in tiles
}
_missing = [t for t in _SHARED if t not in RECIPES]
if _missing:
    raise RuntimeError(
        f"vocabulary read from {POLICY_CUH} missed {_missing}: refusing to "
        f"sweep a partial candidate set (a silent omission is how a winning "
        f"tile goes unmeasured)."
    )


def ladder_for_widths(ba: int, bb: int) -> str:
    """Mirror of manifest_for (policy.cuh): 2-byte pairs get the congruent
    ladder, 1-byte pairs the byte one, a mixed or crosswise problem the shared
    six. The sweep is NT (crosswise 0), so mixed width is the Cross user here."""
    if ba == 2 and bb == 2:
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
    """A tile name without the Tile_ / _Fast dressing, for run headers."""
    return tile.removeprefix("Tile_").removesuffix("_Fast")


def reachability_report() -> list[str]:
    """One line per tile a row cannot reach, for the run header."""
    return [f"{ladder}: {dead}" for ladder, dead in UNREACHABLE.items() if dead]


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


_TAG_RE = re.compile(r"\[gemm-plan\] (table|degraded|forced recipe)\b")


def _last_tag(text: str) -> str:
    """The decision tag of the launch that just happened (gemm.cuh logs one
    decision line per plan_gemm call, then the launch line)."""
    tags = _TAG_RE.findall(text)
    if not tags:
        return "?"
    return "degraded (no table)" if tags[-1] == "degraded" else tags[-1]


def _planned_tag(run) -> str:
    """The decision tag of one launch, with the plan log on for that launch
    only: gemm.cuh re-reads ASTR_GEMM_PLAN per call, so the log can be toggled
    around it and its fprintf stays out of the timed loop. The tag is read from
    fd 2 — the log is C-level fprintf, not Python's stderr."""
    log = tempfile.TemporaryFile()
    saved = os.dup(2)
    os.environ["ASTR_GEMM_PLAN"] = "1"
    os.dup2(log.fileno(), 2)
    try:
        run()
        torch.cuda.synchronize()
    finally:
        os.environ.pop("ASTR_GEMM_PLAN", None)
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


def _candidate_row_files() -> dict[str, str]:
    """One synthetic open row per candidate — the recipe's (cta, stages, kK)
    on -1/-1 keys, so the row is the only plan source and plan_from_row gates
    it on the ring and the operand widths, exactly like an emitted row."""
    directory = tempfile.mkdtemp(prefix="gemm_recipe_rows_")
    files: dict[str, str] = {}
    for recipe, (cta, stages, kk) in RECIPES.items():
        path = os.path.join(directory, f"row_{recipe}.txt")
        with open(path, "w") as f:
            f.write(f"0 0 0 0 -1 -1 {cta} {stages} 0 {kk}\n")
        files[recipe] = path
    return files


def parse_positive_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


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

    # Candidates are one-row table files toggled per launch: the row
    # source re-reads that env per call, so all candidates at a shape
    # share its GPU clock/thermal state. The "model" candidate measures
    # the degraded band rows (the cost model is deleted from the C++):
    # it is the "this band has no table row" reference of the min-gain
    # mode, not a planner.
    row_files = _candidate_row_files()
    tags: dict[tuple[str, str], str] = {}

    device = torch.device(torch.cuda.current_device())
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
                        # "-" = AOT off: skip both override and builtin rows
                        # — the degraded-bands reference (the cost model is
                        # deleted from the C++).
                        os.environ["ASTR_GEMM_TABLE"] = "-"
                    else:
                        os.environ["ASTR_GEMM_TABLE"] = row_files[recipe]
                    expected = "degraded (no table)" if recipe == "model" else "table"
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
    os.environ.pop("ASTR_GEMM_TABLE", None)
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
        expected = "degraded (no table)" if point["recipe"] == "model" else "table"
        if "planned" in point and point["planned"] != expected:
            continue  # demoted candidate: those numbers are the fallback's
        _check_recipe(point["perf_class"], point["recipe"])
        key = (point["perf_class"], point["m"], point["n"])
        over = aggregate.setdefault(key, {})
        tflops = point["tflops"]
        over[point["recipe"]] = max(over.get(point["recipe"], 0.0), tflops)
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


@click.command()
@click.option(
    "--m-values",
    default="512,2048,4096",
    show_default=True,
    callback=lambda _c, _p, v: parse_positive_ints(v),
)
@click.option(
    "--shapes",
    "shape_values",
    multiple=True,
    help="NAME:N:K — the sweep's weight shapes (N, K); a small probe grid "
    "is used when omitted.",
)
@click.option(
    "--n-values",
    default=None,
    callback=lambda _c, _p, v: parse_positive_ints(v) if v else None,
    help="Explicit N grid (used when --shapes is empty).",
)
@click.option(
    "--k-values",
    default=None,
    callback=lambda _c, _p, v: parse_positive_ints(v) if v else None,
    help="Explicit K grid (used when --shapes is empty).",
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
    help="Plan-table row file (ASTR_GEMM_TABLE=/path/to/this).",
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
        click.echo("  model  ASTR_GEMM_TABLE=- (degraded bands), the reference")
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
        elif n_values and k_values:
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
        "# Generated by csrc/bench/gen_plan_table.py; tune the grid then "
        "re-run.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + "\n".join(rows) + ("\n" if rows else ""))
    click.echo(f"wrote {len(rows)} rows to {output}")
    click.echo(f"use: ASTR_GEMM_TABLE={output}")


if __name__ == "__main__":
    plan_table_command()
