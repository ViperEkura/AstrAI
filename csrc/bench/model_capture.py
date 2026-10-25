"""Replay the retained GEMM dispatch models against saved measurements.

Retained candidates: geom_cta (smallest observed worst regression), ncu_spill
(sector/local-memory risk estimate), geom_barrier (best observed aggregate),
and model_exact (the existing C++ model baseline). These rankings are from
36 RTX 5090 ABBA points, not a cross-device performance guarantee.

    python csrc/bench/model_capture.py results.json --staging cpasync
    python csrc/bench/model_capture.py results.json --rule geom_cta
    python csrc/bench/model_capture.py results.json --rule model_exact --check

Selection uses geometry and compiled CUDA metadata, never saved timings.
Timings only evaluate the selected recipe. Replay considers measured recipes;
--check considers the full manifest for either C++ counterpart.
geom_cta mirrors C++ heuristic; model_exact mirrors C++ model. Inputs are
contiguous NT, BF16 output, with MX disabled, matching the sweep harness.
No kernels are launched or timed by this script. Use same-process ABBA for
performance decisions; phase-ordered sweep capture is only a diagnostic.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict

import click
import torch

from astrai.extension import kernel, plan

_TUNE = Path(__file__).resolve().parent / "tune_plan_table.py"


def _load_tune():
    spec = importlib.util.spec_from_file_location("tune_plan_table", _TUNE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tpt = _load_tune()


def device_facts() -> Dict:
    """The queried device geometry, the same fields the C++ prices against
    (common/device.cuh DeviceFacts). smem is the PER-SM figure and smem_max
    the PER-BLOCK opt-in ceiling: residency prices from the former, ring
    feasibility from the latter. Both are read, never written down — a
    hard-coded 100KB made every rule below mis-price on a part with a
    different smem/SM (228KB on Hopper/Blackwell datacenter)."""
    props = torch.cuda.get_device_properties(0)
    return {
        "sms": props.multi_processor_count,
        "threads": props.max_threads_per_multi_processor,
        "smem": props.shared_memory_per_multiprocessor,
        "smem_max": props.shared_memory_per_block_optin,
        "regs": props.regs_per_multiprocessor,
        "cc": props.major * 10 + props.minor,
    }


def geometry(name: str) -> Dict:
    m = tpt._FACTS_RE.fullmatch(name)
    bm, bn, k_tile, wm, wn, k_stages = m.groups()
    bm, bn, k_tile, wm, wn, k_stages = map(int, (bm, bn, k_tile, wm, wn, k_stages))
    # The 16-warp widening (warp_widened_t) doubles the load threads of a
    # small k_tile=64 tile on any pair with a 2-byte operand; a byte pair keeps
    # the 8-warp form. The rules below must see the widened shape or they
    # mis-price every small 2-byte candidate (that mistake cost the first
    # bytes-rule prototype its class-1 capture).
    threads = (bm // wm) * (bn // wn) * 32
    return {
        "bm": bm,
        "bn": bn,
        "k_tile": k_tile,
        "k_stages": k_stages,
        "threads": threads,
        "wm": wm,
        "wn": wn,
    }


def priced(
    name: str,
    m: int,
    n: int,
    k: int,
    ba: int,
    bb: int,
    dev: Dict,
    tma: bool,
    batch: int = 1,
) -> Dict:
    g = geometry(name)
    # widening: 2-byte operand, small CTA, k_tile=64 (policy/gemm/manifest.h warp_widened_t)
    widened = ba + bb >= 3 and g["bm"] == 64 and g["bn"] == 64 and g["k_tile"] == 64
    threads = 512 if widened else g["threads"]
    ring = (g["k_stages"] + 1) * g["k_tile"] * (g["bm"] * ba + g["bn"] * bb)
    # policy/gemm/geometry.h: min_ctas_for_ring is bytes <= 48KB ? 2 : 1 — residency is
    # charged the TMA pad/barriers and otherwise smem-capped ONLY (no thread term). An earlier version of this file
    # added its own thread cap and simulated picks the kernel never makes,
    # which is how a rule that scored +2.6pp here regressed 8 cells
    # end-to-end. Fidelity is now checked against the binding (--check).
    staged = ring + 1024 + 16 * (g["k_stages"] + 1) if tma else ring
    resident = min(dev["smem"] // staged, 2 if ring <= 48 * 1024 else 1)
    ok = resident > 0 and staged <= dev["smem_max"]
    blocks = batch * ((m + g["bm"] - 1) // g["bm"]) * ((n + g["bn"] - 1) // g["bn"])
    waves = max(
        1,
        (blocks + dev["sms"] * max(resident, 1) - 1) // (dev["sms"] * max(resident, 1)),
    )
    wave_eff = blocks / (waves * dev["sms"] * max(resident, 1))
    return {
        **g,
        "ring": ring,
        "resident": resident,
        "blocks": blocks,
        "ok": ok,
        "waves": waves,
        "wave_eff": wave_eff,
        "widened": widened,
        "threads": threads,
        # the dispatch key's first column, for orderings that prefer a class
        "cta": tpt.tile_facts(name)[0],
        # per-block operand bytes through L2 (the tile's own traffic; the
        # problem's total is this times blocks)
        "b2": k * (g["bm"] * ba + g["bn"] * bb),
        "fat": g["bm"] * g["bn"],
        "kiters": (k + g["k_tile"] - 1) // g["k_tile"],
    }


def attach_resources(ctx, combo, m, n, k, batch, tma):
    da, db = tpt.COMBOS[combo]
    with plan.override(planner="heuristic", tma=tma, mx=False):
        info = kernel.gemm.probe(m, n, k, da, db, batch=batch)
    ctx["tma"] = info["tma"]
    ctx["batch"] = batch
    ctx["resources"] = {tuple(row[:3]): row for row in info["resources"]}


def rules() -> Dict:
    """Candidate orderings: (name, key fn) — larger key wins. Each key fn
    takes (features, problem) so a rule may consult the shape, like the
    model's own byte-pair floor does."""

    def shipped(p, q):
        # gemm/plan/planner.cpp model_plan, mirrored term for term. Two staging forms
        # (the residency sign flips with staging — 2026-09-16):
        # - cp.async (q["tma"] false): zero-constant L20 form — raw-floor
        #   residency in the wave denominator, the k-tail priced whole.
        # - TMA: per_cta = max(operand + output + k-tile issue, mma arm),
        #   the issue pricing kK at 8 output-cell-bytes per k-iteration
        #   (byte pairs none), the mma arm at 64 bytes
        #   per instruction; W_eff = waves*resident on two-byte pairs,
        #   ceil(blocks/sms) resident-blind on byte and mixed. Every pair
        #   ranks on the cost alone.
        byte = q["ba"] == 1 and q["bb"] == 1
        dev = device_facts()
        if not q.get("tma", True):
            operand = (
                p["kiters"] * p["k_tile"] * (p["bm"] * q["ba"] + p["bn"] * q["bb"])
            )
            mu = dev["smem"] // p["ring"]
            slots = dev["sms"] * mu
            waves = (p["blocks"] + slots - 1) // slots if slots else 1
            return (-(operand + 2 * p["fat"]) * waves,)
        mem = p["b2"] + 2 * p["fat"] + (0 if byte else 8.0 * p["fat"] * p["kiters"])
        mma_k = 32 if byte else 16
        mma = (
            64 * p["kiters"] * (p["k_tile"] // mma_k) * (p["bm"] // 16) * (p["bn"] // 8)
        )
        if q["ba"] == 2 and q["bb"] == 2:
            weff = p["waves"] * p["resident"]
        else:
            weff = (p["blocks"] + dev["sms"] - 1) // dev["sms"]
        return (-max(mem, mma) * weff,)

    def work(p, q):
        row = q["resources"].get((p["cta"], p["k_stages"], p["k_tile"]))
        if row is None or row[9] <= 0:
            return None
        _, _, _, bm, bn, k_tile, wm, wn, threads, resident, _, local = row
        ba, bb = q["ba"], q["bb"]
        blocks = q["batch"] * ((q["m"] + bm - 1) // bm) * ((q["n"] + bn - 1) // bn)
        steps = (q["k"] + k_tile - 1) // k_tile
        warps = threads / 32.0
        mma_k = 32 if ba == bb == 1 else 16
        copy = k_tile * (bm * ba + bn * bb) / 512.0
        mma = (k_tile / mma_k) * (bm / 16) * (bn / 8) / warps
        frag = k_tile * (wm * ba + wn * bb) / 512.0
        conv = k_tile * (wm * (ba < bb) + wn * (bb < ba)) / 64.0
        return dict(
            bm=bm,
            bn=bn,
            k_tile=k_tile,
            threads=threads,
            resident=resident,
            local=local,
            blocks=blocks,
            steps=steps,
            warps=warps,
            copy=copy,
            mma=mma,
            frag=frag,
            conv=conv,
            output=2 * bm * bn / 512.0,
        )

    def geometric(p, q, cta_correction):
        v = work(p, q)
        if v is None:
            return (-math.inf,)
        sms = q["dev"]["sms"]
        resident = min(v["resident"], math.ceil(v["blocks"] / sms))
        waves = math.ceil(v["blocks"] / (sms * resident))
        steps, warps = v["steps"], v["warps"]
        fragment = v["frag"] + v["conv"]
        issue = 2.0 if q["tma"] else v["copy"]
        lw, lr, ll, lp = map(math.log, (waves, resident, steps, warps))
        a, b = ll + math.log(v["copy"]), math.log(v["output"])
        log_memory = max(a, b) + math.log1p(math.exp(-abs(a - b)))
        memory_residency = 0.0 if cta_correction and not q["tma"] else lr
        log_costs = [
            lw + memory_residency + log_memory,
            lw + lr + ll + lp + math.log(v["mma"]),
            lw + ll + math.log(v["mma"] + v["frag"] + v["conv"]),
            lw + lr + ll + lp + math.log(fragment),
            lw + lr + ll + math.log(2 * warps + issue),
        ]
        # Drop common normalization and the monotone fifth root. Factorized
        # logs avoid constructing large arm products; no exp(score) is needed.
        return (-sum(log_costs),)

    def ncu_spill(p, q):
        v = work(p, q)
        if v is None:
            return (-math.inf,)
        resident = v["resident"]
        waves = math.ceil(v["blocks"] / (q["dev"]["sms"] * resident))
        ba, bb = q["ba"], q["bb"]
        ca = max(1.0, v["bm"] * v["k_tile"] * ba / (16 * v["threads"]))
        cb = max(1.0, v["bn"] * v["k_tile"] * bb / (16 * v["threads"]))
        transfer = (
            v["k_tile"]
            * (
                v["bm"] * ba * (1 if q["tma"] else min(ca, 2))
                + v["bn"] * bb * (1 if q["tma"] else min(cb, 2))
            )
            / 512.0
        )
        # Preserve the tested hypothesis: one local-footprint read/write per
        # K tile. localSizeBytes is NOT a dynamic spill count; this is a risk
        # proxy, and the sector factor is approximate for unaligned K/tails.
        transfer += 2 * v["local"] * v["threads"] / 512.0
        dep = v["mma"] + v["frag"] + v["conv"]
        cost = (
            waves
            * resident
            * (transfer + dep + (v["steps"] - 1) * max(transfer, dep) + v["output"])
        )
        return (-cost,)

    return {
        "model_exact": shipped,
        "geom_cta": lambda p, q: geometric(p, q, True),
        "ncu_spill": ncu_spill,
        "geom_barrier": lambda p, q: geometric(p, q, False),
    }


def choose(rule, feats, ctx, order):
    """Keep manifest order for equal scores; exclude infeasible recipes."""
    pick, best = None, None
    for recipe in order:
        if recipe not in feats or not feats[recipe]["ok"]:
            continue
        score = rule(feats[recipe], ctx)
        if not math.isfinite(score[0]):
            continue
        if pick is None or score > best:
            pick, best = recipe, score
    return pick


@click.command()
@click.argument("results_json", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--rule",
    type=click.Choice(list(rules())),
    default=None,
    help="Only this rule (default: all four; --check: model_exact).",
)
@click.option(
    "--class-filter", default=None, type=int, help="Only this GemmPerfClass id."
)
@click.option(
    "--check",
    is_flag=True,
    help="Verify model_exact or geom_cta against C++; fail on mismatches.",
)
@click.option(
    "--staging",
    type=click.Choice(["tma", "cpasync"]),
    default="tma",
    show_default=True,
    help="Staging switch; actual TMA eligibility follows device and strides.",
)
def main(results_json, rule, class_filter, check, staging):
    if check and rule not in (None, "model_exact", "geom_cta"):
        raise click.BadParameter(
            "--check supports model_exact and geom_cta, which have C++ counterparts.",
            param_hint="--rule",
        )
    table = rules()
    if check and rule is None:
        rule = "model_exact"
    names = [rule] if rule else list(table)
    dev = device_facts()
    use_tma = staging == "tma" and dev["cc"] >= 90
    by_point = defaultdict(dict)
    for point in json.loads(results_json.read_text()):
        key = (
            point["combo"],
            point["n"],
            point["k"],
            point["m"],
            point.get("batch", 1),
        )
        by_point[key][point["recipe"].removesuffix("_Fast")] = point["tflops"]

    captures = defaultdict(list)
    mismatches = []
    checked = 0
    for (combo, n, k, m, batch), over in sorted(by_point.items()):
        perf_class = tpt.PERF_CLASS[combo]
        if class_filter is not None and perf_class != class_filter:
            continue
        measured = [r for r in over if r != "model"]
        if not measured:
            continue
        ba, bb = tpt.BYTES[combo]
        point_tma = use_tma and (k * ba) % 16 == 0 and (k * bb) % 16 == 0
        ctx = dict(m=m, n=n, k=k, batch=batch, ba=ba, bb=bb, tma=point_tma, dev=dev)
        if any(name != "model_exact" for name in names):
            attach_resources(ctx, combo, m, n, k, batch, point_tma)
        cands = (
            list(tpt.REACHABLE[tpt.ladder_for_widths(ba, bb)]) if check else measured
        )
        feats = {r: priced(r, m, n, k, ba, bb, dev, ctx["tma"], batch) for r in cands}
        order = tpt.order_for(perf_class)[:-1]
        picks = {name: choose(table[name], feats, ctx, order) for name in names}
        if check:
            da, db = tpt.COMBOS[combo]
            with plan.override(
                planner="heuristic" if rule == "geom_cta" else "model",
                tma=use_tma,
                mx=False,
            ):
                info = kernel.gemm.probe(m, n, k, da, db, batch=batch)
            real = (info["cta"], info["k_stages"], info["k_tile"])
            got = tpt.RECIPES[picks[rule]] if picks[rule] else None
            checked += 1
            if got != real:
                mismatches.append(
                    f"{combo} m{m} n{n} k{k} batch{batch}: {got} vs {real}"
                )
            continue
        best = max(over[r] for r in measured)
        if best <= 0:
            raise click.ClickException(f"No positive throughput at {combo} {m}x{n}x{k}")
        for name, pick in picks.items():
            if pick is not None:
                captures[(name, perf_class)].append(over[pick] / best)
        if "model" in over:
            captures[("model(measured)", perf_class)].append(over["model"] / best)

    if check:
        if not checked:
            raise click.ClickException("No eligible points to check.")
        if mismatches:
            raise click.ClickException(
                f"{rule}: {len(mismatches)}/{checked} mismatches\n"
                + "\n".join(mismatches[:5])
            )
        click.echo(f"{rule}: MATCH ({checked} points)")
        return

    classes = sorted({c for _, c in captures})
    click.echo("rule".ljust(18) + "".join(f"class{c:>8d}" for c in classes))
    for name in names + ["model(measured)"]:
        cells = []
        counts = []
        for c in classes:
            vals = captures.get((name, c), [])
            counts.append(str(len(vals)))
            if not vals:
                cells.append("       -")
            elif any(v <= 0 for v in vals):
                cells.append(f"{0.0:8.3f}")
            else:
                gm = math.exp(sum(map(math.log, vals)) / len(vals))
                cells.append(f"{gm:8.3f}")
        click.echo(name.ljust(18) + "".join(cells) + "   n=" + "/".join(counts))


if __name__ == "__main__":
    main()
