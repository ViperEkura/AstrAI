"""Offline capture harness for planner candidates.

Scores a planner *rule* against saved measurements without launching a
kernel: for every measured point, the rule's pick is compared with the best
measured recipe, and the per-class geomean of that ratio is the capture.
This is the gate a model change must pass before it touches gemm.cuh — the
2026-09-13/14 lessons (tc_eff, wave_eff, thermal-ordered diff rows) were all
analytic terms shipped without one.

    python csrc/bench/model_capture.py /tmp/tuned_step0b.json
    python csrc/bench/model_capture.py /tmp/tuned_step0b.json --rule bytes

Rules are functions of (tile geometry, problem shape, device facts) only —
never of the measurement — so a rule that scores well here can be written in
C++ verbatim. `resident` mirrors policy.cuh's min_ctas_for_ring with the
ring the sweep computes, plus the load-thread cap; `wave_eff` is the same
fill fraction the sweep's wave terms use.

Measured points come from `tune_plan_table.py sweep --save-results`; those
numbers are phase-ordered (see the workspace AGENTS.md on thermal bias), so
use this harness for RULE selection and the interleaved A/B for shipping
decisions.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import click
import torch

_TUNE = Path(__file__).resolve().parent / "tune_plan_table.py"


def _load_tune():
    spec = importlib.util.spec_from_file_location("tune_plan_table", _TUNE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tpt = _load_tune()


def device_facts() -> dict:
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


def geometry(name: str) -> dict:
    m = tpt._FACTS_RE.fullmatch(name)
    bm, bn, kk, wm, wn, stages = m.groups()
    bm, bn, kk, wm, wn, stages = map(int, (bm, bn, kk, wm, wn, stages))
    # The 16-warp widening (warp_widened_t) doubles the load threads of a
    # small kk=64 tile on any pair with a 2-byte operand; a byte pair keeps
    # the 8-warp form. The rules below must see the widened shape or they
    # mis-price every small 2-byte candidate (that mistake cost the first
    # bytes-rule prototype its class-1 capture).
    threads = (bm // wm) * (bn // wn) * 32
    return {
        "bm": bm,
        "bn": bn,
        "kk": kk,
        "stages": stages,
        "threads": threads,
        "wm": wm,
        "wn": wn,
    }


def priced(name: str, m: int, n: int, k: int, ba: int, bb: int, dev: dict) -> dict:
    g = geometry(name)
    # widening: 2-byte operand, small CTA, kk=64 (policy.cuh warp_widened_t)
    widened = ba + bb >= 3 and g["bm"] == 64 and g["bn"] == 64 and g["kk"] == 64
    threads = 512 if widened else g["threads"]
    ring = (g["stages"] + 1) * g["kk"] * (g["bm"] * ba + g["bn"] * bb)
    # policy.cuh: min_ctas_for_ring is bytes <= 48KB ? 2 : 1 — residency is
    # smem-capped ONLY (no thread term). An earlier version of this file
    # added its own thread cap and simulated picks the kernel never makes,
    # which is how a rule that scored +2.6pp here regressed 8 cells
    # end-to-end. Fidelity is now checked against the binding (--check).
    resident = min(dev["smem"] // ring, 2 if ring <= 48 * 1024 else 1)
    # Feasibility is the ring's, not a floor on residency: the C++'s
    # plan_resident_ctas returns 0 (not 1) for a ring past the opt-in
    # ceiling, and a rule that prices it anyway simulates a launch the
    # binding never makes. The rule loop skips ok=false.
    ok = resident > 0 and ring <= dev["smem_max"]
    blocks = ((m + g["bm"] - 1) // g["bm"]) * ((n + g["bn"] - 1) // g["bn"])
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
        "kiters": (k + g["kk"] - 1) // g["kk"],
    }


def rules() -> dict:
    """Candidate orderings: (name, key fn) — larger key wins. Each key fn
    takes (features, problem) so a rule may consult the shape, like the
    model's own byte-pair floor does."""

    def shipped(p, q):
        # gemm.cuh ModelPlanner, mirrored term for term. Two staging forms
        # (the residency sign flips with staging — 2026-09-16):
        # - cp.async (q["tma"] false): zero-constant L20 form — raw-floor
        #   residency in the wave denominator, the k-tail priced whole.
        # - TMA: per_cta = max(operand + output + k-tile issue, mma arm),
        #   the issue pricing kK at 8 output-cell-bytes per k-iteration
        #   (byte pairs none — all-kK=64 ladder), the mma arm at 64 bytes
        #   per instruction; W_eff = waves*resident on two-byte pairs,
        #   ceil(blocks/sms) resident-blind on byte and mixed. Every pair
        #   ranks on the cost alone.
        byte = q["ba"] == 1 and q["bb"] == 1
        dev = device_facts()
        if not q.get("tma", True):
            operand = p["kiters"] * p["kk"] * (p["bm"] * q["ba"] + p["bn"] * q["bb"])
            mu = dev["smem"] // p["ring"]
            slots = dev["sms"] * mu
            waves = (p["blocks"] + slots - 1) // slots if slots else 1
            return (-(operand + 2 * p["fat"]) * waves,)
        mem = p["b2"] + 2 * p["fat"] + (0 if byte else 8.0 * p["fat"] * p["kiters"])
        mma = 64 * p["fat"] * q["k"] // (128 * (32 if byte else 16))
        if q["ba"] == 2 and q["bb"] == 2:
            weff = p["waves"] * p["resident"]
        else:
            weff = (p["blocks"] + dev["sms"] - 1) // dev["sms"]
        return (-max(mem, mma) * weff,)

    def cost_of(p, issue=0.0):
        # shipped cost plus, when issue > 0, a per-k-tile overhead every
        # block pays ceil(k/kk) times — the mainloop's own tile_count, tail
        # partial tile included (ring refill, barrier, mma issue): the one
        # term kk never had, which is why the model cannot price kk32 vs
        # kk64 (2026-09-16 grid diagnosis: 64x64x64_S3 beats kk32 at k3584).
        return (
            (p["b2"] + 2 * p["fat"] + issue * p["fat"] * p["kiters"])
            * p["waves"]
            * p["resident"]
        )

    def deepgemm(p, q):
        # sm90 heuristics.hpp get_layout_info, term for term: cycles =
        # max(L1, L2 bytes)/bandwidth / wave_eff. On this tile set every
        # bm >= 64, so wgmma_m=64 never rereads (max(wgmma_m, bm) == bm)
        # and the L1 side is a constant multiple of the L2 side — the rule
        # degenerates to ranking -(b2 + 2*fat)*waves*resident, i.e. cost
        # alone with the (resident, stages) prefix gone. Kept as the real
        # formula so a future bm<64 tile re-opens the L1 domain honestly.
        # NOTE the degeneracy above is DeepGEMM's (wgmma reads smem
        # directly); sm_120 has no wgmma — AstrAI's mma re-reads fragments
        # at K*bm*bn*(wa/wn + wb/wm), which on this vocabulary stays at
        # 0.4-0.6x of the L2 flow and never binds either (checked on the
        # 2026-09-16 grid, benchmark/sweep-grid-2026-09-16/README.md).
        dev = device_facts()
        l1_bw = 128.0 * dev["sms"]
        l2_bw = min(64.0 * dev["sms"], 8e6 / 1.3e3)
        eb = float(q["ba"])
        cd = p["fat"] * 2.0  # bf16 output, no accumulate
        blocks = p["blocks"]
        l2_c = (p["b2"] + cd) * blocks / l2_bw
        tc = q["k"] * (max(64, p["bm"]) + p["bn"]) * eb + cd
        l1_c = (p["b2"] + tc + cd) * blocks / l1_bw
        return (-max(l1_c, l2_c) / max(p["wave_eff"], 1e-9),)

    return {
        "model_exact": shipped,
        # the pre-floor ranking, to size the floor rule's contribution
        "resource": lambda p, q: (p["resident"], p["stages"]),
        # cost alone for every pair: what DeepGEMM's formula collapses to
        # here (see deepgemm above) — no resource prefix at all
        "cost_only": lambda p, q: (-cost_of(p),),
        # ---- 2026-09-16 full-grid winners (768-cell x 3 datasets) ----
        # two-byte pairs: pure cost + issue(8), the whole (resident,
        # stages) prefix dropped. 0.9155->0.9623 new grid, 0.9239->0.9664
        # and 0.8906->0.9854 on both 09-14 grids. The prefix's stages tier
        # was the S3-over-S3 confusion (485/768 measured winners are S2).
        "cost_kt8": lambda p, q: (-cost_of(p, issue=8.0),),
        # mixed pairs: keep resident, drop only the stages tier.
        # 0.9056->0.9504 new grid, 0.9387->0.9588 on the 09-14 grid.
        "res_kt8": lambda p, q: (p["resident"], -cost_of(p, issue=8.0)),
        # byte pairs: issue(2) re-weights fat only (their ladder is all
        # kk=64); +0.5-0.8pp new grid but a wash on 09-14 — kept for the
        # record, not a change worth shipping over cost_only.
        "cost_kt2": lambda p, q: (-cost_of(p, issue=2.0),),
        # DeepGEMM sm90 cycles, the honest port
        "deepgemm": deepgemm,
        # ---- documented losers (2026-09-16 grid), do not retry ----
        # flipping the stages tier alone (s2_first) loses the resident-tied
        # cells the prefix was right about: 0.9155 -> 0.8852. Tied with
        # no_stages on this menu.
        "s2_first": lambda p, q: (p["resident"], -p["stages"], -cost_of(p)),
        # fattest tile that still fills, fill-first when nothing fills
        "fat_fill": lambda p, q: (
            p["fat"] if p["wave_eff"] >= 0.5 else 0.0,
            p["wave_eff"],
        ),
        # fill-weighted harmonic tile size: bigger tiles amortise traffic,
        # fill decides how much of the tile's work overlaps
        "fill_harm": lambda p, q: p["wave_eff"] * p["fat"] / max(1, p["bm"] + p["bn"]),
        # traffic per unit fill: minimise bytes for the fill achieved
        "traffic_fill": lambda p, q: p["wave_eff"] / max(1, p["b2"]),
        # resident weighted by how compute-rich the tile's ring is
        # C++ mirrors this exactly: the product, then ring depth
        "density": lambda p, q: (p["resident"] * p["kk"], p["stages"]),
        # traffic alone, fill as a tie-break (no constants at all)
        "traffic": lambda p, q: (p["wave_eff"], -p["b2"]),
    }


@click.command()
@click.argument("results_json", type=click.Path(exists=True, path_type=Path))
@click.option("--rule", default=None, help="Only this rule (default: all).")
@click.option(
    "--class-filter", default=None, type=int, help="Only this GemmPerfClass id."
)
@click.option(
    "--check",
    is_flag=True,
    default=False,
    help="Fidelity: compare each rule's pick with ops.gemm.probe on "
    "the measured points (the model rule must match 100%).",
)
@click.option(
    "--staging",
    type=click.Choice(["tma", "cpasync"]),
    default="tma",
    show_default=True,
    help="The staging SWITCH (set_staging): the priced form still follows "
    "the device — TMA needs sm_90+, so on an sm_89 part (L20/4090) the "
    "cp.async cost runs whatever this says. '*_cpasync' datasets from "
    "sm_90+ boxes want --staging cpasync; the --check probes run with "
    "set_staging(tma=False) to match.",
)
def main(results_json, rule, class_filter, check, staging):
    dev = device_facts()
    # The C++ predicate: TMA needs sm_90+, so the switch alone does not
    # make the binding stage via TMA — an sm_89 part (L20/4090) prices
    # the cp.async form whatever the switch says.
    use_tma = staging == "tma" and dev["cc"] >= 90
    by_point: dict[tuple, dict[str, float]] = defaultdict(dict)
    for point in json.loads(results_json.read_text()):
        key = (point["combo"], point["n"], point["k"], point["m"])
        # datasets saved before 2026-09-16 spell names with a `_Fast` suffix
        by_point[key][point["recipe"].removesuffix("_Fast")] = point["tflops"]

    table = rules()
    if rule:
        table = {rule: table[rule]}
    names = sorted(table)

    check_seen: dict[str, list[str]] = defaultdict(list)
    captures: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (combo, n, k, m), over in by_point.items():
        cands = [r for r in over if r != "model"]
        if not cands:
            continue
        perf_class = tpt.PERF_CLASS[combo]
        if class_filter is not None and perf_class != class_filter:
            continue
        best = max(over[r] for r in cands)
        ba, bb = tpt.BYTES[combo]
        ctx = {"m": m, "n": n, "k": k, "ba": ba, "bb": bb, "tma": use_tma}
        feats = {r: priced(r, m, n, k, ba, bb, dev) for r in cands}
        # order_for mirrors dispatch_tile's first-match tie-break
        order = tpt.order_for(perf_class)[:-1]
        for name in names:
            key = table[name]
            pick, pick_key = None, None
            for r in order:
                # A candidate the device cannot launch is not a candidate
                # (C++'s price() gate): pricing it would simulate a pick the
                # binding never makes.
                if r not in feats or not feats[r]["ok"]:
                    continue
                v = key(feats[r], ctx)
                if pick is None or v > pick_key:
                    pick, pick_key = r, v
            if pick is None:
                continue
            captures[(name, perf_class)].append(over[pick] / best)
        captures[("model(measured)", perf_class)].append(over.get("model", 0.0) / best)

    if check:
        from astrai.extension import ops, plan

        ops.gemm.set_table("")
        plan.configure(rows="", tier="injected")
        ops.gemm.set_planner("model")
        if staging == "cpasync":
            ops.gemm.set_staging(tma=False)  # price what this dataset ran
        seen: set[tuple] = set()
        for combo, n, k, m in sorted(by_point):
            perf_class = tpt.PERF_CLASS[combo]
            if class_filter is not None and perf_class != class_filter:
                continue
            if (combo, n, k, m) in seen:
                continue
            seen.add((combo, n, k, m))
            act, weight = tpt.COMBOS[combo]
            info = ops.gemm.probe(m, n, k, act, weight)
            real = (info["cta"], info["stages"], info["kk"])
            ba, bb = tpt.BYTES[combo]
            ctx = {"m": m, "n": n, "k": k, "ba": ba, "bb": bb, "tma": use_tma}
            # The binding chooses among the whole ladder, so the fidelity
            # comparison must too. Restricting to the recipes this dataset
            # measured reported every cell where the model picks a tile the
            # dataset predates as a mismatch (meas_w16a16 has no tall entry).
            cands = list(tpt.REACHABLE[tpt.ladder_for_widths(ba, bb)])
            feats = {r: priced(r, m, n, k, ba, bb, dev) for r in cands}
            order = tpt.order_for(perf_class)[:-1]
            for name in names:
                key = table[name]
                pick, pick_key = None, None
                for r in order:
                    if r not in feats or not feats[r]["ok"]:
                        continue
                    v = key(feats[r], ctx)
                    if pick is None or v > pick_key:
                        pick, pick_key = r, v
                if pick is None:
                    continue
                got = tpt.RECIPES[pick]
                if got != real and len(check_seen[name]) < 5:
                    check_seen[name].append(
                        f"combo={combo} m{m} n{n} k{k}: harness {got} vs binding {real}"
                    )
        ops.gemm.set_planner("")
        if staging == "cpasync":
            ops.gemm.set_staging(tma=True)
        click.echo("fidelity vs the binding (mismatches, first 5 per rule):")
        for name in names + ["model(measured)"]:
            bad = check_seen.get(name, [])
            status = (
                "n/a"
                if name == "model(measured)"
                else ("MATCH" if not bad else f"{len(bad)}+")
            )
            click.echo(f"  {name:16s} {status}")
            for line in bad:
                click.echo(f"      {line}")
        return

    classes = sorted({c for _n, c in captures})
    header = "rule".ljust(16) + "".join(f"class{c:>8d}" for c in classes)
    click.echo(header)
    for name in names + ["model(measured)"]:
        cells = []
        for c in classes:
            vals = captures.get((name, c), [])
            if not vals:
                cells.append("       -")
                continue
            gm = math.exp(sum(math.log(max(v, 1e-3)) for v in vals) / len(vals))
            cells.append(f"{gm:8.3f}")
        click.echo(name.ljust(16) + "".join(cells))
    click.echo(
        "\npoints per class: "
        + ", ".join(
            f"class{c}={len(captures[('model(measured)', c)])}" for c in classes
        )
    )


if __name__ == "__main__":
    main()
