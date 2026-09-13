"""One-shot per-device plan-table baseline: sweep -> holdout -> install.

The manual tuning loop this replaces was three commands whose results
lived in scratch files: gen_plan_table.py --full-coverage (sweep every
candidate recipe the ladders carry, emit band-merged rows),
validate_plan_table.py (interleaved holdout against the builtin table),
then a hand copy into ASTR_GEMM_TABLE or plan_table.h. A plan-table band
bound is device arithmetic (SM count, L2 bytes, smem), so every new part
needs the loop again — this script runs it end to end and files the
result under the device signature the runtime autotuner keys on
(astrai/extension/gemm_autotune.py), where it is picked up on the next
process start.

    python csrc/bench/tune_baseline.py                  # defaults below
    python csrc/bench/tune_baseline.py --shapes up_gate:14336:4096
    python csrc/bench/tune_baseline.py --skip-validate  # sweep only

Each stage runs as a subprocess (the sweep and the validator both toggle
ASTR_GEMM_TABLE per launch inside their own process; sharing one process
with them would mean fighting over that env). The candidate rows are
emitted to a temp file, holdout-validated against the compiled-in table,
and only then installed; a >=2% per-shape regression fails the run and
leaves the candidate next to the install path for inspection. Whether the
rows should ALSO be pasted into plan_table.h's GENERATED block (making
them the shipped default for this part) stays a human decision — the
holdout grid here is a smoke test, not the evidence a release wants.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import click
import torch

from astrai.extension.gemm_autotune import device_signature

REPO = Path(__file__).resolve().parents[2]
GEN = Path(__file__).resolve().parent / "gen_plan_table.py"
VALIDATE = Path(__file__).resolve().parent / "validate_plan_table.py"

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
    """The autotuner's facts dict, from the same attributes DeviceFacts
    queries — the signature must not drift from gemm_autotune's key."""
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "sms": props.multi_processor_count,
        # torch exposes no per-block smem opt-in query; the planner constant
        # is close enough for a cache KEY, and the sweep itself measures on
        # the real device either way.
        "smem_max": 101376 if props.major >= 8 else 48 * 1024,
        "smem_per_sm": props.shared_memory_per_multiprocessor,
        "regs_per_sm": props.regs_per_multiprocessor,
        "l2_bytes": props.L2_cache_size,
        "cc": props.major * 10 + props.minor,
    }


def _run(cmd: list[str]) -> str:
    env = {k: v for k, v in os.environ.items() if k != "ASTR_GEMM_TABLE"}
    click.echo(f"+ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True, env=env, cwd=REPO
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
        f"# generated {date.today():%Y-%m-%d} by csrc/bench/tune_baseline.py\n"
        f"# argv: {' '.join(argv)}\n"
    )
    dest.write_text(header + rows_path.read_text())
    return dest


@click.command()
@click.option("--m-values", default=None, help="Passthrough to gen_plan_table.")
@click.option(
    "--shapes",
    "shape_values",
    multiple=True,
    help="NAME:N:K passthrough to gen_plan_table.",
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
@click.option(
    "--combos", default=None, help="Comma list passthrough to gen_plan_table."
)
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
    help="Install dir (default: the autotuner cache, ASTR_GEMM_TUNE_DIR or"
    " ~/.astrai/cache/gemm_plans).",
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
        gen_cmd = [sys.executable, GEN, "--full-coverage", "--output", candidate]
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
                VALIDATE,
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
        "serve without rebuild: ASTR_GEMM_TABLE="
        f"{dest}\n(runtime autotuner picks it up as its cache automatically)"
    )


if __name__ == "__main__":
    main()
