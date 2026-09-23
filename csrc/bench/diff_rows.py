"""Emit plan rows as the measured DIFF of the model, in interleaved A/B.

Why this exists instead of `tune_plan_table.py sweep --min-gain`: the sweep
measures its candidates in order and runs the `model` reference LAST at
every point — the hottest clock state. On heavy shapes that underprices the
model by 10-20%, so a diff built from sweep numbers overclaims, and a
phase-ordered validate inherits the same bias (measured 2026-09-14: five
holdout regressions 2-7% that interleaved re-measurement turned into ties or
wins). A row may only be claimed from an interleaved comparison.

Flow: a sweep supplies each point's same-phase WINNER (recipe candidates
compared with each other in one phase is fine — that is what the sweep grid
is for); this script then compares that winner against the model's own
dispatch with alternating arms, and keeps the point only when the winner is
>= --min-gain faster. Points the model's own structural rules own (the
byte-pair m<=8 floor, see ModelPlanner in gemm.cuh) are neutralised to ties
so no row can override a rule with noise.

    python csrc/bench/diff_rows.py --sweep-json /tmp/sweep.json \
        --shapes square:1536:1536 --shapes qkv:6144:1536 \
        -m 1,8,16,32,64,128,256,512,1024,2048,4096 \
        --combos w16a16,w8a16,w8a8 --output /tmp/diff.rows
    # or the product grid instead of named shapes (START:END:STEP, end incl.):
    python csrc/bench/diff_rows.py --sweep-json /tmp/sweep.json \
        -m 256:4096:256 -n 256:4096:256 -k 512:3584:1536 --output /tmp/diff.rows
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import click
import torch

from astrai.extension import ops
from astrai.extension.ops.gemm import quant_gemm

_TUNE = Path(__file__).resolve().parent / "tune_plan_table.py"


def _load_tune():
    spec = importlib.util.spec_from_file_location("tune_plan_table", _TUNE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tpt = _load_tune()

# The byte-pair floor the model owns (ModelPlanner in gemm.cuh): m <= 8 on a
# 1-byte x 1-byte pair is bandwidth-floor-bound, every candidate measured
# identical, so a row there would only override the rule with noise.
FLOOR_MMAX = 8


def scale_for(dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    if dtype == torch.bfloat16:
        return None
    return torch.tensor([0.1], dtype=torch.float32, device=device)


@click.command()
@click.option("--sweep-json", required=True, type=click.Path(path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option(
    "--emit",
    type=click.Choice(("rows", "cpp")),
    default="rows",
    help="cpp emits std::array initializers for the GENERATED block.",
)
@click.option("--shapes", "shape_values", multiple=True, help="NAME:N:K")
@click.option(
    "-m",
    "--m-values",
    default="1,8,16,32,64,128,256,512,1024,2048,4096",
    callback=lambda _c, _p, v: tpt.parse_positive_ints(v),
    help="M grid: START:END:STEP (end inclusive) or a comma list.",
)
@click.option(
    "-n",
    "--n-values",
    default=None,
    callback=lambda _c, _p, v: tpt.parse_positive_ints(v) if v else None,
    help="N grid; with -k, the product grid instead of --shapes.",
)
@click.option(
    "-k",
    "--k-values",
    default=None,
    callback=lambda _c, _p, v: tpt.parse_positive_ints(v) if v else None,
    help="K grid; with -n, the product grid instead of --shapes.",
)
@click.option("--combos", default=",".join(tpt.COMBOS))
@click.option(
    "--min-gain",
    type=click.FloatRange(min=0.0),
    default=2.0,
    help="Relative gain (%) the winner must show over the model.",
)
@click.option("--warmup", type=click.IntRange(min=1), default=4)
@click.option("--iterations", type=click.IntRange(min=1), default=20)
@click.option("--trials", type=click.IntRange(min=1), default=4)
def main(
    sweep_json,
    output,
    emit,
    shape_values,
    m_values,
    n_values,
    k_values,
    combos,
    min_gain,
    warmup,
    iterations,
    trials,
):
    winners: dict[tuple[str, int, int, int], tuple[str, float]] = {}
    for point in json.loads(sweep_json.read_text()):
        if point["recipe"] == "model":
            continue
        key = (point["combo"], point["n"], point["k"], point["m"])
        # datasets saved before 2026-09-16 spell names with a `_Fast` suffix
        recipe = point["recipe"].removesuffix("_Fast")
        if key not in winners or point["tflops"] > winners[key][1]:
            winners[key] = (recipe, point["tflops"])

    combos = tuple(c for c in combos.split(",") if c)
    shapes = [tpt.parse_shape(v) for v in shape_values]
    if n_values or k_values:
        if not (n_values and k_values):
            raise click.BadParameter(
                "-n and -k go together — the grid is their product"
            )
        shapes += [(f"n{n}k{k}", n, k) for n in n_values for k in k_values]
    if not shapes:
        raise click.BadParameter("give --shapes NAME:N:K and/or the -n/-k grid")
    device = torch.device(torch.cuda.current_device())
    torch.manual_seed(0)

    results: list[dict] = []
    for combo in combos:
        act_dtype, weight_dtype = tpt.COMBOS[combo]
        a_scale = scale_for(act_dtype, device)
        b_scale = scale_for(weight_dtype, device)
        for _name, n, k in shapes:
            weight = (torch.rand(n, k, device=device) * 0.2 - 0.1).to(weight_dtype)
            for m in m_values:
                key = (combo, n, k, m)
                if key not in winners:
                    continue
                recipe = winners[key][0]
                acts = (torch.rand(m, k, device=device) * 0.2 - 0.1).to(act_dtype)
                row = tpt._candidate_rows()[recipe]

                def run(acts=acts, weight=weight, a_scale=a_scale, b_scale=b_scale):
                    return quant_gemm(acts, weight, a_scale, b_scale)

                best = {"model": float("inf"), recipe: float("inf")}
                for trial in range(trials):
                    order = ("model", recipe) if trial % 2 == 0 else (recipe, "model")
                    for arm in order:
                        ops.gemm.set_table("" if arm == "model" else row)
                        for _ in range(warmup):
                            run()
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(iterations):
                            run()
                        torch.cuda.synchronize()
                        best[arm] = min(
                            best[arm], (time.perf_counter() - start) / iterations
                        )
                flops = 2.0 * m * n * k
                ba, bb = tpt.BYTES[combo]
                floor = ba == 1 and bb == 1 and m <= FLOOR_MMAX
                model_tf = flops / best["model"]
                winner_tf = model_tf if floor else flops / best[recipe]
                for arm, tflops, planned in (
                    (recipe, winner_tf, "override"),
                    ("model", model_tf, "model"),
                ):
                    results.append(
                        {
                            "combo": combo,
                            "perf_class": tpt.PERF_CLASS[combo],
                            "m": m,
                            "n": n,
                            "k": k,
                            "batch": 1,
                            "recipe": arm,
                            "planned": planned,
                            "ms": best["model"] if arm == "model" else best[recipe],
                            "tflops": tflops,
                        }
                    )
                gain = best["model"] / best[recipe]
                flag = (
                    " floor"
                    if floor
                    else (" KEEP" if gain >= 1 + min_gain / 100 else "")
                )
                print(
                    f"{combo:13s} {_name:10s} m{m:5d} {recipe[5:]:26s} "
                    f"model {model_tf:7.1f} winner {flops / best[recipe]:7.1f} "
                    f"x{gain:.3f}{flag}",
                    flush=True,
                )
    ops.gemm.set_table("")

    rows = tpt.build_rows(results, min_gain=min_gain / 100.0, full_coverage=False)
    if emit == "cpp":
        output.write_text(tpt.emit_cpp_rows(rows))
    else:
        output.write_text("\n".join(rows) + "\n")
    click.echo(f"wrote {len(rows)} rows to {output} ({emit})")


if __name__ == "__main__":
    main()
