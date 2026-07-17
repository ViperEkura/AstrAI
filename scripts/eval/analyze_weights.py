"""SVD effective rank & weight statistics analysis for model checkpoints."""

import argparse
import json
from pathlib import Path

import safetensors.torch
import torch


def effective_rank_metrics(w: torch.Tensor) -> dict:
    if w.ndim == 1:
        return {"shape": tuple(w.shape), "is_1d": True}

    w = w.float()
    s = torch.linalg.svdvals(w)
    s_sq = s**2
    total = s_sq.sum()
    cumsum = torch.cumsum(s_sq, dim=0) / total

    min_dim = min(w.shape[0], w.shape[1])
    er_90 = (cumsum < 0.90).sum().item() + 1
    er_95 = (cumsum < 0.95).sum().item() + 1
    er_99 = (cumsum < 0.99).sum().item() + 1

    p = s_sq / total
    p = p[p > 1e-30]
    entropy = -(p * torch.log(p)).sum()
    entropic_rank = torch.exp(entropy).item()

    return {
        "shape": tuple(w.shape),
        "min_dim": min_dim,
        "er_90": er_90,
        "er_95": er_95,
        "er_99": er_99,
        "er_99_norm": er_99 / min_dim,
        "er_95_norm": er_95 / min_dim,
        "entropic_rank": entropic_rank,
        "entropic_rank_norm": entropic_rank / min_dim,
        "top1_ratio": s[0].item() / s.sum().item(),
        "top5_ratio": s[:5].sum().item() / s.sum().item(),
        "decay_ratio": s[-1].item() / s[0].item(),
        "condition_number": s[0].item() / s[-1].item(),
        "mean": w.mean().item(),
        "std": w.std().item(),
        "min": w.min().item(),
        "max": w.max().item(),
    }


def format_header(headers: list[str], widths: list[int]) -> str:
    return "".join(h.ljust(w) for h, w in zip(headers, widths))


def format_row(values: list[str], widths: list[int]) -> str:
    return "".join(v.ljust(w) for v, w in zip(values, widths))


def group_by_component(results: dict[str, dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for key, r in results.items():
        parts = key.split(".")
        if parts[0] == "layers" and len(parts) >= 3:
            sub = parts[2:]
            if sub[0] == "attention":
                comp = f"attn.{sub[1]}"
            elif sub[0] == "mlp":
                comp = f"mlp.{sub[1]}"
            elif sub[0] == "input_norm":
                comp = "input_norm"
            elif sub[0] == "post_attention_norm":
                comp = "post_attn_norm"
            else:
                comp = ".".join(sub)
        else:
            comp = key
        groups.setdefault(comp, []).append(r)
    return groups


def print_component_summary(results: dict[str, dict], title: str):
    groups = group_by_component(results)
    matrix_groups = {
        k: [v for v in vs if not v.get("is_1d")]
        for k, vs in groups.items()
        if any(not v.get("is_1d") for v in vs)
    }

    widths = [20, 12, 12, 12, 12, 12]
    print(f"\n{title}")
    print(
        format_header(
            ["Component", "N", "ER@99%", "EntRank%", "Top1 σ(%)", "Cond. Num"], widths
        )
    )
    print("-" * sum(widths))

    for name in sorted(matrix_groups.keys()):
        items = matrix_groups[name]
        n = len(items)
        print(
            format_row(
                [
                    name,
                    str(n),
                    f"{sum(r['er_99_norm'] for r in items) / n:.4f}",
                    f"{sum(r['entropic_rank_norm'] for r in items) / n:.4f}",
                    f"{sum(r['top1_ratio'] for r in items) / n:.4f}",
                    f"{sum(r['condition_number'] for r in items) / n:.1f}",
                ],
                widths,
            )
        )

    all_er = [
        r["er_99_norm"]
        for vs in matrix_groups.values()
        for r in vs
        if not r.get("is_1d")
    ]
    if all_er:
        m = sum(all_er) / len(all_er)
        print(f"\n  Overall Mean ER@99: {m:.4f} ({m * 100:.1f}% of dimension)")
        if m > 0.85:
            print("  → HIGH utilization: model near capacity → need more params")
        elif m > 0.5:
            print("  → MODERATE utilization: some headroom left")
        else:
            print("  → LOW utilization: significant unused capacity")


def print_layer_grid(results: dict[str, dict]):
    comps = [
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
        "mlp.up",
        "mlp.gate",
        "mlp.down",
    ]
    widths = [6] + [10] * len(comps)
    metric = "er_99_norm"

    print(f"\n--- Per-Layer Effective Rank (99% energy) ---")
    print(format_header(["Layer"] + comps, widths))
    print("-" * sum(widths))

    layer_data: dict[int, dict[str, dict]] = {}
    for key, r in results.items():
        parts = key.split(".")
        if parts[0] != "layers":
            continue
        li = int(parts[1])
        sub = parts[2:]
        if sub[0] == "attention":
            cname = f"attn.{sub[1]}"
        elif sub[0] == "mlp":
            cname = f"mlp.{sub[1]}"
        else:
            continue
        layer_data.setdefault(li, {})[cname] = r

    for li in sorted(layer_data):
        values = [str(li)]
        for c in comps:
            v = layer_data[li].get(c, {}).get(metric, 0)
            values.append(f"{v:.4f}")
        print(format_row(values, widths))


def print_weight_stats(results: dict[str, dict]):
    groups = group_by_component(results)
    widths = [20, 12, 12, 12, 12]
    print(f"\n--- Weight Value Statistics ---")
    print(format_header(["Component", "Mean", "Std", "Min", "Max"], widths))
    print("-" * sum(widths))

    for name in sorted(groups.keys()):
        items = groups[name]
        means = [r.get("mean", 0) for r in items]
        stds = [r.get("std", 0) for r in items]
        mins = [r.get("min", 0) for r in items]
        maxs = [r.get("max", 0) for r in items]
        g_mean = sum(means) / len(means)
        g_std = sum(stds) / len(stds)
        g_min = min(mins)
        g_max = max(maxs)
        print(
            format_row(
                [
                    name,
                    f"{g_mean:.6f}",
                    f"{g_std:.6f}",
                    f"{g_min:.6f}",
                    f"{g_max:.6f}",
                ],
                widths,
            )
        )


def print_params_summary(results: dict[str, dict]):
    total_2d = sum(
        r["shape"][0] * r["shape"][1] for r in results.values() if not r.get("is_1d")
    )
    total_1d = sum(r["shape"][0] for r in results.values() if r.get("is_1d"))
    print(f"\n  Total 2D params: {total_2d:,}")
    print(f"  Total 1D params: {total_1d:,}")
    print(f"  Total params:     {total_2d + total_1d:,}")


def main():
    parser = argparse.ArgumentParser(
        description="SVD effective rank & weight statistics of a model checkpoint."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (containing model.safetensors + config.json).",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="*",
        help="Additional checkpoint directories to compare against.",
    )
    parser.add_argument(
        "--no_svd",
        action="store_true",
        help="Skip SVD analysis, only show weight statistics (mean/std/min/max).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results as JSON to this path.",
    )
    args = parser.parse_args()

    all_results = {}

    def analyze_one(ckpt_dir: str, label: str):
        ckpt_dir = Path(ckpt_dir)
        weights_path = ckpt_dir / "model.safetensors"
        if not weights_path.exists():
            print(f"ERROR: {weights_path} not found")
            return {}

        meta = {}
        meta_path = ckpt_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        print(f"\n{'=' * 70}")
        print(f"  {label}: {ckpt_dir}")
        if meta:
            print(
                f"  Iteration: {meta.get('iteration', '?')}, "
                f"Strategy: {meta.get('strategy', '?')}, "
                f"nprocs={meta.get('nprocs', '?')}"
            )
        print(f"{'=' * 70}")

        print(f"Loading weights...")
        sd = safetensors.torch.load_file(str(weights_path))
        print(f"  {len(sd)} keys loaded")

        weight_keys = [
            k
            for k in sd
            if ".weight" in k and "rotary_embedding" not in k and "freqs_cis" not in k
        ]

        results = {}
        if not args.no_svd:
            print(f"Computing SVD on {len(weight_keys)} tensors...")
            for i, k in enumerate(sorted(weight_keys)):
                print(f"  [{i + 1}/{len(weight_keys)}] {k:<60s}", end="\r")
                results[k] = effective_rank_metrics(sd[k])
            print()
        else:
            print(f"Computing stats on {len(weight_keys)} tensors (no SVD)...")
            for i, k in enumerate(sorted(weight_keys)):
                t = sd[k]
                results[k] = {
                    "shape": tuple(t.shape),
                    "is_1d": t.ndim == 1,
                    "mean": t.float().mean().item(),
                    "std": t.float().std().item(),
                    "min": t.float().min().item(),
                    "max": t.float().max().item(),
                }

        print_params_summary(results)
        if not args.no_svd:
            print_component_summary(
                results, "\n=== SVD Effective Rank by Component ==="
            )
            print_layer_grid(results)
        print_weight_stats(results)
        all_results[label] = results
        return results

    analyze_one(args.ckpt_dir, "Primary")

    if args.compare:
        for cdir in args.compare:
            analyze_one(cdir, f"Compare_{cdir}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
