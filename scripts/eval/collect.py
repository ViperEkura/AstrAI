"""Collect results/<bench>_<tag>.json files into a markdown comparison table."""

import argparse

from astrai.bench import collect_results, render_table


def _split(value: str):
    return [x for x in value.split(",") if x] if value else None


def main():
    p = argparse.ArgumentParser(
        description="Collect eval results into a comparison table"
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Directory with <bench>_<tag>.json files",
    )
    p.add_argument(
        "--benchmarks", default=None, help="Comma-separated filter, e.g. mmlu,humaneval"
    )
    p.add_argument("--tags", default=None, help="Comma-separated tag filter")
    p.add_argument(
        "--output", default=None, help="Write markdown here (default: stdout)"
    )
    args = p.parse_args()

    collected = collect_results(
        args.results_dir, benchmarks=_split(args.benchmarks), tags=_split(args.tags)
    )
    table = render_table(collected)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"written to {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
