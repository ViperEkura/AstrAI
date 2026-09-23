"""Evaluate checkpoints as they land while training is running.

Polls --ckpt_dir for epoch_<e>_step_<n> checkpoints and runs the benchmark
suite on each new one, pinned to the spare GPUs in --gpus — never the GPUs
training runs on. Run from the repo root (job commands use root-relative
paths). Result tags are "<tag>_epoch_<e>_step_<n>"; see
docs/guides/evaluation.md for behavior details.
"""

import argparse

from astrai.bench import watch_checkpoints


def main():
    p = argparse.ArgumentParser(
        description="Watch a training ckpt_dir and evaluate new checkpoints"
    )
    p.add_argument(
        "--ckpt_dir", required=True, help="Training checkpoint directory to watch"
    )
    p.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated spare GPU ids, e.g. 0,1,2 (must not overlap training GPUs)",
    )
    p.add_argument(
        "--benchmarks",
        default="ifeval,humaneval,mbpp2",
        help="Comma-separated subset of mmlu,ifeval,humaneval,mbpp,mbpp2,hellaswag",
    )
    p.add_argument(
        "--tag",
        default="",
        help="Prefix for result tags; default is the bare checkpoint dir name",
    )
    p.add_argument(
        "--every",
        type=int,
        default=1,
        help="Only evaluate checkpoints whose step is a multiple of N",
    )
    p.add_argument("--poll", type=int, default=30, help="Poll interval in seconds")
    p.add_argument("--attempts", type=int, default=3, help="Retries per benchmark")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--logs-dir", default="logs")
    p.add_argument(
        "--smoke", action="store_true", help="Cheap per-benchmark invocations"
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Evaluate existing checkpoints and exit instead of polling",
    )
    args = p.parse_args()

    benches = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    watch_checkpoints(
        args.ckpt_dir,
        benches,
        gpus,
        tag=args.tag,
        poll_s=args.poll,
        once=args.once,
        every=args.every,
        smoke=args.smoke,
        results_dir=args.results_dir,
        logs_dir=args.logs_dir,
        attempts=args.attempts,
    )


if __name__ == "__main__":
    main()
