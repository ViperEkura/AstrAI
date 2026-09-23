"""Run the standard benchmark suite for one checkpoint, with per-job retries.

Plan (dry-run) or launch the suite on free GPUs; each benchmark is one pinned
job, retried when its process dies without writing the output file. Skips
benchmarks whose output file already exists (idempotent restarts). Wrap in
nohup for long runs: if this runner itself dies, rerunning the same command
skips completed benchmarks and continues.
"""

import argparse

from astrai.bench import plan_jobs, run_suite, summarize_tag


def main():
    p = argparse.ArgumentParser(
        description="Run the standard benchmark suite with retries"
    )
    p.add_argument("--ckpt", required=True, help="Checkpoint path for --param_path")
    p.add_argument(
        "--tag", required=True, help="Output tag: results/<bench>_<tag>.json"
    )
    p.add_argument(
        "--benchmarks", default="mmlu,ifeval,humaneval", help="Comma-separated subset"
    )
    p.add_argument("--gpus", default=None, help="Comma-separated preferred GPU indices")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument(
        "--smoke", action="store_true", help="Cheap per-benchmark invocations"
    )
    p.add_argument("--dry-run", action="store_true", help="Print the job plan and exit")
    args = p.parse_args()

    benches = [x for x in args.benchmarks.split(",") if x]
    gpus = [int(x) for x in args.gpus.split(",") if x] if args.gpus else None
    jobs, skipped = plan_jobs(args.ckpt, args.tag, benches, gpus=gpus, smoke=args.smoke)
    for job in jobs:
        print(f"[plan] {job.bench} -> {job.outfile}")
        print(f"       {job.cmd}")
    for bench in skipped:
        print(f"[plan] {bench}: skipped (no smoke invocation; run it full)")

    if args.dry_run:
        return

    status = run_suite(jobs, attempts=args.attempts)
    for bench, state in status.items():
        print(f"[suite] {bench}: {state}")
    print(summarize_tag(args.tag))


if __name__ == "__main__":
    main()
