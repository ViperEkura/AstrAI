# InfraSWE: MoE router consistency

This directory binds AstrAI's MoE router consistency change to its NVIDIA L20
evidence. It uses InfraSWE's `project-fit-system-path-v0.5.1` comparison and
scoring models because the candidate changes framework routing semantics rather
than an isolated kernel.

Before the PR was updated, InfraSWE commit
`811bc775ed5b3a6ec853219245f3469f78818020` was used to validate the
`ProjectComparisonCell`, run the frozen ProjectFit and BenchmarkTrust scoring
functions, and execute the Draft/system-path test subsets.

The visible-evidence diagnostic ProjectFit is **93.16/100** and BenchmarkTrust
is **97.40/100**. Official scoring remains unresolved because this evidence is
unsealed and lacks five fresh-process replays, a system trace, hidden probes,
and a verified evidence manifest. The score is diagnostic, not a certification.

## L20 result

The reproducible probe uses BF16 inputs, seed 3407, 20 warmups, and 100 timed
iterations per case on GPU5. The candidate keeps router probabilities in FP32
until after expert selection. It matches the FP32 reference in every measured
case, while the BF16-before-top-k baseline changes expert sets for 0.05% to
0.15% of ordinary random tokens and 95.95% to 100% of near-tie tokens.

| Tokens | Experts | K | Baseline median | Candidate median | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 8 | 2 | 0.0512 ms | 0.0502 ms | -2.00% |
| 8,192 | 8 | 2 | 0.0543 ms | 0.0511 ms | -5.78% |
| 8,192 | 64 | 8 | 0.0666 ms | 0.0604 ms | -9.23% |
| 32,768 | 64 | 8 | 0.1413 ms | 0.1382 ms | -2.17% |

Reproduce the timings and mismatch rates with
`python benchmarks/training_consistency/benchmark_moe_router.py`; the script
prints machine-readable JSON to standard output without adding generated
results to the repository. The largest measured shape keeps 4 MiB of
additional probability storage.

## Evidence scope

- Baseline: AstrAI `ce2f9d13b32f729c561d0175fd46927a37d9b0a2`.
- Candidate implementation: `d783e4f963316bda4f839ba697ec5b5fe5678a31`.
- Deployment cell: NVIDIA L20 SM89, CUDA 12.8, PyTorch 2.11.0, BF16, GPU5.
- Local regression: 638 passed, 171 environment-dependent skips.
- InfraSWE validation: 41 Draft/system-path tests passed.
- Scope limit: the GPU evidence is a single-process router microbenchmark; it
  does not claim a long-running distributed MoE training soak.
