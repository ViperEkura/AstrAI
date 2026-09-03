# InfraSWE: configurable GRPO and DAPO objective

This directory binds AstrAI's configurable GRPO/DAPO objective to the
checked-in NVIDIA L20 evidence. It uses InfraSWE's
`project-fit-system-path-v0.5.1` comparison and scoring models because the
candidate changes the training objective, CLI contract, validation behavior,
and reward-shaping path rather than an isolated kernel.

InfraSWE v0.5's generic Draft document currently admits kernel and pure-Triton
formula identifiers only. Substituting a kernel formula would misclassify this
change, so the repository stores and validates the native
`ProjectComparisonCell`. The result remains explicitly diagnostic and
unsealed.

Before the PR was opened, InfraSWE commit
`811bc775ed5b3a6ec853219245f3469f78818020` was used to validate the comparison
cell, run the frozen system-path ProjectFit and BenchmarkTrust functions, run
53 Draft/system-path engine tests, and verify that official scoring remains
unresolved without its required evidence envelope.

The visible-evidence diagnostic score is **92.15/100** and BenchmarkTrust is
**97.40/100**. The lower load and cold/steady inputs record that this evidence
measures objective math and deterministic training tests, not downstream reward
quality or a long-running training job. Complete inputs and rationales are in
`benchmarks/results/dapo_objective_l20_infraswe_score.json`.

## Scope

The candidate adds three independently selectable objective pieces:

- DAPO Clip-Higher through independent lower and upper ratio bounds;
- token-level DAPO or equal-sequence GRPO loss aggregation; and
- optional linear soft-overlong reward shaping before group advantage
  normalization.

Defaults retain AstrAI's existing symmetric, token-normalized objective and
disable overlong shaping. Dynamic sampling is intentionally excluded because
correct support requires a rollout refill buffer rather than silently dropping
zero-variance groups from an already generated batch.

## L20 result

The reproducible objective probe uses FP32 log-probabilities, seed 3407, 20
warmups, and 100 timed iterations per case. The default candidate loss matches
the baseline exactly. Its median latency ranges from -0.80% to 0.00% relative
to baseline. Opting into Clip-Higher and soft-overlong shaping adds 0.056 to
0.059 ms in the isolated objective microbenchmark (about 22% relative to the
sub-millisecond objective, excluding model forward/backward).

| Batch×group | Response length | Default parity | Baseline | Default candidate | Full DAPO |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4×8 | 256 | 0.0 abs | 0.2560 ms | 0.2540 ms | 0.3124 ms |
| 4×8 | 1,024 | 0.0 abs | 0.2570 ms | 0.2570 ms | 0.3164 ms |
| 8×8 | 2,048 | 0.0 abs | 0.2693 ms | 0.2693 ms | 0.3297 ms |
| 4×8 | 4,096 | 0.0 abs | 0.2693 ms | 0.2683 ms | 0.3287 ms |

Raw timings are checked in at
`benchmarks/results/dapo_objective_l20_sm89.json`.

## Digest construction

- target/baseline: SHA-256 of target commit
  `88c06db096f197acac2a66953bde445c3d720121`;
- candidate: SHA-256 of the sorted per-file SHA-256 list for implementation,
  tests, benchmark tool, and changed project documentation, excluding the
  InfraSWE and raw-score artifacts to avoid a recursive digest;
- acceptance: GRPO strategy, online end-to-end, and CLI tests;
- probe/workload: the benchmark tool and checked-in raw result; and
- required deployment cell: the literal
  `nvidia-l20-sm89-single-gpu-objective-gpu5`.

The benchmark source identifier is implementation commit
`bd10b800018794db742cbfe6cf7bcc7c32ea6f44`. The local suite passed 635 tests
with 103 environment-dependent skips, and the focused L20 suite passed 26
tests.
