# InfraSWE: async rollout policy-version consistency

This directory binds AstrAI's online rollout consistency change to checked-in
NVIDIA L20 evidence. It uses InfraSWE's
`project-fit-system-path-v0.5.1` comparison and scoring models because the
candidate changes optimizer publication, scheduler locking, rollout scoring,
cache publication, and checkpoint lifecycle behavior rather than an isolated
kernel.

Before the PR was opened, InfraSWE commit
`811bc775ed5b3a6ec853219245f3469f78818020` was used to validate the
`ProjectComparisonCell`, run the frozen ProjectFit and BenchmarkTrust scoring
functions, and execute the Draft/system-path test subsets.

The visible-evidence diagnostic ProjectFit is **92.41/100** and BenchmarkTrust
is **97.40/100**. Official scoring remains unresolved because this evidence is
unsealed and lacks five candidate fresh-process replays, a system trace, hidden
probes, and a verified evidence manifest. The score is diagnostic, not a
certification.

## L20 result

The reproducible probe forces one policy-version advance while the reward model
scores every real one-token CUDA rollout. Both revisions use seed 3407, five
warmups, and 50 measured trials on GPU5.

| Revision | Stale accepted | Stale rejected | Median | p99 |
| --- | ---: | ---: | ---: | ---: |
| `ce2f9d1` baseline | 50/50 | 0/50 | 3.2959 ms | 5.0569 ms |
| `3483c3a` candidate | 0/50 | 50/50 | 3.3122 ms | 4.4414 ms |

The candidate eliminates observed stale acceptance. Median trial latency moves
by **+0.50%**, within the declared 2% ceiling, while p99 moves by **-12.17%**.
Raw values are stored in
`benchmarks/results/async_rollout_version_l20_sm89.json`.

## Coverage

- Optimizer mutation and policy-version publication share the generation lock.
- Direct scheduler updates cannot enter during a generator policy snapshot.
- Future versions and results beyond `rollout_max_policy_lag` fail explicitly.
- Results are revalidated after reward scoring and while publishing or reusing
  the rollout cache.
- Online checkpoints persist the actual rollout policy version.
- The contract is exercised through both online GRPO and online DPO.

The complete local suite passed 641 tests with 171 environment-dependent skips;
the focused L20 suite passed 91 tests. This is a deterministic, single-process
race replay, not a long-running multiprocess or external reward-service soak.

## Digest construction

- target profile: sorted SHA-256 list for baseline `README.md` and
  `pyproject.toml`;
- baseline: SHA-256 of target commit
  `ce2f9d13b32f729c561d0175fd46927a37d9b0a2`;
- candidate: SHA-256 of the sorted per-file SHA-256 list for implementation,
  tests, documentation, and the benchmark tool in commit `3483c3a`;
- acceptance: corresponding scheduler, rollout, online-strategy, callback, and
  online end-to-end tests;
- probe/workload: benchmark tool and checked-in raw L20 result; and
- required deployment cell: literal
  `nvidia-l20-sm89-single-gpu-cuda12.8-gpu5`.
