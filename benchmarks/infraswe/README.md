# InfraSWE: inference runtime release and resume

This directory binds AstrAI's inference lifecycle change to the checked-in
single-GPU NVIDIA L20 evidence. It uses InfraSWE's
`project-fit-system-path-v0.5.1` comparison and scoring models because the
candidate changes the scheduler, engine, and rollout lifecycle rather than an
isolated kernel.

InfraSWE v0.5's generic Draft document currently admits kernel and pure-Triton
formula identifiers only. Substituting a kernel formula would misclassify this
change, so the repository stores and validates the native
`ProjectComparisonCell`. The result remains explicitly diagnostic and
unsealed.

Before the PR was opened, commit
`811bc775ed5b3a6ec853219245f3469f78818020` of InfraSWE was used to:

1. validate the comparison cell with the `ProjectComparisonCell` Pydantic
   model;
2. run the frozen system-path ProjectFit and BenchmarkTrust scoring functions;
3. run the Draft and system-path engine test subsets; and
4. verify that official scoring stays unresolved without a seal, system-trace
   evidence, hidden probes, and a verified manifest.

The visible-evidence diagnostic score is **92.29/100** and BenchmarkTrust is
**97.40/100**. The lower load and cold/steady inputs record that the real L20
probe used a batch-four synthetic workload with CUDA Graph disabled and did
not run a long-lived concurrent serving soak. The complete inputs and
rationales are stored in
`benchmarks/results/inference_release_resume_l20_infraswe_score.json`.

Digest construction is deterministic:

- target/baseline: SHA-256 of target commit
  `88c06db096f197acac2a66953bde445c3d720121`;
- candidate: SHA-256 of the sorted per-file SHA-256 list for implementation,
  tests, benchmark tool, and lifecycle documentation;
- acceptance: the corresponding engine, scheduler, and rollout test files;
- probe/workload: the benchmark tool and three checked-in raw result files;
- required deployment cell: the literal
  `nvidia-l20-sm89-single-gpu-bf16`.

The benchmark source identifier is implementation commit
`5900c786322b522162af0bef0674464806f1628a`. Across 2K, 8K, and 32K context
capacities, five release/resume cycles retained greedy-output parity while
reclaiming 95.96% to 99.74% of scheduler-owned runtime memory.
