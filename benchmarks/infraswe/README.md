# InfraSWE: multi-process DDP online rollout

This directory binds the DDP online-rollout change to AstrAI's repository
contract and the checked-in two-GPU NVIDIA L20 evidence. It uses InfraSWE's
`project-fit-system-path-v0.5.1` comparison and scoring models because this is a
training/inference lifecycle path, not a standalone kernel candidate.

InfraSWE v0.5's generic Draft document currently admits kernel and pure-Triton
formula identifiers only. Substituting a kernel formula would misclassify this
change, so the repository stores and validates the native
`ProjectComparisonCell` instead. The result remains explicitly diagnostic and
unsealed.

For the refreshed six-GPU evidence, the latest InfraSWE `main` commit
`a955e00cc3ac79b261d515fb6dd393ba5fd306dd` was used to:

1. validate the comparison cell with the `ProjectComparisonCell` Pydantic model;
2. run the frozen system-path ProjectFit and BenchmarkTrust scoring functions;
3. run the complete InfraSWE test suite (283 tests); and
4. verify that official scoring stays unresolved without a seal, system-trace
   evidence, hidden probes, and a verified manifest.

The visible-evidence diagnostic score remains **95.00/100** and BenchmarkTrust
remains **97.40/100**. The evidence now includes five fresh-process replays of
the real six-rank online GRPO test, plus a six-rank divergent-rollout lifecycle
probe and the non-extension regression suite. Those timings measure replay
stability only; this correctness PR makes no throughput or latency improvement
claim. The complete inputs and rationale are stored in
`benchmarks/results/ddp_rollout_l20_sm89_infraswe_score.json`.
