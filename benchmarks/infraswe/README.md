# InfraSWE Draft: AstrAI topology-aware DLO

This directory binds the benchmark to AstrAI as an explicit repository target. AstrAI
is not one of InfraSWE v0.5's ten built-in projects, so using a built-in default would
silently score the change against the wrong host project. The Draft therefore uses
`target.mode = repository` and remains `D3-contract-proposed`; it does not claim human
review, sealing, or official evaluation.

The Draft was validated and resolved with InfraSWE commit
`811bc775ed5b3a6ec853219245f3469f78818020`:

```bash
PYTHONPATH=src .venv/bin/python -m infraswe.cli draft validate \
  /path/to/AstrAI/benchmarks/infraswe/astrai-topology-dlo-draft.json

PYTHONPATH=src .venv/bin/python -m infraswe.cli draft resolve \
  --local-draft /path/to/AstrAI/benchmarks/infraswe/astrai-topology-dlo-draft.json \
  --output /tmp/astrai-draft-resolution.json
```

Digest construction is deterministic:

- target/baseline: SHA-256 of the target commit ID `1fad50d8476abe7d4b4cb527eb6e174c511fd409`;
- candidate: SHA-256 of the ordered per-file SHA-256 list for the planner, launch
  integration, benchmark scripts, tests, and distributed guide;
- project profile: the same construction over `pyproject.toml`, `README.md`, the
  parallel public API, and distributed guide;
- workload: SHA-256 of `benchmarks/results/dlo_topology_8x5060ti.json`;
- acceptance/probes: ordered per-file digests of the topology tests and captured
  benchmark result;
- precedents: ordered per-file digests of vLLM-Omni's DLO design and backend at the
  retrieval cutoff.

The 7-replay worst-rank collective result is the fast-loop evidence. The same workload
artifact also records successful 2-step and 10-step MiniMax-H3 T2VA requests with
online FP8 and DLO AllGather across eight GPUs. Both remain provisional until an
AstrAI maintainer reviews the contract and advances the Draft lifecycle.

Running InfraSWE's frozen `project-fit-kernel-v0.5` formula over the visible
acceptance evidence produces a diagnostic ProjectFit of **86.73/100** and a
BenchmarkTrust score of **93.06/100**. The machine-readable score card is
`benchmarks/results/dlo_topology_8x5060ti_infraswe_score.json`; it records every
subcomponent input and its rationale. The score is deliberately marked non-official.
InfraSWE leaves the official score unresolved until the Draft is sealed, hidden probes
are complete, and the evidence manifest is verified.
