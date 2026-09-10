from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
DRAFT_PATH = PROJECT_ROOT / "benchmarks/infraswe/astrai-topology-dlo-draft.json"


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _digest_file(path: str) -> str:
    return _digest_bytes((PROJECT_ROOT / path).read_bytes())


def _digest_file_list(paths: tuple[str, ...]) -> str:
    manifest = "".join(
        f"{hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()}  {path}\n"
        for path in paths
    )
    return _digest_bytes(manifest.encode())


def test_infraswe_draft_local_material_digests_are_current() -> None:
    draft = json.loads(DRAFT_PATH.read_text(encoding="utf-8"))
    assert draft["target"]["repository"] == "https://github.com/ViperEkura/AstrAI"
    assert draft["target"]["revision"] == _digest_bytes(
        b"1fad50d8476abe7d4b4cb527eb6e174c511fd409"
    )
    assert draft["baseline"]["revision"] == draft["target"]["revision"]

    candidate_files = (
        "astrai/parallel/topology.py",
        "astrai/parallel/setup.py",
        "astrai/parallel/executor.py",
        "astrai/parallel/__init__.py",
        "scripts/tools/plan_dlo_topology.py",
        "scripts/tools/benchmark_dlo_topology.py",
        "tests/parallel/test_topology.py",
        "tests/parallel/test_topology_cli.py",
        "tests/parallel/test_parallel.py",
        "docs/guides/distributed.md",
    )
    assert draft["candidate"]["revision"] == _digest_file_list(candidate_files)

    project_profile_files = (
        "pyproject.toml",
        "README.md",
        "astrai/parallel/__init__.py",
        "docs/guides/distributed.md",
    )
    assert draft["target"]["project_profile_sha256"] == _digest_file_list(
        project_profile_files
    )

    workload_path = "benchmarks/results/dlo_topology_8x5060ti.json"
    assert draft["deployment"]["workload_portfolio"]["sha256"] == _digest_file(
        workload_path
    )
    assert draft["deployment"]["request_or_step_protocol"]["sha256"] == (
        _digest_file("docs/guides/distributed.md")
    )

    acceptance_files = (
        "tests/parallel/test_topology.py",
        "tests/parallel/test_topology_cli.py",
        "tests/parallel/test_parallel.py",
        workload_path,
    )
    assert draft["acceptance_contract"]["sha256"] == _digest_file_list(acceptance_files)
    probe_files = (
        "tests/parallel/test_topology.py",
        "tests/parallel/test_topology_cli.py",
    )
    assert draft["acceptance_contract"]["probe_set_sha256"] == _digest_file_list(
        probe_files
    )

    profile_files = (
        "astrai/parallel/topology.py",
        "docs/guides/distributed.md",
        "docs/benchmarks/topology_aware_dlo_8x5060ti.md",
    )
    assert draft["project_objectives"]["profile_set_sha256"] == _digest_file_list(
        profile_files
    )
