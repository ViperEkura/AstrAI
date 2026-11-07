"""Keep native CUDA interfaces and planners independent of Torch."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc"
INCLUDE = CSRC / "include"
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
FORBIDDEN = ("api/torch/", "torch/", "ATen/", "c10/", "pybind11/")


def _native_roots():
    headers = sorted(
        path
        for path in INCLUDE.rglob("*")
        if path.suffix in (".h", ".cuh")
        and not path.is_relative_to(INCLUDE / "api" / "torch")
    )
    return headers + [
        CSRC / "gemm" / "plan" / name
        for name in ("planner.cpp", "table.cpp", "builtin.cpp")
    ]


@pytest.mark.parametrize(
    "source", _native_roots(), ids=lambda p: str(p.relative_to(CSRC))
)
def test_native_dependencies_are_torch_free(source):
    pending = [(source, [source])]
    visited = set()
    while pending:
        path, chain = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for name in INCLUDE_RE.findall(path.read_text()):
            trace = " -> ".join(str(p.relative_to(CSRC)) for p in chain)
            assert not name.startswith(FORBIDDEN) and name != "Python.h", (
                f"Native dependency reaches Torch/Python: {trace} -> {name}"
            )
            # Resolve private quoted includes and root-qualified project headers.
            candidates = (path.parent / name, INCLUDE / name)
            dependency = next((p.resolve() for p in candidates if p.is_file()), None)
            if dependency is not None and dependency.is_relative_to(CSRC):
                pending.append((dependency, chain + [dependency]))
            elif name.startswith("api/"):
                pytest.fail(f"Missing API include: {trace} -> {name}")
