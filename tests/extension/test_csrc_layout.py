"""Guards for the `csrc` stage-tree include discipline and the torch-free
device surface.

One include root: `csrc/include` (the CMake `target_include_directories` root
and the `-I` of every documented harness command). Every project header is
included with the angle-bracket root-qualified spelling — `<kernel/gemm.cuh>`,
`<mma/mma.cuh>`, `<policy.cuh>` — so a spelling names exactly one file and
"project include" is mechanically recognizable (angle bracket + a first path
segment from the closed set). Quoted includes are reserved for the toolchain
(`"cuda_bf16.h"` was quoted for years) and the harness-local test_utils.cuh.

The stages, by humming's rule (what the code DOES, not which family owns it):

    kernel/    the entry __global__ and its composition (mainloops, split-kv)
    memory/    data movement with stage semantics (loaders, pipelines, TMA,
               staging layout policies)
    mma/       tensor-core backends and fragment helpers
    epilogue/  moving/merging/writing results out
    arith/     value transforms on register fragments (softmax, reductions)
    datatype/  dtype traits and dequant primitives (stage-agnostic)
    utils/     stage-agnostic vocabulary (shapes, swizzles, tensors, dtype
               words, launch checks, device facts) — the sink of the graph
    launcher/  the host surface: the only place headers may touch
               torch/ATen/c10/Python

Properties pinned here, each cheap to break and expensive to notice:

1. the top-level directory set under csrc/include is closed — a new stage is
   a deliberate decision, not a typo;
2. no stage header includes the host surface (launcher/); the one legacy
   device->host edge is registered in PENDING_HOST_EDGES and must be emptied
   by the same change that splits it (the assertion is two-sided, so the
   entry cannot rot into a stale allowlist either);
3. stage headers never include torch/ATen/c10/Python — that is what lets the
   standalone csrc/tests/*.cu harnesses compile without torch at all, and the
   csrc/<family>/ translation units are the torch boundary itself;
4. every project include is root-qualified and resolves to exactly one file
   (a same-directory lookup of the same spelling must not reach a different
   file — the shadowing hazard), and every include resolves at all, so a
   rename that leaves a dead include fails here instead of at the next cold
   build.
"""

import re
from pathlib import Path

CSRC = Path(__file__).resolve().parents[2] / "csrc"
INCLUDE = CSRC / "include"
TESTS = CSRC / "tests"

# The closed set of stage directories under the include root. Mirror of the
# "File Layout" section of docs/developer/kernels/README.md. launcher/ is the
# host surface, not a stage: it may include torch.
STAGES = ("arith", "datatype", "epilogue", "kernel", "memory", "mma", "utils")
HOST_DIRS = ("launcher",)

# Device->host edges that still exist. Each entry is (includer, include
# target), both include-root-relative. The plan is to empty this set by
# splitting gemm.cuh's host planning half out; the assertions below fail on
# BOTH an unlisted new edge and a listed edge that has since been split, so
# the set cannot rot.
PENDING_HOST_EDGES = {
    ("kernel/gemm.cuh", "launcher/plan_table.h"),
}

# Harness-local headers: outside the include root, spelled quoted and found
# beside the harness (in csrc/tests, or through the bench harness's extra
# `-I csrc/tests`). They carry test-only helpers, so the CMake modules never
# see them.
HARNESS_LOCAL = {"test_utils.cuh"}

SOURCE_SUFFIXES = (".cu", ".cuh", ".h")
HEADER_SUFFIXES = (".cuh", ".h")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+[<"](?P<name>[^">]+)[">]')
TORCH_INCLUDE_RE = re.compile(r"^(torch|ATen|c10|Python)(/|\.h$)")
# Include-root-relative path (a stage dir, launcher/, or a root header).
ROOT_SPELLING_RE = re.compile(
    r"^(?:" + "|".join(STAGES + HOST_DIRS) + r")/[^/]+$|^policy\.cuh$|^scheduler\.cuh$"
)


def project_sources() -> list[Path]:
    return sorted(
        path
        for path in CSRC.rglob("*")
        if path.suffix in SOURCE_SUFFIXES
        and path.is_file()
        and "__pycache__" not in path.parts
    )


def include_lines(path: Path) -> list[str]:
    names = []
    for line in path.read_text(errors="replace").splitlines():
        match = INCLUDE_RE.match(line)
        if match is not None:
            names.append(match.group("name"))
    return names


def is_project_include(name: str) -> bool:
    """Angle-bracket spellings under the include root, plus quoted names that
    reach one of our files through any search root (the quoted form itself is
    the violation, but detecting it requires resolving it)."""
    if name in HARNESS_LOCAL:
        return True
    if ROOT_SPELLING_RE.match(name):
        return True
    return False


def search_roots(src: Path, name: str) -> list[Path]:
    """The search path the preprocessor uses, in order.

    A quoted include always searches the including file's own directory
    first — that is the preprocessor, not a flag — so every quoted spelling
    gets src.parent as its first root (the harness-local test_utils.cuh
    rides the same rule). The bench harness additionally carries -I
    csrc/tests, and every spelling ends with the include root csrc/include.
    """
    roots: list[Path] = [src.parent]
    if src.relative_to(CSRC).parts[0] == "bench":
        roots.append(TESTS)
    roots.append(INCLUDE)
    return roots


def resolves_to(src: Path, name: str) -> list[Path]:
    """Every distinct existing file the spelling reaches, in search order."""
    found = []
    for root in search_roots(src, name):
        candidate = root / name
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in found:
                found.append(resolved)
    return found


def project_includes() -> list[tuple[Path, str]]:
    return [
        (path, name)
        for path in project_sources()
        for name in include_lines(path)
        if is_project_include(name)
    ]


def stage_headers() -> list[Path]:
    return sorted(
        path
        for stage in STAGES
        for path in (INCLUDE / stage).rglob("*")
        if path.suffix in HEADER_SUFFIXES and path.is_file()
    )


def test_stage_directories_are_the_closed_set() -> None:
    found = sorted(
        child.name
        for child in INCLUDE.iterdir()
        if child.is_dir() and not child.name.startswith((".", "__"))
    )
    assert found == sorted(STAGES + HOST_DIRS), (
        "csrc/include/ child directories changed; update STAGES/HOST_DIRS and "
        "docs/developer/kernels/README.md together"
    )


def test_quoted_includes_are_harness_local_only() -> None:
    """A quoted project spelling is invisible to the angle-bracket checks.

    Quoted includes search the includer's own directory first, so a quoted
    name means whatever file sits beside it — exactly the silent retarget
    the root-qualified discipline exists to prevent (a bare "shape.cuh"
    resolves differently per directory). The sanctioned quoted spellings are
    the harness-local test_utils.cuh and the toolchain's own quoted headers
    (cuda_bf16.h and friends), which never resolve under csrc/include. A
    quoted name that reaches a file of ours through any search root is a
    project header spelled wrong.
    """
    offenders = []
    for path in project_sources():
        for line in path.read_text(errors="replace").splitlines():
            match = re.match(r'^\s*#\s*include\s+"([^"]+)"', line)
            if match is None:
                continue
            name = match.group(1)
            if name in HARNESS_LOCAL:
                continue
            if resolves_to(path, name):
                offenders.append(f"{path.relative_to(CSRC)}: {name!r}")
    assert not offenders, (
        "project headers are included with the angle-bracket root-qualified "
        "spelling; a quoted project path resolves through the includer's own "
        "directory first and silently changes meaning on a move:\n  "
        + "\n  ".join(offenders)
    )


def test_stage_headers_do_not_include_the_host_surface() -> None:
    edges = set()
    for path in stage_headers():
        rel_dir = path.parent.relative_to(INCLUDE).as_posix()
        for name in include_lines(path):
            if name.startswith("launcher/"):
                edges.add((rel_dir + "/" + path.name, name))
    undeclared = edges - PENDING_HOST_EDGES
    assert not undeclared, (
        "stage headers must not include the host surface (launcher/) — a "
        "device header reaching a torch-pulling header breaks the no-torch "
        "harness build. New edges need a split, not an allowlist entry:\n  "
        + "\n  ".join(f"{a} -> {b}" for a, b in sorted(undeclared))
    )
    stale = PENDING_HOST_EDGES - edges
    assert not stale, (
        "a registered pending edge no longer exists — empty PENDING_HOST_EDGES "
        "in the same change that split it (the set must not rot):\n  "
        + "\n  ".join(f"{a} -> {b}" for a, b in sorted(stale))
    )


def test_stage_headers_do_not_include_torch() -> None:
    offenders = []
    for path in stage_headers():
        hits = [n for n in include_lines(path) if TORCH_INCLUDE_RE.match(n)]
        if hits:
            offenders.append(f"{path.relative_to(INCLUDE)}: {hits[0]}")
    assert not offenders, (
        "stage headers must stay torch-free (they are the no-torch compile "
        "gate for csrc/tests/*.cu); the torch surface is launcher/ + the "
        "translation units:\n  " + "\n  ".join(offenders)
    )


def test_harnesses_stay_torch_free() -> None:
    offenders = []
    for path in project_sources():
        rel = path.relative_to(CSRC)
        if rel.parts[0] not in ("tests", "bench"):
            continue
        hits = [n for n in include_lines(path) if TORCH_INCLUDE_RE.match(n)]
        if hits:
            offenders.append(f"{rel}: {hits[0]}")
    assert not offenders, (
        "the standalone harnesses must stay torch-free (they are the "
        "no-torch correctness gate):\n  " + "\n  ".join(offenders)
    )


def test_every_project_include_is_root_qualified() -> None:
    offenders = []
    for path, name in project_includes():
        if name in HARNESS_LOCAL:
            continue
        if not ROOT_SPELLING_RE.match(name):
            offenders.append(f"{path.relative_to(CSRC)}: {name!r}")
    assert not offenders, (
        "every project include is spelled from the csrc/include root "
        "('<kernel/gemm.cuh>', not 'gemm.cuh' or '../kernel/gemm.cuh'):\n  "
        + "\n  ".join(offenders)
    )


def test_no_include_spelling_is_shadowed() -> None:
    offenders = []
    for path, name in project_includes():
        found = resolves_to(path, name)
        if len(found) > 1:
            shown = ", ".join(f.relative_to(CSRC).as_posix() for f in found)
            offenders.append(f"{path.relative_to(CSRC)}: {name!r} reaches {shown}")
    assert not offenders, (
        "an include spelling must reach exactly one file; a duplicate name "
        "silently changes which one wins:\n  " + "\n  ".join(offenders)
    )


def test_every_project_include_resolves() -> None:
    offenders = []
    for path, name in project_includes():
        if not resolves_to(path, name):
            offenders.append(f"{path.relative_to(CSRC)}: {name!r}")
    assert not offenders, (
        "these includes resolve to no file (a rename left a dead include "
        "behind, or the include root is missing an entry):\n  " + "\n  ".join(offenders)
    )
