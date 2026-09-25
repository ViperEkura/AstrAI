"""Guards for the `csrc` include discipline and the torch-free device surface.

One include root: `csrc/kernels` (the CMake `target_include_directories` root
and the `-I` of every documented harness command). Every project include under
it is spelled root-qualified — `"gemm/mainloop.cuh"`, `"common/mma.cuh"` — so a
spelling names exactly one file. Three headers are called `common.h` and two
`launch.cuh`; a bare include would make its meaning depend on the directory it
happens to sit in, so moving or renaming a file could silently retarget it.

Four properties are pinned here, each cheap to break and expensive to notice:

1. the top-level directory set under `csrc/kernels` is closed — a new family
   directory is a deliberate decision, not a typo;
2. every project include is root-qualified, and resolves to exactly one file
   (a same-directory lookup of the same spelling must not reach a different
   file — the shadowing hazard);
3. every project include resolves, so a rename that leaves a dead include
   behind fails here instead of at the next cold build;
4. the headers that pull in torch/ATen/c10/Python are exactly the declared
   host surface, and the harnesses that must build without torch stay
   torch-free. That split is what lets the standalone `csrc/tests/*.cu`
   harnesses compile with no torch at all.

The torch set is an explicit list rather than a directory rule on purpose: it
is this tree's 7-header reality, and it is one-directional — `gemm/plan_table.h`
is host-side but torch-free, so only the "includes torch" direction is pinned.
Moving those headers under `<family>/host/` would make the rule structural.
"""

import re
from pathlib import Path

CSRC = Path(__file__).resolve().parents[2] / "csrc"
KERNELS = CSRC / "kernels"
TESTS = CSRC / "tests"

# The closed set of top-level directories under the include root. Mirror of
# the "File Layout" section of docs/developer/kernels/README.md.
KERNEL_DIRS = ("attention", "common", "gated_deltanet", "gemm", "quantize")

# Harness-local headers: outside the include root, spelled by their own name
# and found beside the harness (in csrc/tests, or through the bench harness's
# extra `-I csrc/tests`). They carry test-only helpers, so the CMake modules
# never see them.
HARNESS_LOCAL = {"test_utils.cuh"}

# Headers allowed to include torch, relative to csrc/. Every entry must still
# include it — the assertion is two-sided, so the list cannot rot into a stale
# allowlist. The entry translation units are not listed: they are the torch
# boundary itself.
TORCH_HEADERS = {
    "kernels/attention/dtype_list.cuh",
    "kernels/attention/entry_utils.cuh",
    "kernels/gated_deltanet/gated_deltanet.h",
    "kernels/gemm/api.h",
    "kernels/gemm/fp8_state.cuh",
    "kernels/quantize/checks.h",
    "kernels/quantize/launch.cuh",
}

SOURCE_SUFFIXES = (".cu", ".cuh", ".h")
HEADER_SUFFIXES = (".cuh", ".h")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+[<"](?P<name>[^">]+)[">]')
TORCH_INCLUDE_RE = re.compile(r"^(torch|ATen|c10|Python)(/|\.h$)")


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


def is_project_include(src: Path, name: str) -> bool:
    """True when the name denotes a file of this repository.

    Quoted includes cover the toolchain too (`"cuda_bf16.h"`,
    `"torch/extension.h"`, `"c10/core/ScalarType.h"` are all quoted here), so
    what makes an include ours is that it names an include-root directory,
    sits beside the includer, is a declared harness-local header, or *resolves
    to a file of ours* — that last arm is what catches a bare name reaching
    another directory through the include root, which the spelling rule must
    still reject.
    """
    if name in HARNESS_LOCAL:
        return True
    if name.split("/", 1)[0] in KERNEL_DIRS:
        return True
    if (src.parent / name).is_file():
        return True
    return bool(resolves_to(src, name))


def search_roots(src: Path) -> list[Path]:
    """The search path the build provides for this file, in order.

    Quoted includes always search the including file's own directory first —
    that is the preprocessor, not a flag. `csrc/kernels` is the include root
    everywhere. `csrc/tests` rides along only for the bench harness, whose
    header comment carries `-I csrc/kernels -I csrc/tests`.
    """
    roots = [src.parent, KERNELS]
    if src.relative_to(CSRC).parts[0] == "bench":
        roots.append(TESTS)
    return roots


def resolves_to(src: Path, spelling: str) -> list[Path]:
    """Every distinct existing file the spelling reaches, in search order."""
    found = []
    for root in search_roots(src):
        candidate = root / spelling
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
        if is_project_include(path, name)
    ]


def test_kernel_directories_are_the_closed_set() -> None:
    found = sorted(
        child.name
        for child in KERNELS.iterdir()
        if child.is_dir() and not child.name.startswith((".", "__"))
    )
    assert found == sorted(KERNEL_DIRS), (
        "csrc/kernels/ child directories changed; update KERNEL_DIRS and "
        "docs/developer/kernels/README.md together"
    )


def test_project_includes_are_root_qualified() -> None:
    offenders = []
    for path, name in project_includes():
        if name in HARNESS_LOCAL:
            continue
        rel = path.relative_to(CSRC)
        if "/" not in name:
            offenders.append(f"{rel}: bare include {name!r}")
        elif name.split("/", 1)[0] not in KERNEL_DIRS:
            offenders.append(f"{rel}: {name!r} is not under an include root")
    assert not offenders, (
        "every project include is spelled from the csrc/kernels root "
        '("gemm/mainloop.cuh", not "mainloop.cuh"):\n  ' + "\n  ".join(offenders)
    )


def test_no_include_spelling_is_shadowed() -> None:
    offenders = []
    for path, name in project_includes():
        found = resolves_to(path, name)
        if len(found) > 1:
            rel = path.relative_to(CSRC)
            shown = ", ".join(f.relative_to(CSRC).as_posix() for f in found)
            offenders.append(f"{rel}: {name!r} reaches {shown}")
    assert not offenders, (
        "an include spelling must reach exactly one file; a duplicate name or "
        "a same-named subdirectory silently changes which one wins:\n  "
        + "\n  ".join(offenders)
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


def test_torch_surface_is_the_declared_host_headers() -> None:
    includes_torch = set()
    harness_offenders = []
    for path in project_sources():
        hits = [n for n in include_lines(path) if TORCH_INCLUDE_RE.match(n)]
        if not hits:
            continue
        rel = path.relative_to(CSRC).as_posix()
        if KERNELS not in path.parents:
            harness_offenders.append(f"{rel}: {hits[0]}")
        elif path.suffix in HEADER_SUFFIXES:
            includes_torch.add(rel)
        # else: an entry translation unit under csrc/kernels — the torch
        # boundary itself, expected to include it.
    assert not harness_offenders, (
        "the standalone harnesses must stay torch-free (they are the "
        "no-torch correctness gate):\n  " + "\n  ".join(harness_offenders)
    )

    undeclared = sorted(includes_torch - TORCH_HEADERS)
    assert not undeclared, (
        "these headers include torch but are not declared as host surface — "
        "the device headers must stay torch-free so csrc/tests/*.cu compile "
        "without torch:\n  " + "\n  ".join(undeclared)
    )

    stale = sorted(TORCH_HEADERS - includes_torch)
    assert not stale, (
        "these declared torch headers no longer include torch (or moved) — "
        "TORCH_HEADERS must list the real host surface:\n  " + "\n  ".join(stale)
    )
