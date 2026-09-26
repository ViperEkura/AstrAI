"""Snapshot and verify the concrete resolution of every project include site.

The discipline the 2026-09-25 rewrite used, kept as a reusable preflight:
before any include-path rewrite, record which *file* each include site
actually resolves to; after the rewrite, assert it still resolves to the same
file. A bare or family-qualified spelling whose meaning depends on the
directory it sits in (three headers are called `common.h`, two `launch.cuh`)
would otherwise silently retarget during a move.

    .venv/bin/python scripts/csrc_include_resolve.py snapshot --json /tmp/inc.json
    .venv/bin/python scripts/csrc_include_resolve.py verify --json /tmp/inc.json

The snapshot is keyed by (source file, line number, old spelling) so a rewrite
that only touches the spelling (never the line count before the include) can
be matched up afterwards. `verify` re-runs the resolution and compares targets.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
KERNELS = CSRC / "kernels"
TESTS = CSRC / "tests"

INCLUDE_RE = re.compile(r'^\s*#\s*include\s+([<"])(?P<name>[^">]+)[">]')

# Roots the build provides today: the includer's own directory (quoted
# includes search it first, a preprocessor fact), then the single include
# root csrc/kernels, and csrc/tests only for the bench harness.
KERNEL_DIRS = ("attention", "common", "gated_deltanet", "gemm", "quantize")
HARNESS_LOCAL = {"test_utils.cuh"}


def project_sources() -> list[Path]:
    return sorted(
        path
        for path in CSRC.rglob("*")
        if path.suffix in (".cu", ".cuh", ".h")
        and path.is_file()
        and "__pycache__" not in path.parts
    )


def search_roots(src: Path) -> list[Path]:
    roots = [src.parent, KERNELS]
    if src.relative_to(CSRC).parts[0] == "bench":
        roots.append(TESTS)
    return roots


def is_project_include(src: Path, name: str) -> bool:
    if name in HARNESS_LOCAL:
        return True
    if name.split("/", 1)[0] in KERNEL_DIRS:
        return True
    if (src.parent / name).is_file():
        return True
    for root in search_roots(src):
        if (root / name).is_file():
            return True
    return False


def resolve(src: Path, name: str) -> list[Path]:
    found = []
    for root in search_roots(src):
        candidate = root / name
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in found:
                found.append(resolved)
    return found


def snapshot() -> dict:
    sites = {}
    for src in project_sources():
        text = src.read_text(errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = INCLUDE_RE.match(line)
            if match is None:
                continue
            name = match.group("name")
            if not is_project_include(src, name):
                continue
            targets = [str(p.relative_to(ROOT)) for p in resolve(src, name)]
            key = f"{src.relative_to(ROOT)}:{lineno}"
            sites[key] = {"spelling": name, "targets": targets}
    return sites


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--json", required=True, type=Path)
    ver = sub.add_parser("verify")
    ver.add_argument("--json", required=True, type=Path)

    args = parser.parse_args()
    if args.cmd == "snapshot":
        sites = snapshot()
        args.json.write_text(json.dumps(sites, indent=1, sort_keys=True))
        print(f"snapshot: {len(sites)} include sites -> {args.json}")
        return 0

    old = json.loads(args.json.read_text())
    new = snapshot()
    bad = []
    for key, entry in old.items():
        cur = new.get(key)
        if cur is None:
            bad.append(f"{key}: site gone ({entry['spelling']!r})")
        elif cur["targets"] != entry["targets"]:
            bad.append(
                f"{key}: {entry['spelling']!r} -> {entry['targets']} "
                f"now resolves to {cur['targets']} ({cur['spelling']!r})"
            )
    if bad:
        print("RESOLUTION DRIFT:", file=sys.stderr)
        for line in bad:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"verify: all {len(old)} sites resolve to the same files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
