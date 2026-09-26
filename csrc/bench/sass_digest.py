"""Per-symbol SASS digest of a compile tree — the zero-behavior-refactor gate.

Hashes every kernel symbol's SASS (`cuobjdump -sass`) out of the `.o` files
under a build directory, keyed by mangled symbol name. Keys are symbol names,
not object paths, precisely so the digest survives a file move: that is the
one refactor class this tool exists to adjudicate.

    .venv/bin/python csrc/bench/sass_digest.py build/relayout-before \
        --output /tmp/before.json
    .venv/bin/python csrc/bench/sass_digest.py build/relayout-after \
        --compare /tmp/before.json          # exits nonzero on any difference

--compare reports added / removed / changed functions; for changed ones it
prints the instruction-count delta and the mnemonic-histogram delta, which is
the adjudication an inline-boundary drift class falls back to when per-symbol
identity cannot hold (see docs/developer/kernels/README.md). REG usage rides
along in the record as `reg` and is reported, but the gate is the SASS hash.

The digest is only meaningful against a build produced by the *same*
configure command: different arch tokens (120 vs 120a), `--use_fast_math`, or
a different CUDA toolkit all move SASS without any source change. Rebuild both
sides the same way, and run once twice on an unchanged tree first — that
control group is what tells you a later difference is real.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

FUNCTION_RE = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$")
SASS_LINE_RE = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/")
MNEMONIC_RE = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(@!?U?P\d+\s+)?([A-Z][A-Z0-9._]*)")
RES_USAGE_RE = re.compile(r"^\s*REG:(\d+)\s+STACK:(\d+)")
# nvcc embeds the TU's on-disk path into the anonymous-namespace segment of
# symbols defined in one: _GLOBAL__N__<16hex>_<dir>_<file>_cu_<rest>. A pure
# file move keeps the SASS but mangles this name segment, so a move-adjudicating
# comparison normalizes it away first. Two spellings exist: the TU-qualified
# form (<hash>_<dir>_<file>_cu_) and the bare form (<hash> glued straight to
# the symbol name — gated_deltanet's kernels land there, and the hash also
# changes when an unrelated namespace block is inserted around the TU's
# definitions, so both must normalize).
ANON_NAMESPACE_RE = re.compile(r"_GLOBAL__N__[0-9a-f]+")


def find_cuobjdump() -> str:
    """Return the cuobjdump to use, preferring an explicit override."""
    override = os.environ.get("CUOBJDUMP")
    if override:
        return override
    found = shutil.which("cuobjdump")
    if found:
        return found
    fallback = "/usr/local/cuda/bin/cuobjdump"
    if Path(fallback).exists():
        return fallback
    raise SystemExit("cuobjdump not found; set CUOBJDUMP=/path/to/cuobjdump")


def run(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed on {cmd[-1]}:\n{result.stderr}")
    return result.stdout


def split_functions(text: str) -> dict[str, list[str]]:
    """Split a `cuobjdump -sass` dump into mangled name -> its SASS lines."""
    functions: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = FUNCTION_RE.match(line)
        if match is not None:
            current = match.group(1)
            functions.setdefault(current, [])
            continue
        if current is not None and SASS_LINE_RE.match(line):
            functions[current].append(line.rstrip())
    return functions


def parse_res_usage(text: str) -> dict[str, int]:
    """Map mangled name -> REG count from a `cuobjdump -res-usage` dump."""
    registers: dict[str, int] = {}
    current: str | None = None
    for line in text.splitlines():
        match = FUNCTION_RE.match(line)
        if match is not None:
            current = match.group(1)
            continue
        usage = RES_USAGE_RE.match(line)
        if usage is not None and current is not None and current not in registers:
            registers[current] = int(usage.group(1))
    return registers


def mnemonic_histogram(lines: list[str]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for line in lines:
        match = MNEMONIC_RE.search(line)
        if match is None:
            continue
        key = match.group(2)
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def digest_object(cuobjdump: str, obj: Path) -> dict[str, dict]:
    sass = split_functions(run([cuobjdump, "-sass", str(obj)]))
    registers = parse_res_usage(run([cuobjdump, "-res-usage", str(obj)]))

    records: dict[str, dict] = {}
    for name, lines in sass.items():
        body = "\n".join(lines)
        records[name] = {
            "sha256": hashlib.sha256(body.encode()).hexdigest(),
            "instrs": len(lines),
            "reg": registers.get(name, -1),
            "mnemonics": mnemonic_histogram(lines),
        }
    return records


def project_objects(build_dir: Path) -> list[Path]:
    """Every project object under a build dir, in a path-stable order.

    CMake's compiler-identification probes are excluded: they are throwaway
    sm_75 dumps, not project code.
    """
    objects = [
        obj
        for obj in build_dir.rglob("*.o")
        if "CompilerIdCUDA" not in obj.parts and "CMakeCUDACompilerId" not in obj.name
    ]
    return sorted(objects)


def collect(cuobjdump: str, build_dir: Path) -> dict:
    objects = project_objects(build_dir)
    if not objects:
        raise SystemExit(f"no project .o files under {build_dir}; build it first")

    # A template instantiated in two translation units lands in both objects
    # (the fp8 quantize kernels live in quantize.cu and in the gemm module).
    # Key the gate on the SET of per-copy hashes, so the result cannot depend
    # on which object happened to be walked first.
    per_symbol: dict[str, list[tuple[str, dict]]] = {}
    for obj in objects:
        for name, record in digest_object(cuobjdump, obj).items():
            per_symbol.setdefault(name, []).append((obj.name, record))

    functions: dict[str, dict] = {}
    multi_copy: list[str] = []
    for name, copies in per_symbol.items():
        if len(copies) > 1:
            multi_copy.append(name)
        hashes = sorted({record["sha256"] for _, record in copies})
        first_object, first = copies[0]
        functions[name] = {
            "sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
            "copies": len(copies),
            "sites": sorted({obj for obj, _ in copies}),
            "instrs": first["instrs"],
            "reg": first["reg"],
            "object": first_object,
            "mnemonics": first["mnemonics"],
        }

    return {
        "tool": run([cuobjdump, "--version"]).strip().splitlines()[-1],
        "cuobjdump": cuobjdump,
        "n_objects": len(objects),
        "n_functions": len(functions),
        "total_instrs": sum(record["instrs"] for record in functions.values()),
        "multi_copy_symbols": sorted(multi_copy),
        "functions": functions,
    }


def normalize_symbol(name: str) -> str:
    name = ANON_NAMESPACE_RE.sub("_GLOBAL__N___", name)
    # The hash segment may also trail the _cu_ qualifier, glued to the
    # symbol name proper (a third nvcc spelling): strip a run of hex
    # immediately after "_cu_" as well, or the two builds of one TU under
    # different namespace wrappers compare as different symbols.
    return re.sub(r"(_cu_)[0-9a-f]+", r"\1", name)


def compare(before: dict, after: dict, normalize: bool = False) -> int:
    old, new = before["functions"], after["functions"]
    if normalize:
        old = {normalize_symbol(k): v for k, v in old.items()}
        new = {normalize_symbol(k): v for k, v in new.items()}
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(
        name
        for name in set(old) & set(new)
        if old[name]["sha256"] != new[name]["sha256"]
    )
    recopied = sorted(
        name
        for name in set(old) & set(new)
        if old[name]["copies"] != new[name]["copies"] and name not in changed
    )

    print(
        f"objects {before['n_objects']} -> {after['n_objects']}, "
        f"functions {before['n_functions']} -> {after['n_functions']}, "
        f"instructions {before['total_instrs']} -> {after['total_instrs']}"
    )

    for name in added:
        print(f"  + {name} ({new[name]['instrs']} instrs)")
    for name in removed:
        print(f"  - {name} ({old[name]['instrs']} instrs)")

    for name in changed:
        old_record, new_record = old[name], new[name]
        delta = new_record["instrs"] - old_record["instrs"]
        print(
            f"  ~ {name}: {old_record['instrs']} -> {new_record['instrs']} instrs ({delta:+d})"
        )
        if old_record["reg"] != new_record["reg"]:
            print(f"      REG {old_record['reg']} -> {new_record['reg']}")
        histogram_delta = []
        for key in sorted(set(old_record["mnemonics"]) | set(new_record["mnemonics"])):
            diff = new_record["mnemonics"].get(key, 0) - old_record["mnemonics"].get(
                key, 0
            )
            if diff:
                histogram_delta.append((abs(diff), key, diff))
        for _, key, diff in sorted(histogram_delta, reverse=True)[:8]:
            print(f"      {key} {diff:+d}")

    # Same SASS but a different number of instantiation sites means a header
    # reached a translation unit it did not reach before — in a refactor that
    # claims to change nothing, that is a change.
    for name in recopied:
        print(
            f"  = {name}: {old[name]['copies']} -> {new[name]['copies']} instantiation sites"
        )

    failures = len(added) + len(removed) + len(changed) + len(recopied)
    if failures:
        print(
            f"MISMATCH: {len(added)} added, {len(removed)} removed, "
            f"{len(changed)} changed, {len(recopied)} re-copied"
        )
        return 1
    print("MATCH: every symbol's SASS is identical")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", help="a cmake build directory holding .o files")
    parser.add_argument("--output", help="write the digest JSON here")
    parser.add_argument(
        "--compare", help="a baseline digest JSON; exit nonzero on any difference"
    )
    parser.add_argument(
        "--with-mnemonics",
        action="store_true",
        help="keep per-symbol mnemonic histograms (they are dropped otherwise)",
    )
    parser.add_argument(
        "--normalize-anon",
        action="store_true",
        help=(
            "strip the anonymous-namespace path segment from symbol names "
            "before comparing — the adjudication mode for pure file moves, "
            "where nvcc re-hashes the TU's on-disk path into every symbol "
            "it defines in an anonymous namespace while the SASS itself is "
            "unchanged"
        ),
    )
    args = parser.parse_args()

    cuobjdump = find_cuobjdump()
    digest = collect(cuobjdump, Path(args.build_dir))

    if not args.with_mnemonics:
        for record in digest["functions"].values():
            record.pop("mnemonics")

    if digest["multi_copy_symbols"]:
        print(
            f"note: {len(digest['multi_copy_symbols'])} symbol(s) instantiated in more than one object"
        )

    print(
        f"{args.build_dir}: {digest['n_objects']} objects, {digest['n_functions']} functions, "
        f"{digest['total_instrs']} instructions ({digest['tool']})"
    )

    if args.output:
        Path(args.output).write_text(json.dumps(digest, indent=1, sort_keys=True))
        print(f"wrote {args.output}")

    if args.compare:
        before = json.loads(Path(args.compare).read_text())
        return compare(before, digest, normalize=args.normalize_anon)
    return 0


if __name__ == "__main__":
    sys.exit(main())
