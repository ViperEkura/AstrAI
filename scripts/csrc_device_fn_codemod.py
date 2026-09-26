#!/usr/bin/env python3
"""Codemod: introduce ASTRAI_DEVFN and re-anchor the qualifier spellings.

1. The two duplicate `#define DEVICE_FORCEINLINE` (layout_policies.cuh,
   mma/mma.cuh) become `#include <utils/device_fn.cuh>` (the include guard
   dedups across the tree).
2. Every non-static `__device__ __forceinline__` pair at a function-signature
   position becomes ASTRAI_DEVFN, and the file gains the include if missing.
3. The `static __device__ __forceinline__` member-helper sites keep their
   spelling but ride the macro: ASTRAI_DEVFN_STATIC, matching the legacy
   DEVICE_FORCEINLINE macros' contract.

Harness (csrc/tests, csrc/bench) sources are included in the rewrite: they
compile through -I csrc/include too. The blank line after a replaced #define
pair is preserved by matching to end-of-statement, never past the newline.

Idempotent: a second run finds nothing to change.
"""

import re
import sys
from pathlib import Path

CSRC = Path(__file__).resolve().parents[1] / "csrc"
INCLUDE_LINE = "#include <utils/device_fn.cuh>"

# At line start modulo leading whitespace, before the return type or other
# qualifiers; not inside a comment (none of the tree's hits are).
PLAIN_PAIR = re.compile(r"^(\s*)(__device__ __forceinline__)(?=[\s(])", re.M)
STATIC_PAIR = re.compile(r"^(\s*)(static __device__ __forceinline__)(?=[\s(])", re.M)


def add_include(text: str) -> str:
    if INCLUDE_LINE in text:
        return text
    # Insert after the last existing #include line (project includes come
    # last in this tree's convention: toolchain first, then project).
    lines = text.splitlines(keepends=True)
    last = -1
    for i, ln in enumerate(lines):
        if re.match(r"^\s*#\s*include\b", ln):
            last = i
    if last < 0:
        # No includes at all (a bare .cuh): put it after the #pragma once.
        for i, ln in enumerate(lines):
            if ln.strip() == "#pragma once":
                lines.insert(i + 1, "\n" + INCLUDE_LINE + "\n")
                return "".join(lines)
        print(f"  !! no anchor for include, skipped")
        return text
    lines.insert(last + 1, INCLUDE_LINE + "\n")
    return "".join(lines)


def main() -> int:
    changed = 0
    for path in sorted(CSRC.rglob("*")):
        if path.suffix not in (".cu", ".cuh", ".h") or not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(CSRC)
        text = orig = path.read_text()
        if path.name == "device_fn.cuh":
            continue

        # 1. replace the two local macro definitions with the shared include
        if rel == Path("include/memory/layout_policies.cuh"):
            text = text.replace(
                "#define HOST_FORCEINLINE static __host__ __forceinline__\n"
                "#define DEVICE_FORCEINLINE static __device__ __forceinline__\n",
                "",
            )
        if rel == Path("include/mma/mma.cuh"):
            text = text.replace(
                "#define DEVICE_FORCEINLINE static __device__ __forceinline__\n",
                "",
            )

        n_plain = len(PLAIN_PAIR.findall(text))
        n_static = len(STATIC_PAIR.findall(text))
        if n_plain == 0 and n_static == 0 and text == orig:
            continue

        # 2/3. swap the spellings (static first so the plain regex can't
        # match its tail); the leading-indent capture group is preserved
        text = STATIC_PAIR.sub(r"\1ASTRAI_DEVFN_STATIC", text)
        text = PLAIN_PAIR.sub(r"\1ASTRAI_DEVFN", text)

        if text != orig:
            text = add_include(text)
            path.write_text(text)
            changed += 1
            print(f"  {rel}: {n_plain} plain, {n_static} static")
    print(f"changed {changed} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
