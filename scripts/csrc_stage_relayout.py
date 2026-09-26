"""One-shot stage-tree relayout of csrc/ (2026-09-26).

Moves every header under csrc/kernels/ into csrc/include/<stage>/, the
translation units up to csrc/<family>/, rewrites every project include to the
angle-bracket root-qualified spelling, fixes the CMake registry, and rewrites
the harness compile commands in the file-header comments. Re-runnable (idem-
potent on an already-moved tree: every step checks before acting).

The move map follows the plan's layout (and the corrected mainloop placement
from notes/csrc-include-stage-layout-2026-09-21.md — kernel/, NOT memory/):
    kernels/common/*.cuh      -> include/{utils,mma,memory,arith}/...
    kernels/attention/*.cuh   -> include/{kernel,memory,mma,arith,utils,launcher}/...
    kernels/gemm/*.cuh        -> include/{kernel,memory,epilogue,utils,launcher}/... + root
    kernels/quantize/*.cuh    -> include/{kernel,datatype,utils,launcher}/...
    kernels/gated_deltanet/*.h-> include/launcher/
    kernels/<family>/*.cu     -> <family>/*.cu  (translation units, no headers left)
    kernels/rotary_emb.cu     -> rotary_emb.cu  (already at csrc/ top, stays)

Include spellings after the move (angle brackets, root = csrc/include):
    project headers: <stage/file.cuh> / <launcher/file.h> / <policy.cuh>
    harness-local test_utils.cuh: quoted, as today.

Run from the repo root AFTER scripts/csrc_include_resolve.py snapshot:
    .venv/bin/python scripts/csrc_stage_relayout.py --apply
Then verify:
    .venv/bin/python scripts/csrc_include_resolve.py verify --json /tmp/inc.json
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
KERNELS = CSRC / "kernels"

# (old path under csrc/, new path under csrc/). Headers keep their names
# unless a same-name collision forces the family prefix (the three common.h).
MOVES: dict[str, str] = {
    # common/ — the cross-family vocabulary, split by stage semantics
    "kernels/common/device.cuh": "include/utils/device.cuh",
    "kernels/common/dtype.cuh": "include/utils/dtype.cuh",
    "kernels/common/launch.cuh": "include/utils/launch.cuh",
    "kernels/common/shape.cuh": "include/utils/shape.cuh",
    "kernels/common/swizzle.cuh": "include/utils/swizzle.cuh",
    "kernels/common/tensor.cuh": "include/utils/tensor.cuh",
    "kernels/common/mma.cuh": "include/mma/mma.cuh",
    "kernels/common/pipeline.cuh": "include/memory/pipeline.cuh",
    "kernels/common/reduce.cuh": "include/arith/reduce.cuh",
    "kernels/common/tma.cuh": "include/memory/tma.cuh",
    # attention/
    "kernels/attention/common.h": "include/utils/attention_common.h",
    "kernels/attention/dispatchers.cuh": "include/kernel/attention_dispatch.cuh",
    "kernels/attention/dtype_list.cuh": "include/launcher/dtype_list.h",
    "kernels/attention/entry_utils.cuh": "include/launcher/entry_utils.h",
    "kernels/attention/layout_policies.cuh": "include/memory/layout_policies.cuh",
    "kernels/attention/mma_utils.cuh": "include/mma/utils.cuh",
    "kernels/attention/softmax.cuh": "include/arith/softmax.cuh",
    "kernels/attention/decode_split_kv.cuh": "include/kernel/attention_decode_split_kv.cuh",
    "kernels/attention/decode_split_kv_mma.cuh": "include/kernel/attention_decode_split_kv_mma.cuh",
    "kernels/attention/prefill_split_q.cuh": "include/kernel/attention_prefill_split_q.cuh",
    "kernels/attention/prefill_split_q_mma.cuh": "include/kernel/attention_prefill_split_q_mma.cuh",
    # gemm/
    "kernels/gemm/common.h": "include/utils/gemm_common.h",
    "kernels/gemm/api.h": "include/launcher/api.h",
    "kernels/gemm/gemm.cuh": "include/kernel/gemm.cuh",
    "kernels/gemm/policy.cuh": "include/policy.cuh",
    "kernels/gemm/plan_table.h": "include/launcher/plan_table.h",
    "kernels/gemm/load.cuh": "include/memory/load.cuh",
    "kernels/gemm/scheduler.cuh": "include/scheduler.cuh",
    "kernels/gemm/mainloop.cuh": "include/kernel/gemm_mainloop.cuh",
    "kernels/gemm/epilogue.cuh": "include/epilogue/writer.cuh",
    "kernels/gemm/fp8_state.cuh": "include/launcher/fp8_state.h",
    # quantize/
    "kernels/quantize/common.h": "include/utils/quantize_common.h",
    "kernels/quantize/checks.h": "include/launcher/checks.h",
    "kernels/quantize/dequant.cuh": "include/datatype/dequant.cuh",
    "kernels/quantize/launch.cuh": "include/launcher/launch.h",
    "kernels/quantize/quantize.cuh": "include/kernel/quantize.cuh",
    # gated_deltanet/ — the only header is the torch-facing declaration
    "kernels/gated_deltanet/gated_deltanet.h": "include/launcher/gated_deltanet.h",
}

# Translation units: same name, family directory directly under csrc/.
TU_MOVES: dict[str, str] = {
    "kernels/attention/decode.cu": "attention/decode.cu",
    "kernels/attention/prefill.cu": "attention/prefill.cu",
    "kernels/attention/paged_decode.cu": "attention/paged_decode.cu",
    "kernels/attention/paged_prefill.cu": "attention/paged_prefill.cu",
    "kernels/quantize/quantize.cu": "quantize/quantize.cu",
    "kernels/gated_deltanet/gated_deltanet_fwd.cu": "gated_deltanet/gated_deltanet_fwd.cu",
    "kernels/gated_deltanet/gated_deltanet_bwd.cu": "gated_deltanet/gated_deltanet_bwd.cu",
    "kernels/gated_deltanet/bindings.cu": "gated_deltanet/bindings.cu",
    "kernels/gemm/gemm.cu": "gemm/gemm.cu",
    "kernels/gemm/bindings.cu": "gemm/bindings.cu",
    "kernels/gemm/fp8_linear.cu": "gemm/fp8_linear.cu",
    "kernels/gemm/gemm_bf16_bf16.cu": "gemm/gemm_bf16_bf16.cu",
    "kernels/gemm/gemm_bf16_int8.cu": "gemm/gemm_bf16_int8.cu",
    "kernels/gemm/gemm_int8_int8.cu": "gemm/gemm_int8_int8.cu",
    "kernels/gemm/gemm_fp8_e4m3_fp8_e4m3.cu": "gemm/gemm_fp8_e4m3_fp8_e4m3.cu",
    "kernels/gemm/gemm_fp8_e5m2_fp8_e5m2.cu": "gemm/gemm_fp8_e5m2_fp8_e5m2.cu",
    # kernels/ top-level single TU
    "kernels/rotary_emb.cu": "rotary_emb.cu",
}

# Old spelling (quoted, kernels-root-qualified or bare) -> new angle-bracket
# spelling. Covers every include the tree can spell; anything not in this map
# keeps its spelling (system/toolchain headers, harness-local test_utils.cuh).
SPELLINGS: dict[str, str] = {
    # common/ vocabulary
    "common/device.cuh": "<utils/device.cuh>",
    "common/dtype.cuh": "<utils/dtype.cuh>",
    "common/launch.cuh": "<utils/launch.cuh>",
    "common/shape.cuh": "<utils/shape.cuh>",
    "common/swizzle.cuh": "<utils/swizzle.cuh>",
    "common/tensor.cuh": "<utils/tensor.cuh>",
    "common/mma.cuh": "<mma/mma.cuh>",
    "common/pipeline.cuh": "<memory/pipeline.cuh>",
    "common/reduce.cuh": "<arith/reduce.cuh>",
    "common/tma.cuh": "<memory/tma.cuh>",
    # attention/
    "attention/common.h": "<utils/attention_common.h>",
    "attention/dispatchers.cuh": "<kernel/attention_dispatch.cuh>",
    "attention/dtype_list.cuh": "<launcher/dtype_list.h>",
    "attention/entry_utils.cuh": "<launcher/entry_utils.h>",
    "attention/layout_policies.cuh": "<memory/layout_policies.cuh>",
    "attention/mma_utils.cuh": "<mma/utils.cuh>",
    "attention/softmax.cuh": "<arith/softmax.cuh>",
    "attention/decode_split_kv.cuh": "<kernel/attention_decode_split_kv.cuh>",
    "attention/decode_split_kv_mma.cuh": "<kernel/attention_decode_split_kv_mma.cuh>",
    "attention/prefill_split_q.cuh": "<kernel/attention_prefill_split_q.cuh>",
    "attention/prefill_split_q_mma.cuh": "<kernel/attention_prefill_split_q_mma.cuh>",
    # gemm/
    "gemm/common.h": "<utils/gemm_common.h>",
    "gemm/api.h": "<launcher/api.h>",
    "gemm/gemm.cuh": "<kernel/gemm.cuh>",
    "gemm/policy.cuh": "<policy.cuh>",
    "gemm/plan_table.h": "<launcher/plan_table.h>",
    "gemm/load.cuh": "<memory/load.cuh>",
    "gemm/scheduler.cuh": "<scheduler.cuh>",
    "gemm/mainloop.cuh": "<kernel/gemm_mainloop.cuh>",
    "gemm/epilogue.cuh": "<epilogue/writer.cuh>",
    "gemm/fp8_state.cuh": "<launcher/fp8_state.h>",
    # quantize/
    "quantize/common.h": "<utils/quantize_common.h>",
    "quantize/checks.h": "<launcher/checks.h>",
    "quantize/dequant.cuh": "<datatype/dequant.cuh>",
    "quantize/launch.cuh": "<launcher/launch.h>",
    "quantize/quantize.cuh": "<kernel/quantize.cuh>",
    # gated_deltanet/
    "gated_deltanet/gated_deltanet.h": "<launcher/gated_deltanet.h>",
}

# Angle-bracket rewrite of the harness compile commands in file-header
# comments: the harnesses document their own nvcc invocation; the include
# root moves from csrc/kernels to csrc/include.
HARNESS_COMMANDS = {
    "csrc/tests/quant_gemm_test.cu": [
        ("-I csrc/kernels", "-I csrc/include"),
    ],
    "csrc/tests/attn_test.cu": [
        ("-I csrc/kernels", "-I csrc/include"),
    ],
    "csrc/tests/attn_paged_test.cu": [
        ("-I csrc/kernels", "-I csrc/include"),
    ],
    "csrc/bench/bench_tile_sweep.cu": [
        ("-I csrc/kernels -I csrc/tests", "-I csrc/include -I csrc/tests"),
    ],
}

INCLUDE_LINE_RE = re.compile(
    r'^(?P<prefix>\s*#\s*include\s+)(?P<open>[<"])(?P<name>[^">]+)(?P<close>[">])'
)

SYSTEM_INCLUDES = {
    "cuda_bf16.h",
    "cuda_fp16.h",
    "cuda_fp8.h",
    "cuda_runtime.h",
    "cuda/std/atomic",
    "cublas_v2.h",
    "cudaTypedefs.h",
}


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{result.stdout}{result.stderr}"
        )
    return result


def rewrite_text(text: str, is_moved_header: bool) -> tuple[str, int]:
    """Rewrite include spellings in one file's text.

    Bare same-directory spellings of moved headers only exist inside kernels/
    families (the fb5a581 discipline made everything root-qualified except the
    harness-local test_utils.cuh), so the rewrite is a pure mapping of the
    old root-qualified spellings onto the new angle-bracket ones.
    """
    out_lines = []
    changed = 0
    for line in text.splitlines(keepends=False):
        match = INCLUDE_LINE_RE.match(line)
        if match is None:
            out_lines.append(line)
            continue
        name = match.group("name")
        new = SPELLINGS.get(name)
        if new is None or name in SYSTEM_INCLUDES or name == "test_utils.cuh":
            out_lines.append(line)
            continue
        # Keep the line's quote style decision simple: project headers always
        # become angle brackets; nothing else changes.
        out_lines.append(f"{match.group('prefix')}{new}")
        changed += 1
    joined = "\n".join(out_lines)
    if text.endswith("\n"):
        joined += "\n"
    return joined, changed


def move_files() -> None:
    all_moves = {**MOVES, **TU_MOVES}
    for old, new in sorted(all_moves.items()):
        src = CSRC / old
        dst = CSRC / new
        if not src.exists():
            if dst.exists():
                print(f"  already moved: {old}")
                continue
            raise FileNotFoundError(f"move source missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(f"move target exists: {dst}")
        git("mv", str(src), str(dst))
        print(f"  mv {old} -> {new}")


def rewrite_includes() -> int:
    """Rewrite include lines in every source file that spells a project header."""
    total = 0
    # All files under csrc/ (post-move paths) plus nothing outside: harnesses
    # live under csrc/tests and csrc/bench and reference project headers too.
    for path in sorted(CSRC.rglob("*")):
        if path.suffix not in (".cu", ".cuh", ".h"):
            continue
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        text = path.read_text(errors="replace")
        new_text, changed = rewrite_text(text, is_moved_header=True)
        if changed:
            path.write_text(new_text)
            total += changed
            print(f"  rewrote {changed:2d} includes in {path.relative_to(ROOT)}")
    return total


def rewrite_cmake() -> None:
    cmake = CSRC / "CMakeLists.txt"
    text = cmake.read_text()
    replacements = [
        # registry paths: kernels/<family>/x.cu -> <family>/x.cu
        (
            'list(TRANSFORM _srcs PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/kernels/")',
            'list(TRANSFORM _srcs PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")',
        ),
        # include root
        (
            '"${CMAKE_CURRENT_SOURCE_DIR}/kernels"\n',
            '"${CMAKE_CURRENT_SOURCE_DIR}/include"\n',
        ),
        # registry comment wording
        (
            "then its source paths under kernels/ (whitespace-",
            "then its source paths under csrc/ (whitespace-",
        ),
        (
            "# may span several TUs; gemm is split into per-dtype-pair instantiation units",
            "# may span several TUs; gemm is split into per-dtype-pair instantiation units",
        ),
    ]
    for old, new in replacements:
        if old not in text:
            if new in text:
                continue  # already applied
            raise RuntimeError(f"CMakeLists pattern not found: {old!r}")
        text = text.replace(old, new)
    cmake.write_text(text)
    print("  CMakeLists.txt: include root + source prefixes rewritten")


def rewrite_harness_comments() -> None:
    for relpath, subs in HARNESS_COMMANDS.items():
        path = ROOT / relpath
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text()
        for old, new in subs:
            if old not in text:
                if new in text:
                    continue
                raise RuntimeError(f"{relpath}: command fragment {old!r} not found")
            text = text.replace(old, new)
        path.write_text(text)
        print(f"  {relpath}: harness command comment rewritten")


def prune_empty_dirs() -> None:
    """Remove the emptied kernels/ tree, empty host/ shells included."""
    if not KERNELS.exists():
        print("  kernels/ already gone")
        return
    leftovers = [p for p in KERNELS.rglob("*") if p.is_file()]
    if leftovers:
        raise RuntimeError(
            "kernels/ still has files, refusing to prune: "
            + ", ".join(str(p.relative_to(ROOT)) for p in leftovers)
        )
    for path in sorted(KERNELS.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
    KERNELS.rmdir()
    print("  kernels/ pruned (empty host/ shells removed)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually move/rewrite (default: dry run, report only)",
    )
    args = parser.parse_args()

    print("moving files:")
    if args.apply:
        move_files()
    else:
        for old in sorted({**MOVES, **TU_MOVES}):
            print(f"  would mv {old} -> {(MOVES | TU_MOVES)[old]}")

    print("rewriting includes:")
    if args.apply:
        n = rewrite_includes()
        print(f"  total: {n} include lines rewritten")
        rewrite_cmake()
        rewrite_harness_comments()
        prune_empty_dirs()
    else:
        print("  (dry run: no files touched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
