"""Reverse controls for tests/extension/test_csrc_layout.py.

Each control copies the pristine tree to a scratch directory, injects ONE
violation, and asserts the layout test FAILS. A guard that cannot fire is
decoration; this script is the "see it red before trusting the green" pass.
Never touches the real tree.

    .venv/bin/python scripts/csrc_layout_controls.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "bin" / "python"


def copy_tree(dest: Path) -> None:
    def ignore(directory: str, entries: list[str]) -> list[str]:
        ignored = []
        for entry in entries:
            if entry in ("build", ".venv", "__pycache__", ".git", ".pytest_cache"):
                ignored.append(entry)
        return ignored

    for part in ("csrc", "tests", "scripts"):
        shutil.copytree(ROOT / part, dest / part, dirs_exist_ok=True, ignore=ignore)


def inject(dest: Path, name: str) -> None:
    """One violation, by name."""

    def edit(relpath: str, old: str, new: str) -> None:
        path = dest / relpath
        text = path.read_text()
        if old not in text:
            raise RuntimeError(f"{name}: pattern {old!r} not found in {relpath}")
        path.write_text(text.replace(old, new, 1))

    if name == "stage_includes_launcher":
        edit(
            "csrc/include/arith/softmax.cuh",
            "#include <",
            "#include <launcher/api.h>\n#include <",
        )
    elif name == "stage_touches_torch":
        edit(
            "csrc/include/kernel/quantize.cuh",
            "#include <utils/launch.cuh>",
            "#include <torch/extension.h>",
        )
    elif name == "harness_touches_torch":
        edit(
            "csrc/tests/quant_gemm_test.cu",
            '#include "test_utils.cuh"',
            '#include "test_utils.cuh"\n#include <torch/extension.h>',
        )
    elif name == "new_toplevel_dir":
        (dest / "csrc/include/vectors").mkdir()
        (dest / "csrc/include/vectors/x.cuh").write_text("// dummy\n")
    elif name == "dead_include":
        edit(
            "csrc/include/kernel/quantize.cuh",
            "#include <utils/launch.cuh>",
            "#include <utils/lanch.cuh>",
        )
    elif name == "bare_quoted_shadow":
        (dest / "csrc/tests/shape.cuh").write_text("// shadow copy\n")
        edit(
            "csrc/include/utils/swizzle.cuh",
            "#include <utils/shape.cuh>",
            '#include "shape.cuh"',
        )
    elif name == "stale_pending_edge":
        # The set is empty since the planning split; an entry that names no
        # real edge must fail the two-sided assertion.
        edit(
            "tests/extension/test_csrc_layout.py",
            "PENDING_HOST_EDGES: set[tuple[str, str]] = set()",
            'PENDING_HOST_EDGES: set[tuple[str, str]] = {\n    ("kernel/gemm.cuh", "launcher/plan_table.h"),\n}',
        )
    elif name == "per_dtype_tu_includes_planning":
        edit(
            "csrc/gemm/gemm_bf16_bf16.cu",
            "#include <kernel/gemm.cuh>",
            "#include <kernel/gemm.cuh>\n#include <launcher/planning.h>",
        )
    elif name == "stage_header_includes_planning":
        edit(
            "csrc/include/kernel/gemm.cuh",
            "#include <policy.cuh>",
            "#include <policy.cuh>\n#include <launcher/planning.h>",
        )
    elif name == "quoted_stage_prefix":
        edit(
            "csrc/include/utils/tensor.cuh",
            "#include <utils/swizzle.cuh>",
            '#include "utils/swizzle.cuh"',
        )
    else:
        raise ValueError(name)


CONTROLS = [
    "stage_includes_launcher",
    "stage_header_includes_planning",
    "per_dtype_tu_includes_planning",
    "stage_touches_torch",
    "harness_touches_torch",
    "new_toplevel_dir",
    "dead_include",
    "bare_quoted_shadow",
    "stale_pending_edge",
    "quoted_stage_prefix",
]


def run_layout_tests(tree: Path) -> bool:
    result = subprocess.run(
        [
            str(VENV_PY),
            "-m",
            "pytest",
            "tests/extension/test_csrc_layout.py",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tree,
        capture_output=True,
        text=True,
    )
    return result.returncode != 0


def main() -> int:
    # pristine copy must be green
    with tempfile.TemporaryDirectory(prefix="csrc-ctl-") as scratch:
        pristine = Path(scratch) / "pristine"
        pristine.mkdir()
        copy_tree(pristine)
        if run_layout_tests(pristine):
            print("PRISTINE COPY FAILED — the tree itself is red; aborting")
            return 1
        print("pristine copy: green (8 passed)")

    failures = []
    for name in CONTROLS:
        with tempfile.TemporaryDirectory(prefix="csrc-ctl-") as scratch:
            tree = Path(scratch) / "tree"
            tree.mkdir()
            copy_tree(tree)
            inject(tree, name)
            fired = run_layout_tests(tree)
            status = "FIRES" if fired else "!! MISSES"
            print(f"  {name}: {status}")
            if not fired:
                failures.append(name)

    if failures:
        print(f"\n{len(failures)} control(s) did not fire: {', '.join(failures)}")
        return 1
    print(f"\nall {len(CONTROLS)} controls fire; the layout test can be trusted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
