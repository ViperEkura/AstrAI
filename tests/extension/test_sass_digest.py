"""SASS comparison keeps kernel identity across source-file moves."""

import runpy
from pathlib import Path


def test_moved_translation_unit_preserves_normalized_symbol():
    tool = runpy.run_path(
        str(Path(__file__).parents[2] / "csrc" / "bench" / "sass_digest.py")
    )
    old = "_ZN45_GLOBAL__N__8c8432a2_12_symmetric_cu_967134e316symmetric_kernelIiEEvv"
    moved = "_ZN43_GLOBAL__N__29f92144_10_kernels_cu_5827182616symmetric_kernelIiEEvv"
    assert tool["normalize_symbol"](old) == tool["normalize_symbol"](moved)
    assert tool["normalize_symbol"]("_ZN_GLOBAL__N__abc123kernelEv") == (
        "_ZN_GLOBAL__N___kernelEv"
    )
