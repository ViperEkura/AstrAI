"""GEMM-kernel interface adapter (the only module touching the pybind).

One adapter per compiled module: this file covers the ``gemm`` module's
``quant_gemm`` binding — stateless, called directly — plus the flat
``set_*`` / ``state`` / ``probe`` / ``facts`` / ``tile_vocabulary`` views over
the planner's bindings in their raw wire shapes (the ``csrc/bench`` tools
parse those keys, so those spellings are contract). The plan's records, the
``plan`` facade and the runtime autotuner are policy and live next door in
``astrai.extension.policy.gemm.plan``; the fp8/int8 quantization policy in
``astrai.extension.policy.quantization.autocast``. The mma consumes bf16 fragments (or the native
fp8 mma for symmetric fp8 pairs); int8 and fp8 operands dequantize in-register
between the smem read and the mma — never a separate F2F pass — and
per-operand scales fold multiplicatively into the epilogue.

The scale contract — what a scale tensor may be and per which operand side —
is stated once in ``docs/developer/kernels/gemm.md`` ("Scales") and in the
kernel's ``GemmParams``: this layer validates to it, it does not restate it.
The ``trans_a``/``trans_b`` flags name
the math (``True`` = operand laid out ``[contract][rows]``);
inner-transposed views fold into the kernel layout at zero copy. ``bias``
(CUDA bf16 1D of length n) fuses into the epilogue.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch

from astrai.extension.policy.gemm.plan import PLANNER_MODES, Rows, note_launch
from astrai.extension.runtime.loader import get_module


def quant_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    trans_a: bool = False,
    trans_b: bool = True,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quantized GEMM: ``(a @ b) * a_scale * b_scale (+ bias)``.

    The operand dtypes pick the kernel (see the module docstring for the
    pairing table); int8 operands require their dequant scale, fp8
    operands take one optionally, bf16 takes none. The result is bf16.
    """
    note_launch(a, b, a_scale, b_scale, trans_a, trans_b, bias)
    return get_module("gemm").quant_gemm(a, b, a_scale, b_scale, trans_a, trans_b, bias)


# ---------------------------------------------------------------------------
# The legacy spellings: same bindings, raw wire types (dict / list of rows).
# Reached for by astrai.extension's flat exports and by the tools that parse
# those keys; new code reads ``plan`` instead.
# ---------------------------------------------------------------------------


def set_table(rows: Rows) -> int:
    """Install the override plan rows (the experiment tier).

    ``rows`` is a row-file path, inline row text (one row per line,
    ``m_min m_max n_min n_max perf_class crosswise cta k_stages raster
    [k_tile]``), ``"-"`` to disable every row tier, or ``""`` to clear the
    override rows and fall back to injected + builtin. Returns the number
    of rows installed.
    """
    source = str(rows) if isinstance(rows, Path) else rows
    patch = {"tier": "override", "table_off": source == "-"}
    if source != "-":
        patch["rows"] = source
    return int(get_module("gemm").configure(patch)["table"]["override_rows"])


def set_planner(mode: str) -> Dict:
    """Select traffic scoring (``model``) or geometry scoring (``heuristic``).

    ``table`` requires a matching row. ``hybrid`` tries rows first and then
    geometry scoring. An empty string clears the explicit mode. The
    unconfigured default is ``model``.
    """
    if mode and mode not in PLANNER_MODES:
        raise ValueError(f"planner must be one of {PLANNER_MODES}, got {mode!r}")
    return get_module("gemm").configure({"planner": mode})


def set_log(enabled: bool = True) -> Dict:
    """Toggle the read-only ``[gemm-plan]`` launch log on stderr."""
    return get_module("gemm").configure({"log": enabled})


def set_staging(tma: Optional[bool] = None, mx: Optional[bool] = None) -> Dict:
    """Toggle the staging A/B switches (both default to enabled)."""
    patch = {}
    if tma is not None:
        patch["tma"] = tma
    if mx is not None:
        patch["mx"] = mx
    return get_module("gemm").configure(patch)


def state() -> Dict:
    """The effective configuration (planner, log, table tiers, staging)."""
    return get_module("gemm").config_state()


def probe(
    m: int,
    n: int,
    k: int,
    dt_a: torch.dtype = torch.bfloat16,
    dt_b: torch.dtype = torch.bfloat16,
    trans_a: bool = False,
    trans_b: bool = True,
    batch: int = 1,
) -> Dict:
    """The dispatch decision for one problem: ``source`` (override /
    injected / builtin / model / degraded) plus the recipe fields."""
    return get_module("gemm").plan_probe(
        m, n, k, dt_a, dt_b, trans_a=trans_a, trans_b=trans_b, batch=batch
    )


def capabilities() -> Dict:
    """Compiled GEMM implementations available on the current CUDA device."""
    return get_module("gemm").capabilities()


def facts() -> Dict:
    """The device facts the planner prices against (SMs, smem, L2, cc)."""
    return get_module("gemm").device_facts_info()


def tile_vocabulary() -> List:
    """Every (crosswise, ba, bb, cta, k_stages, k_tile) recipe the ladders
    instantiate — the sweep candidate space."""
    return get_module("gemm").tile_vocabulary()


# ---------------------------------------------------------------------------

__all__ = [
    "quant_gemm",
    "PLANNER_MODES",
    "set_table",
    "set_planner",
    "set_log",
    "set_staging",
    "state",
    "probe",
    "facts",
    "capabilities",
    "tile_vocabulary",
]
