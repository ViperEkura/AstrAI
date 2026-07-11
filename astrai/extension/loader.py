"""Dynamic discovery and loading of compiled CUDA kernel modules.

Each kernel is registered in ``csrc/build.py`` and built into a ``.so`` placed
in this package directory. On import we try to load each one; kernels that
failed to build (or are running on a CPU-only machine) are marked unavailable
so the wrapper functions can fall back to ``torch`` SDPA.
"""

import importlib
import logging

logger = logging.getLogger(__name__)

KERNEL_NAMES = ["attn_decode", "attn_prefill", "attn_paged_decode"]

_available: dict[str, bool] = {}
_modules: dict[str, object] = {}

for _name in KERNEL_NAMES:
    try:
        _mod = importlib.import_module(f".{_name}", package=__package__)
        _available[_name] = True
        _modules[_name] = _mod
    except ImportError:
        _available[_name] = False
        _modules[_name] = None


def is_available(name: str) -> bool:
    """Return ``True`` if the compiled kernel ``name`` was loaded."""
    return _available.get(name, False)


def get_module(name: str) -> object:
    """Return the loaded kernel module for ``name``, or ``None`` if unavailable."""
    return _modules.get(name)
