"""Discover and lazily load compiled modules from extension.

Each module is imported once; successful loads invalidate dispatch records.
"""

import glob
import importlib
import logging
import os
from typing import Dict, List

from astrai.extension.runtime import dispatch

logger = logging.getLogger(__name__)

_LIB_DIR = os.path.dirname(os.path.dirname(__file__))


def _discover_kernel_names() -> List[str]:
    """Return the module names of compiled kernel extension files in lib/."""
    names: List[str] = []
    for suffix in (".so", ".pyd"):
        for path in glob.glob(os.path.join(_LIB_DIR, f"_C_*{suffix}")):
            names.append(os.path.basename(path).split(".", 1)[0].removeprefix("_C_"))
    return sorted(names)


KERNEL_NAMES = _discover_kernel_names()

_available: Dict[str, bool] = {}
_modules: Dict[str, object] = {}


def _try_load(name: str) -> object:
    """Import and cache the ``name`` kernel module (lazy, one attempt).

    Returns the module, or ``None`` if it is unavailable. Cached so each
    native extension is imported at most once per process; a successful first import
    invalidates the dispatch record caches, whose availability predicates
    consult ``is_available`` (the only availability change that can happen
    without a re-registration).
    """
    if name not in _modules:
        try:
            _modules[name] = importlib.import_module(
                f"._C_{name}", package="astrai.extension"
            )
            _available[name] = True
            dispatch.invalidate()
        except ImportError:
            logger.warning("kernel '%s' failed to import; marking unavailable", name)
            _modules[name] = None
            _available[name] = False
    return _modules[name]


def is_available(name: str) -> bool:
    """Return ``True`` if the compiled kernel ``name`` could be loaded."""
    if name not in _available:
        _try_load(name)
    return _available.get(name, False)


def get_module(name: str) -> object:
    """Return the loaded kernel module for ``name``, importing it on first use.

    Raises ``RuntimeError`` if the kernel is unavailable (not built, or failed
    to import) — callers that can tolerate a torch fallback should check
    ``is_available(name)`` first instead.
    """
    mod = _try_load(name)
    if mod is None:
        raise RuntimeError(
            f"CUDA kernel '{name}' is not available. "
            f"Build with CSRC_KERNELS=true (or use the torch-native fallback)."
        )
    return mod


__all__ = [
    "KERNEL_NAMES",
    "get_module",
    "is_available",
]
