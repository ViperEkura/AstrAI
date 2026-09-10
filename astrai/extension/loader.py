"""Dynamic discovery and loading of compiled CUDA kernel modules.

Each kernel is built by the CMake build in ``csrc/CMakeLists.txt`` into a
``.so`` placed in ``astrai/extension/lib/`` — the module name equals the
``.so`` name equals the pybind name (e.g. ``attn_decode``, defined via
``TORCH_EXTENSION_NAME``). ``KERNEL_NAMES`` is discovered automatically from
the ``.so`` files present, so adding a kernel to the CMake ``KERNELS``
registry needs no change here.

Loading is **lazy and centralized**: module names are discovered eagerly
(cheap glob), but each ``.so`` is imported on first use via the single
``get_module`` accessor, then cached. The wrapper modules (``ops/*.py``) never
touch the internals or keep their own caches — they call ``get_module(name)``
(or ``is_available(name)`` when a torch fallback is acceptable). A kernel that
failed to build (or is running on a CPU-only machine) is ``None`` in the cache,
so ``is_available`` returns ``False`` and ``get_module`` raises a clear error.
"""

import glob
import importlib
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

_LIB_DIR = os.path.join(os.path.dirname(__file__), "lib")


def _discover_kernel_names() -> List[str]:
    """Return the module names of the compiled kernel ``.so`` files in lib/."""
    names: List[str] = []
    for path in glob.glob(os.path.join(_LIB_DIR, "*.so")):
        # strip the "<soabi>.so" suffix, e.g. attn_decode.cpython-312-...so
        names.append(os.path.basename(path).split(".", 1)[0])
    return sorted(names)


KERNEL_NAMES = _discover_kernel_names()

_available: Dict[str, bool] = {}
_modules: Dict[str, object] = {}


def _try_load(name: str) -> object:
    """Import and cache the ``name`` kernel module (lazy, one attempt).

    Returns the module, or ``None`` if it is unavailable. Cached so each
    ``.so`` is imported at most once per process.
    """
    if name not in _modules:
        try:
            _modules[name] = importlib.import_module(
                f".lib.{name}", package=__package__
            )
            _available[name] = True
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
