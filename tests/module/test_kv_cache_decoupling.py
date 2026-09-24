"""KV cache decoupling between the model and inference packages.

The physical KV cache buffers (``KVCache`` / ``KVStorage`` / ``ReqToTokenPool``)
used to live at ``astrai/inference/cache/buffer.py`` while the attention layer
imported them, which made ``astrai.model`` import the inference package through
its ``__init__``.  That cycle
(``model -> inference.cache -> inference/__init__ -> engine -> model``) only
survived because ``astrai/model/__init__.py`` happened to bind ``AutoModel``
before the chain was triggered; reaching the model package through any other
entry point raised
``ImportError: cannot import name 'AutoModel' from partially initialized module
'astrai.model.automodel' (most likely due to a circular import)``.

The buffers now live in :mod:`astrai.model.kv_cache` and the inference side
imports them, so the dependency runs one way (inference → model).  These tests
pin that direction down.
"""

import ast
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "astrai" / "model"


def _imported_modules(path: Path) -> set[str]:
    """Absolute module names imported by one source file.

    Relative imports are resolved against the file's package path so
    ``from ..inference.cache import KVCache`` is caught as
    ``astrai.inference.cache``.
    """
    package = ["astrai"] + list(path.relative_to(_REPO_ROOT / "astrai").parts[:-1])
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    modules.add(node.module)
                continue
            base = package[: len(package) - (node.level - 1)]
            if node.module:
                base = base + node.module.split(".")
            modules.add(".".join(base))
    return modules


def test_model_sources_import_nothing_from_inference():
    """Direct coupling guard: no module under ``astrai/model/`` may import the
    inference package."""
    offenders = {
        str(path.relative_to(_REPO_ROOT)): sorted(
            name
            for name in _imported_modules(path)
            if name == "astrai.inference" or name.startswith("astrai.inference.")
        )
        for path in sorted(_MODEL_DIR.rglob("*.py"))
    }
    offenders = {path: names for path, names in offenders.items() if names}

    assert not offenders, f"model package imports inference: {offenders}"


def test_model_package_imports_without_inference_package():
    """Transitive coupling guard: importing the model package must not pull any
    ``astrai.inference`` module.

    The probe stubs the top-level ``astrai`` package so the model package is
    imported through the same entry shape that used to fail — it reproduces the
    historical circular-import crash rather than relying on
    ``astrai/__init__.py``'s import order.
    """
    probe = (
        "import importlib, sys, types\n"
        f"stub = types.ModuleType('astrai'); stub.__path__ = [{str(_REPO_ROOT / 'astrai')!r}]\n"
        "sys.modules['astrai'] = stub\n"
        "importlib.import_module('astrai.model')\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('astrai.inference'))\n"
        "print('PROBE:' + repr(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=_REPO_ROOT,
    )
    payload = next(
        line for line in result.stdout.splitlines() if line.startswith("PROBE:")
    )
    assert payload.removeprefix("PROBE:") == "[]"


def test_cache_facade_reexports_the_model_buffers():
    """``astrai.inference.cache`` stays a working facade over the buffers'
    new home — same objects, not copies."""
    import astrai.inference.cache as cache_facade
    import astrai.model.kv_cache as kv_cache

    assert cache_facade.KVCache is kv_cache.KVCache
    assert cache_facade.KVStorage is kv_cache.KVStorage
    assert cache_facade.ReqToTokenPool is kv_cache.ReqToTokenPool
