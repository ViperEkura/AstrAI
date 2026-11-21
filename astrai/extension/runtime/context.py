"""Immutable operator selections, nested scopes and process defaults."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)
_OPERATIONS = frozenset(("attention", "rotary", "swiglu", "syrk", "symm"))
_process_lock = threading.Lock()
_process_choices: Optional[Dict[str, str]] = None
_selection_revision = 0


def _seed_process() -> Dict[str, str]:
    global _process_choices
    with _process_lock:
        if _process_choices is None:
            choices = {}
            for item in os.environ.get("ASTR_OPS", "").split(","):
                item = item.strip()
                if not item:
                    continue
                operation, sep, name = item.partition("=")
                operation = operation.strip()
                if sep and operation in _OPERATIONS and name.strip():
                    choices[operation] = name.strip()
                else:
                    logger.warning("ASTR_OPS: ignoring invalid entry %r", item)
            _process_choices = choices
        return dict(_process_choices)


def set_op(operation: str, implementation: Optional[str] = None) -> None:
    """Set a soft, process-wide choice; None clears it."""

    global _process_choices
    if operation not in _OPERATIONS:
        raise ValueError(f"unknown operator family {operation!r}")
    choices = _seed_process()
    if implementation is None:
        choices.pop(operation, None)
    else:
        choices[operation] = implementation
    with _process_lock:
        _process_choices = choices


@dataclass(frozen=True)
class OpSelection:
    """One in-process snapshot; object choices retain their identity."""

    choices: Tuple[Tuple[str, Any], ...] = ()
    process_choices: Optional[Tuple[Tuple[str, str], ...]] = None

    def get(self, operation: str) -> Optional[Any]:
        return dict(self.choices).get(operation)

    def choice(self, operation: str) -> Tuple[Optional[Any], str]:
        value = self.get(operation)
        if value is not None:
            return value, "context"
        value = self.process().get(operation)
        return (value, "env") if value is not None else (None, "chain")

    def process(self) -> Dict[str, str]:
        return dict(self.process_choices or ())

    def with_choices(self, choices: Mapping[str, Any]) -> "OpSelection":
        merged = dict(self.choices)
        merged.update(choices)
        return OpSelection(tuple(merged.items()), self.process_choices)


class OpContext:
    """Own selection scopes; snapshots can be installed in a worker."""

    def __init__(self) -> None:
        self._current: ContextVar[Optional[OpSelection]] = ContextVar(
            "astrai_op_selection", default=None
        )

    def _state(self) -> OpSelection:
        return self._current.get() or OpSelection()

    def snapshot(self) -> OpSelection:
        state = self._state()
        if state.process_choices is None:
            return replace(state, process_choices=tuple(_seed_process().items()))
        return state

    @contextmanager
    def scope(self, **choices: Any) -> Generator:
        unknown = choices.keys() - _OPERATIONS
        if unknown:
            raise ValueError(f"unknown operator family {sorted(unknown)[0]!r}")
        token = self._current.set(self._state().with_choices(choices))
        try:
            yield
        finally:
            self._current.reset(token)

    @contextmanager
    def install(self, snapshot: OpSelection) -> Generator:
        if not isinstance(snapshot, OpSelection):
            raise TypeError("snapshot must be an OpSelection")
        token = self._current.set(snapshot)
        try:
            yield
        finally:
            self._current.reset(token)


op_context = OpContext()
op_backend = op_context.scope
snapshot_op_selection = op_context.snapshot
use_op_selection = op_context.install


def invalidate_selection_cache() -> None:
    """Advance the token after a compiled kernel becomes available."""
    global _selection_revision
    _selection_revision += 1


def selection_cache_token(operation: str) -> Optional[int]:
    snapshot = op_context.snapshot()
    process = snapshot.process()
    if snapshot.get(operation) is not None or process.get(operation) is not None:
        return None
    return _selection_revision
