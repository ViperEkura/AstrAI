"""Shared fixtures for extension tests."""

import pytest

import astrai.extension.runtime.dispatch as dispatch
from astrai.extension import kernel
from astrai.extension.runtime.loader import is_available


@pytest.fixture(autouse=True)
def _reset_dispatch_state():
    """Isolate the process-level dispatch/planner state per test.

    set_op selections, cached record lists and the gemm planner
    configuration are process state; without this a test that touches
    them leaks into the next one.
    """
    previous_selection = dispatch._selection
    try:
        yield
    finally:
        dispatch._selection = previous_selection
        dispatch.invalidate()
        if is_available("gemm"):
            kernel.gemm.set_table("")
            kernel.gemm.set_planner("")  # back to the shipped default
            # staging too: the planner prices per staging variant (gemm.cuh
            # cost_of branches on q.tma), so a test leaving tma disabled would
            # silently move every later probe to the cp.async cost form
            kernel.gemm.set_staging(tma=True, mx=True)
