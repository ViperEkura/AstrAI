"""Thread-local FP8 routing and checkpoint state access."""

import functools
import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.library import Library

from astrai.extension.policy.quantization.fp8_slots import (
    Fp8Slot,
    SlotTable,
    assign_slots,
    clear_slots,
    fp8_slot_of,
    refresh_slots,
    sync_slots,
)
from astrai.extension.runtime.loader import get_module, is_available

__all__ = [
    "FP8Recipe",
    "Fp8Slot",
    "SlotTable",
    "assign_slots",
    "clear_slots",
    "fp8_autocast",
    "fp8_format_pair",
    "fp8_linear_enable",
    "fp8_linear_enabled",
    "fp8_load_state_dict",
    "fp8_reset",
    "fp8_slot_of",
    "fp8_state_dict",
    "refresh_slots",
    "sync_slots",
]


def fp8_format_pair(fmt: Union[str, torch.dtype]) -> Tuple[torch.dtype, torch.dtype]:
    """Return forward/backward dtypes; hybrid uses E4M3/E5M2."""
    if fmt == "hybrid":
        return (torch.float8_e4m3fn, torch.float8_e5m2)
    return (fmt, fmt)


@dataclass
class FP8Recipe:
    """Delayed scaling uses prior amax history; dynamic scaling uses current amax."""

    history_len: int = 16
    margin: int = 0
    dynamic: bool = False


@dataclass
class _ActiveConfig:
    enabled: bool
    recipe: FP8Recipe
    fp8_format: Tuple[torch.dtype, torch.dtype]


# Backward reads state captured during forward, not the current context.
_active_config: ContextVar[Optional[_ActiveConfig]] = ContextVar(
    "astrai_fp8_active_config", default=None
)

# Defaults used outside an explicit FP8 region.
_default_enabled = False
_default_recipe = FP8Recipe()
_default_format = (torch.float8_e4m3fn, torch.float8_e5m2)


def fp8_state_dict() -> Dict:
    """Export delayed-scaling state; return empty state without the extension."""
    if not is_available("gemm"):
        return {"version": 1, "entries": []}
    return get_module("gemm").fp8_state_dict()


def fp8_load_state_dict(sd: Dict) -> None:
    """Restore scaling history; unmatched entries initialize on next use."""
    if not sd.get("entries"):
        return
    get_module("gemm").fp8_load_state_dict(sd)


def fp8_reset() -> None:
    """Drop the C++-side registry (tests / reconfiguration)."""
    if is_available("gemm"):
        get_module("gemm").fp8_reset()


def _active() -> Optional[_ActiveConfig]:
    """An active region overrides the global default, including enabled=False."""
    cfg = _active_config.get()
    if cfg is not None:
        return cfg if cfg.enabled else None
    if _default_enabled:
        return _ActiveConfig(True, _default_recipe, _default_format)
    return None


class fp8_autocast:
    """Nestable FP8 context and decorator. Backward uses state saved by forward."""

    def __init__(
        self,
        enabled: bool = True,
        update_interval: int = 16,
        recipe: Optional[FP8Recipe] = None,
        fp8_format: Union[str, torch.dtype] = "hybrid",
        margin: int = 0,
    ):
        if recipe is None:
            recipe = FP8Recipe(history_len=update_interval, margin=margin)
        self._config = _ActiveConfig(bool(enabled), recipe, fp8_format_pair(fp8_format))
        self._tokens: List[Token] = []

    def __enter__(self) -> "fp8_autocast":
        if self._config.enabled:
            _install_linear_override()
            # Refresh identities after parameter replacement to retain scaling history.
            refresh_slots()
        self._tokens.append(_active_config.set(self._config))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        token = self._tokens.pop()
        _active_config.reset(token)
        if self._config.enabled and is_available("gemm"):
            get_module("gemm").fp8_clear_act_cache()
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate


def fp8_linear_enable(enabled: bool = True) -> None:
    """Set the global default; active regions override it locally."""
    global _default_enabled
    if enabled:
        _install_linear_override()
    _default_enabled = bool(enabled)


def fp8_linear_enabled() -> bool:
    """Whether FP8 dispatch is active in this context."""
    return _active() is not None


def _fp8_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    """K must match; kernels handle unaligned M/N dimensions."""
    return x.dim() >= 2 and w.dim() == 2 and x.size(-1) == w.size(1)


def _linear_cuda_impl(x: torch.Tensor, w: torch.Tensor, bias=None):
    cfg = _active()
    if (
        cfg is not None
        and x.dtype is torch.bfloat16
        and w.dtype is torch.bfloat16
        and _fp8_supported(x, w)
    ):
        # Read grad mode before entering the C++ autograd node.
        # No-grad forwards must not advance delayed-scaling history.
        recipe = cfg.recipe
        return get_module("gemm").fp8_linear(
            x,
            w,
            bias,
            torch.is_grad_enabled(),  # update_rings
            bool(bias is not None and bias.requires_grad),
            recipe.dynamic,
            recipe.history_len,
            recipe.margin,
            cfg.fp8_format[0],
            cfg.fp8_format[1],
            fp8_slot_of(w),
        )
    return torch.ops.aten.linear.default.redispatch(
        torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeImplicitAutograd),
        x,
        w,
        bias,
    )


_linear_libs: Optional[List[Library]] = None
_install_lock = threading.Lock()


def _install_linear_override() -> None:
    """Install on first activation, not import; retain Library handles to keep registrations alive."""
    global _linear_libs
    if _linear_libs is None:
        with _install_lock:
            if _linear_libs is None:
                lib = Library("aten", "IMPL", "CUDA")
                lib.impl("linear", _linear_cuda_impl)
                # Let the C++ node own backward after the FP8 context exits.
                lib_autograd = Library("aten", "IMPL", "AutogradCUDA")
                lib_autograd.impl("linear", _linear_cuda_impl)
                _linear_libs = [lib, lib_autograd]
