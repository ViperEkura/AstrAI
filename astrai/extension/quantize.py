"""Quantization: every scheme's policy and integration in one module.

The policy layer over the kernel side's single quantize family
(``csrc/kernels/quantize/`` — the fp8 quantize kernels plus the int8
dequant the GEMM family consumes). Stateless kernel adapters live one
layer down (``ops/quantize.py`` / ``ops/gemm.py`` — the only modules
touching the pybind).

INT8 (inference, stateless strategies consumed with ``ops.gemm``'s
``mm_w8*`` primitives):

- ``quantize_weight_int8`` — symmetric per-channel weight quantization
  (one-shot at load time), ``w ≈ w8 * scale[:, None]``
- ``quantize_act_int8`` — symmetric per-row dynamic activation quantization

The dequant contract lives in the GEMM family: int8 operands expand to
bf16 fragments exactly in-register, scales re-apply multiplicatively in
the epilogue, so the only error is the round-to-nearest here. There is
deliberately no nn.Module layer on this path.

FP8 (training stack):

- ``FP8Recipe`` — scaling recipes (TE-style delayed scaling over an amax
  history window, or dynamic current-amax scaling)
- ``gemm.fp8_linear`` (``csrc/kernels/gemm/fp8_linear.cu``) — the
  composed fwd+bwd: quantize, ring fold/advance, the weight-cast cache
  and all three GEMMs run in C++ behind a single Python->C++ crossing
- ``fp8_autocast`` — the torch.autocast-style context (plus the
  ``fp8_linear_enable`` global switch) routing ``aten::linear`` through
  that op; hybrid E4M3 forward / E5M2 backward by default
- ``fp8_state_dict``/``fp8_load_state_dict`` — checkpoint snapshots of
  the delayed-scaling rings (delegating to the C++ state)

Usage::

    from astrai.extension.quantize import fp8_autocast
    with fp8_autocast(enabled=True, fp8_format="hybrid"):
        logits = model(input_ids)
    loss.backward()  # fp8 backward runs anywhere; fwd captured state on the node

The context mirrors ``torch.autocast``: the active
``(enabled, recipe, fp8_format)`` triple is thread-local (a ``contextvars``
``ContextVar``, absent outside any region), and the manager is class-based and
reentrant with nested ``enabled=False`` disabling dispatch inside it. The fp8
path targets *training*: x/g are quantized fresh every call, while the weight
cast is reused until the weight's version counter moves (an optimizer step),
so a gradient-accumulation loop quantizes w once per step; the per-operand
scales come from the delayed/dynamic recipe, and under delayed scaling the
publishing kernel hands the host both the next scale and its reciprocal.

The ``aten::linear`` CUDA/AutogradCUDA override installs **lazily**, on the
first activation (autocast enter or the global enable): importing this
module — for the int8 strategies or anything else — never touches the
dispatcher.
"""

import functools
import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass

import torch
from torch.library import Library

from astrai.extension.loader import get_module, is_available

# ---------------------------------------------------------------------------
# INT8: stateless strategies (inference)
# ---------------------------------------------------------------------------


def quantize_weight_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel int8 quantization of a linear weight.

    ``w`` is ``[N, K]`` (the nn.Linear convention); returns
    ``(w8 int8 [N, K] contiguous, scale f32 [N])`` with
    ``w ≈ w8.float() * scale[:, None]`` and ``scale = amax(N) / 127``.
    Quantization runs in float32 regardless of the source dtype.
    """
    wf = w.detach().to(torch.float32)
    scale = wf.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
    q = torch.round(wf / scale.unsqueeze(-1)).clamp_(-127, 127)
    return q.to(torch.int8).contiguous(), scale.contiguous()


def quantize_act_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row dynamic int8 quantization of activations.

    ``x`` is ``[..., K]``; returns ``(q int8 with x's shape, scale f32
    [prod(leading dims)])`` — the scale layout ``quant_gemm``'s per-row a_scale expects once the
    leading dims flatten.
    """
    xf = x.detach().to(torch.float32)
    x2 = xf.reshape(-1, xf.shape[-1])
    scale = x2.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
    q = torch.round(x2 / scale.unsqueeze(-1)).clamp_(-127, 127)
    return q.to(torch.int8).reshape(x.shape), scale


# ---------------------------------------------------------------------------
# FP8: formats, recipes, region config
# ---------------------------------------------------------------------------

# FP8 format vocabulary: the canonical key is the fp8 dtype itself
# (torch.float8_e4m3fn / torch.float8_e5m2 — what the quantize binding
# dispatches and validates on). The only policy-level notion beyond a
# dtype is the hybrid per-direction pair, carried as a plain (fwd, bwd)
# tuple; dtype validation stays in the C++ binding.


def _is_fp8(dtype: torch.dtype) -> bool:
    """A pre-quantized weight takes the GEMM directly (no re-quantize)."""
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def fp8_format_pair(fmt: str | torch.dtype) -> tuple[torch.dtype, torch.dtype]:
    """Format spec -> (fwd, bwd) fp8 dtype pair.

    A dtype is symmetric; ``'hybrid'`` is E4M3 forward / E5M2 backward
    (the training default).
    """
    if fmt == "hybrid":
        return (torch.float8_e4m3fn, torch.float8_e5m2)
    return (fmt, fmt)


@dataclass
class FP8Recipe:
    """Scale-from-amax policy knobs: ``scale = (amax / finfo(fmt).max) / 2^margin``.

    ``dynamic=False`` (default) is TE-style delayed scaling: max over the
    amax history window (amax from *previous* steps; the window trades
    responsiveness against stability). ``dynamic=True`` is current-amax
    scaling (torchao DYNAMIC): measure, then quantize — no history, at an
    extra pass. The formula itself lives in the C++ op (the single home;
    the seed and in-kernel fold both apply it there).
    """

    history_len: int = 16
    margin: int = 0
    dynamic: bool = False


@dataclass
class _ActiveConfig:
    """The immutable (enabled, recipe, format-pair) triple of one open
    region; ``fp8_format`` is the (fwd, bwd) fp8 dtype pair."""

    enabled: bool
    recipe: FP8Recipe
    fp8_format: tuple[torch.dtype, torch.dtype]


# Thread-local active configuration (torch's autocast TLS analog): set by
# fp8_autocast on __enter__, absent outside any region. Autograd engine
# threads run backwards with their own empty context — fine, since backward
# only reads state captured on ctx at forward time.
_active_config: ContextVar[_ActiveConfig | None] = ContextVar(
    "astrai_fp8_active_config", default=None
)

# Persistent out-of-region defaults. Everything else the old Python state
# machine owned — the per-weight meta registry (delayed-scaling rings, the
# version-keyed weight-cast cache, the checkpoint pending queue, the
# generation counter invalidating that cache) — lives in the C++ op's
# translation unit (``csrc/kernels/gemm/fp8_linear.cu``), keyed the
# same way (data_ptr, shape, dtype) with the same registration-order
# snapshot contract.
_default_enabled = False
_default_recipe = FP8Recipe()
_default_format = (torch.float8_e4m3fn, torch.float8_e5m2)


def fp8_state_dict() -> dict:
    """Checkpoint snapshot of the delayed-scaling rings (A1): save it beside
    the optimizer state. Empty under dynamic scaling (no rings exist) and on
    a box without the extension (bf16 training never allocated any)."""
    if not is_available("gemm"):
        return {"version": 1, "entries": []}
    return get_module("gemm").fp8_state_dict()


def fp8_load_state_dict(sd: dict) -> None:
    """Restore a :func:`fp8_state_dict` snapshot. Unmatched entries (a model
    shape/topology change across the checkpoint) re-seed on next use — the
    same resume transient a ring-less restart would pay on every step."""
    if not sd.get("entries"):
        return
    get_module("gemm").fp8_load_state_dict(sd)


def fp8_reset() -> None:
    """Drop the C++-side registry (tests / reconfiguration)."""
    if is_available("gemm"):
        get_module("gemm").fp8_reset()


def _active() -> _ActiveConfig | None:
    """The active config when fp8 dispatch is on, else ``None`` (fast guard).

    A region config wins (honoring nested ``enabled=False`` regions); with no
    region open this falls back to the persistent global switch
    (``fp8_linear_enable``), so that flag still routes aten::linear to fp8.
    """
    cfg = _active_config.get()
    if cfg is not None:
        return cfg if cfg.enabled else None
    if _default_enabled:
        return _ActiveConfig(True, _default_recipe, _default_format)
    return None


def _current_config() -> _ActiveConfig:
    """Like ``_active()`` but always returns a config (disabled regions and
    out-of-region direct calls resolve to the global defaults)."""
    cfg = _active_config.get()
    if cfg is not None:
        return cfg
    return _ActiveConfig(_default_enabled, _default_recipe, _default_format)


class fp8_autocast:
    """Autocast-style context: fp8 linear dispatch on this thread.

    Mirrors ``torch.autocast`` — a class-based, reentrant, nestable context
    over thread-local state::

        with fp8_autocast(enabled=True, fp8_format="hybrid"):
            logits = model(input_ids)   # aten::linear -> fp8 path
        loss.backward()  # fp8 backward; state was captured at forward time

    Nesting follows torch: each ``__enter__`` pushes the new active config, each
    ``__exit__`` restores the previous one, and a nested ``enabled=False`` region
    simply disables dispatch inside it. The instance doubles as a decorator.
    """

    def __init__(
        self,
        enabled: bool = True,
        update_interval: int = 16,
        recipe: FP8Recipe | None = None,
        fp8_format: str | torch.dtype = "hybrid",
        margin: int = 0,
    ):
        if recipe is None:
            recipe = FP8Recipe(history_len=update_interval, margin=margin)
        self._config = _ActiveConfig(bool(enabled), recipe, fp8_format_pair(fp8_format))
        self._tokens: list[Token] = []

    def __enter__(self) -> "fp8_autocast":
        if self._config.enabled:
            _install_linear_override()
        self._tokens.append(_active_config.set(self._config))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        token = self._tokens.pop()
        _active_config.reset(token)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate


def fp8_linear_enable(enabled: bool = True) -> None:
    """Toggle fp8 dispatch for aten::linear globally (the out-of-region default;
    ``fp8_autocast`` regions override it thread-locally)."""
    global _default_enabled
    if enabled:
        _install_linear_override()
    _default_enabled = bool(enabled)


def fp8_linear_enabled() -> bool:
    """Whether fp8 dispatch is active right now (region config or global)."""
    return _active() is not None


def _fp8_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Shape guard for the fp8 path. Unlike a strict 16-alignment requirement,
    the kernels handle unaligned M/N via boundary checks (slower but correct) —
    so no whole-call bf16 fallback for small decode batches. Only the K-dimension
    contraction must match and the weight must be 2D."""
    return x.dim() >= 2 and w.dim() == 2 and x.size(-1) == w.size(1)


def _linear_cuda_impl(x: torch.Tensor, w: torch.Tensor, bias=None):
    cfg = _active()
    if (
        cfg is not None
        and x.dtype is torch.bfloat16
        and w.dtype is torch.bfloat16
        and _fp8_supported(x, w)
    ):
        # One Python->C++ crossing per linear: quantize, the ring
        # fold/advance, the weight-cast cache and all three GEMMs run inside
        # gemm.fp8_linear. The caller's grad mode gates the delayed-scaling
        # bookkeeping, and it must be read HERE: inside a Function forward
        # grad is always disabled, so the mode would be invisible one frame
        # deeper. A no-grad linear (checkpointing recompute, inference) reads
        # the rings without folding or advancing them. This host bool is also
        # the seam a CUDA-graph device flag replaces (TE's
        # skip_fp8_weight_update).
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
        )
    return torch.ops.aten.linear.default.redispatch(
        torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeImplicitAutograd),
        x,
        w,
        bias,
    )


_linear_libs: list[Library] | None = None
_install_lock = threading.Lock()


def _install_linear_override() -> None:
    """Register the fp8 aten::linear impls once, on first activation.

    Importing this module must stay dispatcher-neutral: the override routes
    every CUDA ``aten::linear`` call through ``_linear_cuda_impl``'s guard,
    so it is installed exactly when fp8 dispatch is first switched on (an
    autocast enter or the global enable) — never at import. The ``Library``
    handles live in a module global for the process lifetime (dropping them
    would unregister the impls).
    """
    global _linear_libs
    if _linear_libs is None:
        with _install_lock:
            if _linear_libs is None:
                lib = Library("aten", "IMPL", "CUDA")
                lib.impl("linear", _linear_cuda_impl)
                # Also replace torch's generated linear autograd formula
                # (which would call aten::linear_backward after the
                # fp8_autocast region exits). The fp8 backward is owned by
                # the C++ node with state captured at forward time, so
                # loss.backward() works wherever it is called; the CUDA
                # registration still covers inference_mode.
                lib_autograd = Library("aten", "IMPL", "AutogradCUDA")
                lib_autograd.impl("linear", _linear_cuda_impl)
                _linear_libs = [lib, lib_autograd]
