"""Attention strategies with registered priorities and per-call capabilities.

Training uses ``fwd=None``; prefill/decode are inference. Context and process
selection share the generic operator dispatcher. Inputs use BLHD layout.
"""

import enum
import functools
import inspect
import logging
import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch as _torch
from torch import Tensor

from astrai.extension.runtime.dispatch import (
    Axes,
    ImplRecord,
    Spec,
    env_selection,
    get_override,
    invalidate,
    register_env_alias,
    register_family,
    reset_override,
    set_override,
    tensor_axes,
)
from astrai.extension.runtime.dispatch import (
    resolve as _dispatch_resolve,
)
from astrai.factory import BaseFactory

try:
    import flash_attn as _flash_attn
except Exception:
    _flash_attn = None

if TYPE_CHECKING:
    from astrai.model.kv_cache import KVCache

logger = logging.getLogger(__name__)


_singletons: Dict[type, "AttentionBackend"] = {}


@functools.lru_cache(maxsize=1)
def flash_attn_available() -> bool:
    if not _torch.cuda.is_available():
        return False
    fa = _flash_attn
    if fa is None:
        return False

    try:
        major = int(fa.__version__.split(".")[0])
        cc = _torch.cuda.get_device_capability()
        cc_num = cc[0] * 10 + cc[1]
    except Exception:
        major, cc_num = 0, 0
    if (major >= 3 and cc_num < 90) or (major < 3 and 0 < cc_num < 70):
        return False

    try:
        if not hasattr(fa, "flash_attn_func"):
            return False
        x = _torch.zeros(1, 1, 1, 64, device="cuda", dtype=_torch.bfloat16)
        out = fa.flash_attn_func(x, x, x, causal=True)
        return bool(_torch.isfinite(out).all().item())
    except Exception:
        return False


class ATTN_BACKEND(enum.Enum):
    """Backend selector enum, mirroring ``torch.nn.attention.SDPBackend``."""

    TORCH_NATIVE = "torch_native"
    CUDA = "cuda"
    FLASH = "flash"


def _instance(backend_cls: type) -> "AttentionBackend":
    """Return the canonical singleton instance for a backend class.

    Backends hold no per-instance state, so a single cached instance is
    safe and avoids per-call allocation on the attention hot path.
    """
    backend = _singletons.get(backend_cls)
    if backend is None:
        backend = backend_cls()
        _singletons[backend_cls] = backend
    return backend


@functools.lru_cache(maxsize=1)
def _priority_backends() -> Tuple["AttentionBackend", ...]:
    """Available registered backends, ordered by priority."""
    return tuple(
        _instance(cls)
        for cls in sorted(
            AttentionBackendFactory._entries.values(), key=lambda cls: cls.priority
        )
        if cls.available()
    )


def _resolve_default_backend() -> "AttentionBackend":
    """Pick the highest-priority available backend (cuda -> flash -> torch).

    Resolved lazily on first use and cached via ``_priority_backends``.
    Per-call capability fallback happens in ``attention()``, so the
    default is safe for training and fp32 models.
    """
    return _priority_backends()[0]


def _resolve_backend(
    backend: Optional[Union[str, ATTN_BACKEND, "AttentionBackend", type]] = None,
) -> "AttentionBackend":
    """Resolve a backend configuration to its canonical instance.

    Accepts a registered name, ``ATTN_BACKEND`` enum value, backend class,
    or instance.  Names/classes resolve to the shared singleton; a caller
    may still pass its own instance to opt out of sharing.
    """
    if backend is not None:
        if isinstance(backend, ATTN_BACKEND):
            return _instance(AttentionBackendFactory.get_component_class(backend.value))
        if isinstance(backend, str):
            return _instance(AttentionBackendFactory.get_component_class(backend))
        if isinstance(backend, type) and issubclass(backend, AttentionBackend):
            return _instance(backend)
        if isinstance(backend, AttentionBackend):
            return backend
        raise TypeError(
            f"expected a registered name, ATTN_BACKEND, AttentionBackend type, "
            f"or instance, got {type(backend).__name__}"
        )
    return _resolve_default_backend()


_ENV_WARNED: set = set()


def _environment_backend() -> Optional["AttentionBackend"]:
    """Resolve the process-wide env override (``ASTR_OPS`` or the legacy
    ``ASTR_BACKEND``) to a backend instance, if it names a registered one.

    Invalid names warn once and are ignored, falling back to default
    resolution — the override is soft, never fatal.
    """
    name = env_selection("attention")
    if name is None:
        return None
    try:
        return _resolve_backend(name)
    except (ValueError, RuntimeError):
        message = (
            f"ASTR_BACKEND/ASTR_OPS value {name!r} is not a registered "
            f"attention backend; falling back to default resolution"
        )
        if message not in _ENV_WARNED:
            _ENV_WARNED.add(message)
            logger.warning(message)
        return None


def get_backend(
    use_default: bool = True,
) -> Optional["AttentionBackend"]:
    """Resolve the active backend: explicit context > env > default.

    An ``attn_backend(...)`` context is the caller's explicit choice and
    always wins.  ``ASTR_BACKEND`` (or ``ASTR_OPS``) is a process-wide
    override consulted only when no context is set.  Pass
    ``use_default=False`` at request submission to retain only an
    environment override or the caller's :func:`attn_backend` value.
    """
    context_backend = get_override("attention")
    if context_backend is not None:
        return context_backend
    env_backend = _environment_backend()
    if env_backend is not None:
        return env_backend
    return _resolve_default_backend() if use_default else None


@contextmanager
def attn_backend(backend: Union[str, ATTN_BACKEND, "AttentionBackend", type]):
    """Context manager to select an attention backend.

    Mirrors ``torch.nn.attention.sdpa_kernel``. Accepts an
    registered name, ``ATTN_BACKEND`` enum value, backend class, or instance.

    Examples::

        with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
            ...
        with attn_backend(TorchNativeBackend):
            ...
        with attn_backend(TorchNativeBackend()):
            ...
    """
    instance = _resolve_backend(backend)
    token = set_override("attention", instance)
    try:
        yield instance
    finally:
        reset_override(token)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand KV heads to match Q heads for GQA."""
    if n_rep == 1:
        return x
    n_heads, head_dim = x.shape[-2:]
    return (
        x.unsqueeze(-2)
        .expand(*x.shape[:-2], n_heads, n_rep, head_dim)
        .reshape(*x.shape[:-2], n_heads * n_rep, head_dim)
    )


def _axes(
    q: Tensor,
    kv_cache: Optional["KVCache"],
    attn_mask: Optional[Tensor],
    is_causal: bool,
    fwd: Optional[str],
    scale: Optional[float] = None,
) -> Axes:
    """Snapshot the axes the attention decision table depends on."""
    return tensor_axes(
        q,
        mode="train" if fwd is None else "infer",
        fwd=fwd,
        ndim=q.dim(),
        head_dim=q.size(-1) if q.dim() >= 1 else None,
        has_cache=kv_cache is not None,
        has_mask=attn_mask is not None,
        scale=scale,
        _call=(q, kv_cache, attn_mask, is_causal, fwd),
    )


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Optional["KVCache"] = None,
    layer_id: int = 0,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    fwd: Optional[str] = None,
    backend: Optional[Union[str, ATTN_BACKEND, "AttentionBackend", type]] = None,
    *,
    scale: Optional[float] = None,
) -> Tensor:
    """Select a capable backend and run attention on projected BLHD tensors."""
    if scale is not None and not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
    explicit = _resolve_backend(backend) if backend is not None else None
    resolution = _dispatch_resolve(
        "attention",
        q,
        kv_cache,
        attn_mask,
        is_causal,
        fwd,
        explicit=explicit,
        **({"scale": scale} if scale is not None else {}),
    )
    args = (q, k, v, kv_cache, layer_id, attn_mask, is_causal, fwd)
    if scale is None:
        return resolution.record.obj.forward(*args)
    return resolution.record.obj.forward(*args, scale=scale)


@functools.cache
def _supports_call_accepts_scale(backend_cls: type) -> bool:
    """Preserve legacy five-argument probes while forwarding supported scale."""
    parameters = inspect.signature(backend_cls.supports_call).parameters
    scale = parameters.get("scale")
    return (
        scale is not None
        and scale.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ) or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())


class AttentionBackend(ABC):
    """Register subclasses with ``AttentionBackendFactory.register(name)``.

    Declare ``priority`` and ``modes``; implement availability, capability,
    and execution. Lower priorities run first.
    """

    priority = 50
    supports_scale = False
    modes = frozenset(("train", "infer"))

    @classmethod
    def supports_axes(cls, ax: Axes) -> bool:
        """Default adapter for third-party backends using supports_call."""
        scale = ax.get("scale")
        if scale is None:
            return _instance(cls).supports_call(*ax["_call"])
        if not cls.supports_scale:
            return False
        if _supports_call_accepts_scale(cls):
            return _instance(cls).supports_call(*ax["_call"], scale=scale)
        return _instance(cls).supports_call(*ax["_call"])

    def __enter__(self) -> "AttentionBackend":
        self._token = set_override("attention", self)
        return self

    def __exit__(self, *exc) -> None:
        reset_override(self._token)

    @classmethod
    @abstractmethod
    def available(cls) -> bool:
        """Check machine-level dependencies."""

    @abstractmethod
    def supports_call(
        self,
        q: Tensor,
        kv_cache: Optional["KVCache"],
        attn_mask: Optional[Tensor],
        is_causal: bool,
        fwd: Optional[str],
    ) -> bool:
        """Check shape, dtype, cache, and mask without side effects."""
        return True

    @staticmethod
    def _check_fwd(fwd: Optional[str]) -> None:
        """Reject unknown forward modes loudly."""
        if fwd not in (None, "prefill", "decode"):
            raise ValueError(f"unsupported attention forward mode: {fwd}")

    @abstractmethod
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        fwd: Optional[str] = None,
    ) -> Tensor:
        """Return attention output; ``fwd`` selects train/prefill/decode."""

    @staticmethod
    def supports_graph() -> bool:
        """Report CUDA-graph capture support."""
        return False


class AttentionBackendFactory(BaseFactory[AttentionBackend]):
    """Register attention strategies and refresh dispatch on new entries."""

    @classmethod
    def register(cls, name):
        decorate = super().register(name)

        def add(backend_cls):
            result = decorate(backend_cls)
            _priority_backends.cache_clear()
            invalidate("attention")
            return result

        return add


from .cuda import CudaBackend
from .flash import FlashAttnBackend
from .torch import TorchNativeBackend


def _attention_records() -> list:
    return [
        ImplRecord(
            family="attention",
            name=name,
            obj=_instance(cls),
            spec=Spec.of(
                cls.supports_axes,
                f"{name} supports call",
            ),
            available=cls.available,
            priority=cls.priority,
            modes=cls.modes,
        )
        for name, cls in AttentionBackendFactory._entries.items()
    ]


def _reference_record() -> ImplRecord:
    return ImplRecord(
        family="attention",
        name=ATTN_BACKEND.TORCH_NATIVE.value,
        obj=_instance(TorchNativeBackend),
        spec=Spec.always(),
        priority=999,
    )


register_family("attention", _axes, _attention_records, _reference_record)
register_env_alias("attention", "ASTR_BACKEND")
