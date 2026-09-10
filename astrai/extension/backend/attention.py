"""Attention backend abstraction with context-manager switching.

The backend encapsulates KV cache I/O and attention computation. The
attention module (GQA/MLA) keeps projections, rotary, QK-norm, gating,
and output projection; the backend handles everything from "write K/V
to cache" through "SDPA output".

Usage — mirroring ``torch.nn.attention.sdpa_kernel``:

    from astrai.extension import attn_backend, ATTN_BACKEND

    with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
        engine.generate("hello")

    # or with an instance:
    with attn_backend(TorchNativeBackend()):
        ...

    # or the shorthand (instance is itself a context manager):
    with TorchNativeBackend():
        ...

Thread-safe via ``contextvars`` — each scheduler thread gets its own
active backend. Backend resolution is a thin facade over the generic
operator dispatcher (``astrai.extension.dispatch``): the three backends
are registered as the "attention" family and the decision table lives in
``_attention_records``.  Resolution follows a strict precedence:

1. explicit ``attn_backend(...)`` context (wins over everything),
2. the process-wide ``ASTR_BACKEND`` environment override
   (or an ``ASTR_OPS`` ``attention=`` entry, which wins over the legacy
   variable),
3. an implicit default picked from the available backends
   (cuda > flash > torch).

Capability is polymorphic: every backend declares ``available()``
(machine-level) and ``supports_call(...)`` (per-call), so adding a new
backend requires no changes to the resolution logic. Training calls
(``fwd=None``, no KV cache) resolve through the same priority list: the
CUDA cache kernels cannot run without a cache, so they fall back to
flash (when it can handle the call — mask-free/causal only) and finally
to the reference ``TorchNativeBackend``.

Layout convention: all q/k/v are ``[batch, seq_len, n_heads, head_dim]``
(blhd). The backend returns ``[batch, seq_len, n_heads * head_dim]``.
"""

import enum
import functools
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.dispatch import (
    Axes,
    ImplRecord,
    Spec,
    axis,
    env_selection,
    get_override,
    register_env_alias,
    register_family,
    reset_override,
    set_override,
    tensor_axes,
)
from astrai.extension.dispatch import (
    resolve as _dispatch_resolve,
)
from astrai.extension.loader import is_available
from astrai.extension.ops.attention import (
    attn_paged_decode,
    attn_paged_prefill,
)
from astrai.factory import BaseFactory

try:
    import flash_attn as _flash_attn
except Exception:
    _flash_attn = None

if TYPE_CHECKING:
    from astrai.inference.cache import KVCache

logger = logging.getLogger(__name__)


_singletons: Dict[type, "AttentionBackend"] = {}


@functools.lru_cache(maxsize=1)
def flash_attn_available() -> bool:
    if not torch.cuda.is_available():
        return False
    fa = _flash_attn
    if fa is None:
        return False

    try:
        major = int(fa.__version__.split(".")[0])
        cc = torch.cuda.get_device_capability()
        cc_num = cc[0] * 10 + cc[1]
    except Exception:
        major, cc_num = 0, 0
    if (major >= 3 and cc_num < 90) or (major < 3 and 0 < cc_num < 70):
        return False

    try:
        if not hasattr(fa, "flash_attn_func"):
            return False
        x = torch.zeros(1, 1, 1, 64, device="cuda", dtype=torch.bfloat16)
        out = fa.flash_attn_func(x, x, x, causal=True)
        return bool(torch.isfinite(out).all().item())
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
    """Available backends in priority order: cuda -> flash -> torch.

    Computed once (machine availability cannot change at runtime) and
    cached forever; the tuple always ends with ``TorchNativeBackend``,
    which is unconditionally available.
    """
    return tuple(
        _instance(cls)
        for cls in (CudaBackend, FlashAttnBackend, TorchNativeBackend)
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
) -> Axes:
    """Snapshot the axes the attention decision table depends on."""
    return tensor_axes(
        q,
        fwd=fwd,
        ndim=q.dim(),
        head_dim=q.size(-1) if q.dim() >= 1 else None,
        has_cache=kv_cache is not None,
        has_mask=attn_mask is not None,
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
) -> Tensor:
    """Functional attention entry point — mirrors ``F.scaled_dot_product_attention``.

    Delegates to the active backend.  ``backend`` (optional) is an explicit
    escape hatch; when omitted the backend is resolved as
    explicit context > ``ASTR_BACKEND`` env > default (cuda > flash > torch).
    Handles KV cache I/O, GQA head expansion, and causal masking so the
    caller only needs to provide projected q/k/v.

    Training calls (``fwd=None``, ``kv_cache=None``) resolve through the
    same capability chain — the CUDA cache kernels cannot run without a
    cache, so they fall back to flash (mask-free/causal calls only) and
    finally to torch SDPA.  An explicitly-selected backend that cannot
    handle the call raises — an implicit one falls back down the priority
    list to the first capable backend.

    Args:
        q: [batch, q_len, n_heads, head_dim] (blhd)
        k: [batch, q_len, n_kv_heads, head_dim] (blhd)
        v: [batch, q_len, n_kv_heads, head_dim] (blhd)
        kv_cache: cache dataclass, or None for training (no cache).
        layer_id: transformer layer index for buffer access.
        attn_mask: pre-built attention mask (SDPA-compatible).
        is_causal: whether to apply causal masking.
        fwd: "prefill" / "decode" for inference, None for training.
        backend: optional explicit backend (name, enum, class, or instance).

    Returns:
        [batch, q_len, n_heads * head_dim]
    """
    explicit = _resolve_backend(backend) if backend is not None else None
    resolution = _dispatch_resolve(
        "attention", q, kv_cache, attn_mask, is_causal, fwd, explicit=explicit
    )
    return resolution.record.obj.forward(
        q, k, v, kv_cache, layer_id, attn_mask, is_causal, fwd
    )


class AttentionBackend(ABC):
    """Abstract base for attention computation strategies.

    Subclasses implement a single ``forward`` and branch on ``fwd``
    ("decode" / "prefill", or None for training) wherever their kernels
    split — the mode taxonomy is the caller's, not the base class's, so
    it lives in the implementations. ``_check_fwd`` is the shared guard
    against unknown mode strings.

    Capability contract — every backend declares:

    * ``available()`` — machine-level: can this backend exist here
      (kernel ``.so`` loaded, flash-attn present, GPU available)?
      Used once to build the default priority list.
    * ``supports_call(q, kv_cache, attn_mask, is_causal, fwd)`` — can this
      backend run this *specific* call (shape/dtype/cache/mask)?  Used by
      ``attention()`` for the per-call fallback.  Resolution logic never
      checks concrete backend types, so adding a backend requires no
      changes outside its own class.

    Three equivalent ways to activate a backend::

        with attn_backend(ATTN_BACKEND.TORCH_NATIVE):  # enum
            ...
        with attn_backend(TorchNativeBackend):          # class
            ...
        with TorchNativeBackend():                       # instance
    """

    def __enter__(self) -> "AttentionBackend":
        self._token = set_override("attention", self)
        return self

    def __exit__(self, *exc) -> None:
        reset_override(self._token)

    @classmethod
    @abstractmethod
    def available(cls) -> bool:
        """Return True if this backend can run on the current machine.

        Checks static availability only (compiled kernels, optional
        packages, GPU presence) — not call-specific constraints.
        """

    @abstractmethod
    def supports_call(
        self,
        q: Tensor,
        kv_cache: Optional["KVCache"],
        attn_mask: Optional[Tensor],
        is_causal: bool,
        fwd: Optional[str],
    ) -> bool:
        """Return True if this backend can run this specific attention call.

        Called on the canonical singleton instance (or a caller-provided
        one); must be side-effect free.
        """
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
        """Run one attention call; ``fwd`` selects the mode.

        Args:
            q: [batch, q_len, n_heads, head_dim]
            k: [batch, q_len, n_kv_heads, head_dim]
            v: [batch, q_len, n_kv_heads, head_dim]
            kv_cache: cache dataclass, or None for training (no cache).
            layer_id: transformer layer index for buffer access.
            attn_mask: pre-built attention mask compatible with SDPA.
            is_causal: whether to apply causal masking.
            fwd: "prefill" / "decode" for inference, None for training.

        Returns:
            [batch, q_len, n_heads * head_dim]
        """

    @staticmethod
    def supports_graph() -> bool:
        """Return True if this backend supports CUDA-graph capture.

        Override in subclasses that can run under ``torch.cuda.graph``.

        Called on the *active* backend instance (or its class) — a cheap
        boolean check with no side-effects.
        """
        return False


class AttentionBackendFactory(BaseFactory[AttentionBackend]):
    """Factory for registered attention backends."""


@AttentionBackendFactory.register(ATTN_BACKEND.TORCH_NATIVE.value)
class TorchNativeBackend(AttentionBackend):
    """Reference backend using torch SDPA with indirect KV cache indexing.

    Writes new K/V into the cache buffers, gathers the full sequence K/V
    via ``req_to_token`` indirect indexing, then calls
    ``F.scaled_dot_product_attention``.

    Packed inference (3-D q) pads the ragged batch to [B, max_q, max_kv]
    and runs a single batched SDPA call with a combined causal+padding
    mask, then unpacks back to the flat layout.

    For training (``kv_cache is None``), skips cache I/O entirely and
    runs SDPA directly on the projected q/k/v.
    """

    @classmethod
    def available(cls) -> bool:
        return True

    def supports_call(
        self,
        q: Tensor,
        kv_cache: Optional["KVCache"],
        attn_mask: Optional[Tensor],
        is_causal: bool,
        fwd: Optional[str],
    ) -> bool:
        return True

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
        self._check_fwd(fwd)
        if q.ndim == 4:
            n_rep = q.size(2) // k.size(2)
            if n_rep > 1:
                k = repeat_kv(k, n_rep)
                v = repeat_kv(v, n_rep)
            return (
                F.scaled_dot_product_attention(
                    q.permute(0, 2, 1, 3),
                    k.permute(0, 2, 1, 3),
                    v.permute(0, 2, 1, 3),
                    attn_mask,
                    is_causal=is_causal,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )

        if kv_cache is None or kv_cache.qo_indptr is None:
            raise ValueError("packed attention requires KV cache metadata")
        kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
        kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v

        # Pad the ragged batch to [B, max_q, max_kv] so one batched SDPA call
        # replaces B per-request calls. The bool mask folds the per-request
        # causal offset (seq_len - q_len) and the kv padding together; padded
        # q rows gather slot/row 0 and are dropped by the [q_valid] unpack,
        # which restores the packed qo_indptr order.
        qo_indptr = kv_cache.qo_indptr
        q_lens = qo_indptr[1:] - qo_indptr[:-1]
        seq_lens = kv_cache.seq_lens
        max_q = int(q_lens.max())
        max_kv = int(seq_lens.max())

        pos = torch.arange(max_kv, device=q.device)
        kv_valid = pos.unsqueeze(0) < seq_lens.unsqueeze(1)
        token_index = kv_cache.req_to_token[kv_cache.req_pool_indices, :max_kv]
        token_index = token_index.masked_fill(~kv_valid, 0)
        k_b = kv_cache.k_buffer[layer_id, token_index]
        v_b = kv_cache.v_buffer[layer_id, token_index]
        n_rep = q.size(1) // k.size(1)
        if n_rep > 1:
            k_b = repeat_kv(k_b, n_rep)
            v_b = repeat_kv(v_b, n_rep)

        q_pos = torch.arange(max_q, device=q.device)
        q_valid = q_pos.unsqueeze(0) < q_lens.unsqueeze(1)
        q_index = (qo_indptr[:-1].unsqueeze(1) + q_pos.unsqueeze(0)).masked_fill(
            ~q_valid, 0
        )
        causal = (
            (seq_lens - q_lens).unsqueeze(1).unsqueeze(2) + q_pos.view(1, max_q, 1)
        ) >= pos.view(1, 1, max_kv)
        mask = (causal & kv_valid.unsqueeze(1)).unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q[q_index].permute(0, 2, 1, 3),
            k_b.permute(0, 2, 1, 3),
            v_b.permute(0, 2, 1, 3),
            attn_mask=mask,
        )
        return out.permute(0, 2, 1, 3)[q_valid]


@AttentionBackendFactory.register(ATTN_BACKEND.CUDA.value)
class CudaBackend(AttentionBackend):
    """CUDA kernel backend with direct KV cache access.

    Decode path: writes K/V to the flat pool, then calls
    ``attn_paged_decode`` with req_to_token + kv_indptr.

    Prefill path: writes K/V to the flat pool, then calls
    ``attn_paged_prefill`` with ragged-batch support via qo_indptr +
    kv_indptr.

    ``kv_cache is None`` (training) raises — the per-call fallback to
    torch SDPA for training / fp32 / unsupported head_dim happens in the
    ``attention()`` entry point.

    Raises ``RuntimeError`` if the required kernel is not available.
    """

    # Head dims supported by the CUDA kernels (single source of truth).
    HEAD_DIMS = (32, 64, 128, 256)

    @classmethod
    def available(cls) -> bool:
        return (
            torch.cuda.is_available()
            and is_available("attn_paged_decode")
            and is_available("attn_paged_prefill")
        )

    def supports_call(
        self,
        q: Tensor,
        kv_cache: Optional["KVCache"],
        attn_mask: Optional[Tensor],
        is_causal: bool,
        fwd: Optional[str],
    ) -> bool:
        # The CUDA kernels are bf16-only, support head_dim in
        # HEAD_DIMS, and need a KV cache (decode/prefill); everything
        # else falls back down the priority list to torch.
        return (
            fwd in ("prefill", "decode")
            and kv_cache is not None
            and q.ndim == 3
            and q.dtype == torch.bfloat16
            and q.size(-1) in self.HEAD_DIMS
            and is_available(f"attn_paged_{fwd}")
        )

    @staticmethod
    def supports_graph() -> bool:
        return True

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
        self._check_fwd(fwd)
        if kv_cache is None:
            raise RuntimeError("CudaBackend does not support training (kv_cache=None)")
        if fwd == "decode":
            return self._decode(q, k, v, kv_cache, layer_id)
        return self._prefill(q, k, v, kv_cache, layer_id, attn_mask, is_causal)

    def _decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: "KVCache",
        layer_id: int,
    ) -> Tensor:
        kv_indptr = kv_cache.kv_indptr

        out = attn_paged_decode(
            q,
            kv_cache.k_buffer[layer_id],
            kv_cache.v_buffer[layer_id],
            kv_cache.req_to_token,
            kv_cache.req_pool_indices,
            kv_indptr,
            new_k=k,
            new_v=v,
            is_causal=True,
            o_part_buf=kv_cache.decode_o_part,
            ml_part_buf=kv_cache.decode_ml_part,
            out_buf=kv_cache.decode_out,
        )
        return out

    def _prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: "KVCache",
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        loc = kv_cache.out_cache_loc
        kv_cache.k_buffer[layer_id, loc] = k
        kv_cache.v_buffer[layer_id, loc] = v

        out = attn_paged_prefill(
            q,
            kv_cache.k_buffer[layer_id],
            kv_cache.v_buffer[layer_id],
            kv_cache.req_to_token,
            kv_cache.req_pool_indices,
            kv_cache.kv_indptr,
            kv_cache.qo_indptr,
            kv_cache.q_tile_to_batch,
            kv_cache.q_tile_to_index,
            attn_mask,
            is_causal=is_causal,
        )
        return out


@AttentionBackendFactory.register(ATTN_BACKEND.FLASH.value)
class FlashAttnBackend(AttentionBackend):
    """FlashAttention backend via the optional ``flash-attn`` package.

    Decode (q_len=1, contiguous cache): writes K/V to the pool, gathers
    flat K/V via the ``req_to_token`` page table, and calls
    ``flash_attn_varlen_func`` over the ragged batch
    (``qo_indptr``/``kv_indptr``).

    Prefill: packed 3-D calls share the ``flash_attn_varlen_func`` path;
    dense 4-D calls go through ``flash_attn_func`` (mask-free only).
    """

    @classmethod
    def available(cls) -> bool:
        return flash_attn_available()

    def supports_call(
        self,
        q: Tensor,
        kv_cache: Optional["KVCache"],
        attn_mask: Optional[Tensor],
        is_causal: bool,
        fwd: Optional[str],
    ) -> bool:
        if not self.available():
            return False
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False
        if fwd is not None:
            return q.ndim == 3 and hasattr(_flash_attn, "flash_attn_varlen_func")
        # Dense (training) path: flash_attn_func cannot apply a custom
        # mask, so only mask-free calls are supported — ``is_causal`` is
        # a flag, not a mask.  Masked training (SFT/DPO/GRPO) must fall
        # back to TorchNativeBackend instead of silently ignoring the mask.
        return attn_mask is None

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
        self._check_fwd(fwd)
        # Decode is always packed; prefill/training split by layout.
        if fwd == "decode" or q.ndim == 3:
            return self._forward_packed(q, k, v, kv_cache, layer_id)
        return self._forward_dense(q, k, v, attn_mask, is_causal)

    def _forward_dense(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        n_rep = q.size(2) // k.size(2)
        if n_rep > 1:
            k = repeat_kv(k, n_rep)
            v = repeat_kv(v, n_rep)

        if attn_mask is not None:
            raise ValueError(
                "FlashAttnBackend cannot handle a custom attention mask; "
                "use a causal mask or select TorchNativeBackend."
            )
        fa = _flash_attn
        if fa is None:
            raise RuntimeError(
                "FlashAttnBackend requires the optional 'flash-attn' package. "
                "Install with `pip install flash-attn`."
            )
        out = fa.flash_attn_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            causal=is_causal,
        )
        return out.contiguous()

    def _forward_packed(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: "KVCache",
        layer_id: int,
    ) -> Tensor:
        fa = _flash_attn
        if fa is None or not hasattr(fa, "flash_attn_varlen_func"):
            raise RuntimeError("packed inference requires flash_attn_varlen_func")
        kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
        kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v
        page_table = kv_cache.req_to_token[
            kv_cache.req_pool_indices, : kv_cache.max_len
        ]
        positions = torch.arange(kv_cache.max_len, device=q.device)
        indices = page_table[positions.unsqueeze(0) < kv_cache.seq_lens.unsqueeze(1)]
        k_flat = kv_cache.k_buffer[layer_id, indices].contiguous()
        v_flat = kv_cache.v_buffer[layer_id, indices].contiguous()
        out = fa.flash_attn_varlen_func(
            q.contiguous(),
            k_flat,
            v_flat,
            kv_cache.qo_indptr,
            kv_cache.kv_indptr,
            int((kv_cache.qo_indptr[1:] - kv_cache.qo_indptr[:-1]).max()),
            int(kv_cache.seq_lens.max()),
            dropout_p=0.0,
            causal=True,
        )
        return out


# Family registration over the generic dispatcher: the "attention" decision
# table.  The Specs mirror each backend's ``supports_call`` exactly (a unit
# test asserts they never drift).  The provider is re-evaluated per
# resolution, so monkeypatching ``flash_attn_available`` (plus clearing
# ``_priority_backends``) is honored, as before.

_CLASS_TO_NAME: Dict[type, str] = {
    CudaBackend: ATTN_BACKEND.CUDA.value,
    FlashAttnBackend: ATTN_BACKEND.FLASH.value,
    TorchNativeBackend: ATTN_BACKEND.TORCH_NATIVE.value,
}

_SPEC_CUDA = (
    axis("fwd").in_("prefill", "decode")
    & axis("has_cache").truthy()
    & axis("ndim").eq(3)
    & axis("dtype").in_(torch.bfloat16)
    & axis("head_dim").in_(*CudaBackend.HEAD_DIMS)
    & Spec.of(
        lambda ax: is_available(f"attn_paged_{ax.get('fwd')}"), "paged kernels loaded"
    )
)

_SPEC_FLASH = (
    axis("dtype").in_(torch.float16, torch.bfloat16)
    & Spec.of(lambda ax: flash_attn_available(), "flash-attn available")
    & (
        (
            axis("fwd").not_none()
            & axis("ndim").eq(3)
            & Spec.of(
                lambda ax: (
                    _flash_attn is not None
                    and hasattr(_flash_attn, "flash_attn_varlen_func")
                ),
                "varlen api present",
            )
        )
        | (axis("fwd").none() & axis("has_mask").falsy())
    )
)


def _attention_records() -> list:
    specs = {
        CudaBackend: _SPEC_CUDA,
        FlashAttnBackend: _SPEC_FLASH,
        TorchNativeBackend: Spec.always(),
    }
    return [
        ImplRecord(
            family="attention",
            name=_CLASS_TO_NAME[type(backend)],
            obj=backend,
            spec=specs[type(backend)],
            priority=position,
        )
        for position, backend in enumerate(_priority_backends())
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
