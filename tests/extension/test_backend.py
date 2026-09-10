"""Backend selection and context-manager switching tests.

These tests do not require CUDA — they only check that the active
backend is correctly set and restored.

Resolution precedence under test: explicit ``attn_backend(...)``
context > ``ASTR_BACKEND`` env override > implicit default.  Training
calls (``fwd=None``, no KV cache) resolve by capability: the CUDA cache
kernels cannot run without a cache, so they fall back to flash (mask-free
calls only) and finally to torch SDPA.
"""

import importlib

import pytest
import torch

from astrai.extension import (
    ATTN_BACKEND,
    AttentionBackend,
    AttentionBackendFactory,
    CudaBackend,
    FlashAttnBackend,
    TorchNativeBackend,
    attention,
    attn_backend,
    get_backend,
)

_attn_module = importlib.import_module("astrai.extension.backend.attention")


def test_default_backend_resolves_to_available():
    """Default backend is the first available in cuda > flash > torch order."""
    backend = get_backend()
    assert isinstance(backend, (CudaBackend, FlashAttnBackend, TorchNativeBackend))


def test_default_backend_is_cached_singleton():
    assert get_backend() is get_backend()


def test_attn_backend_context_with_enum():
    default = get_backend()
    with attn_backend(ATTN_BACKEND.CUDA):
        assert isinstance(get_backend(), CudaBackend)
    assert get_backend() is default


def test_attn_backend_context_with_registered_name():
    default = get_backend()
    with attn_backend("cuda"):
        assert isinstance(get_backend(), CudaBackend)
    assert get_backend() is default


def test_backend_can_read_only_context_selection():
    assert get_backend(use_default=False) is None
    with attn_backend("cuda") as backend:
        assert get_backend(use_default=False) is backend
    assert get_backend(use_default=False) is None


def test_context_beats_environment_backend(monkeypatch):
    """An explicit attn_backend() context wins over ASTR_BACKEND."""
    monkeypatch.setenv("ASTR_BACKEND", "torch_native")
    with attn_backend("cuda"):
        assert isinstance(get_backend(), CudaBackend)
        assert isinstance(get_backend(use_default=False), CudaBackend)


def test_environment_backend_used_without_context(monkeypatch):
    monkeypatch.setenv("ASTR_BACKEND", "torch_native")
    assert isinstance(get_backend(), TorchNativeBackend)
    assert isinstance(get_backend(use_default=False), TorchNativeBackend)


def test_explicit_backend_mismatch_raises(monkeypatch):
    monkeypatch.delenv("ASTR_BACKEND", raising=False)
    q = torch.zeros(1, 2, 4, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="Explicitly-set backend"):
        with attn_backend("cuda"):
            attention(q, q, q)  # cuda + no KV cache -> cannot handle


def test_implicit_backend_falls_back_when_incapable(monkeypatch):
    """An implicit (env) backend that cannot run the call falls back."""
    monkeypatch.setenv("ASTR_BACKEND", "cuda")
    q = torch.zeros(1, 2, 4, 8, dtype=torch.float32)  # fp32: cuda kernels can't
    out = attention(q, q, q, fwd="prefill", is_causal=True)
    assert out.shape == q.shape


def _flash_available(monkeypatch) -> None:
    """Pretend flash-attn is usable and rebuild the priority list."""
    monkeypatch.setattr(_attn_module, "flash_attn_available", lambda: True)
    _attn_module._priority_backends.cache_clear()


def test_training_falls_back_to_flash_before_torch_when_capable(monkeypatch):
    """Training (no cache) prefers flash over torch when flash can run the call."""
    _flash_available(monkeypatch)
    try:
        prio = _attn_module._priority_backends()
        names = [type(b).__name__ for b in prio]
        assert "FlashAttnBackend" in names
        assert names.index("FlashAttnBackend") < names.index("TorchNativeBackend")

        q = torch.zeros(1, 2, 4, 8, dtype=torch.bfloat16)
        # Mask-free training call resolves to flash, not torch.
        resolved = next(b for b in prio if b.supports_call(q, None, None, False, None))
        assert isinstance(resolved, FlashAttnBackend)
    finally:
        _attn_module._priority_backends.cache_clear()


def test_flash_dense_supports_only_mask_free_calls(monkeypatch):
    """FlashAttnBackend cannot apply custom masks in the dense path."""
    _flash_available(monkeypatch)
    flash = _attn_module._instance(_attn_module.FlashAttnBackend)
    q = torch.zeros(1, 2, 4, 8, dtype=torch.bfloat16)
    mask_4d = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

    assert flash.supports_call(q, None, None, False, None) is True
    assert flash.supports_call(q, None, None, True, None) is True
    assert flash.supports_call(q, None, mask_4d, False, None) is False


def test_flash_dense_rejects_custom_mask(monkeypatch):
    """A masked dense call must fail loudly, never silently ignore the mask."""
    _flash_available(monkeypatch)
    flash = _attn_module._instance(_attn_module.FlashAttnBackend)
    q = torch.zeros(1, 2, 4, 8, dtype=torch.bfloat16)
    mask_4d = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="custom attention mask"):
        flash._forward_dense(q, q, q, attn_mask=mask_4d, is_causal=False)


def test_backend_resolution_returns_shared_singletons():
    with attn_backend("cuda") as first:
        pass
    with attn_backend("cuda") as second:
        assert first is second


class _DummyBackend(AttentionBackend):
    """Minimal backend used only to prove capability is polymorphic."""

    @classmethod
    def available(cls) -> bool:
        return True

    def supports_call(self, q, kv_cache, attn_mask, is_causal, fwd) -> bool:
        return True

    def forward(
        self,
        q,
        k,
        v,
        kv_cache=None,
        layer_id=0,
        attn_mask=None,
        is_causal=False,
        fwd=None,
    ):
        self._check_fwd(fwd)
        return q


def test_custom_backend_usable_without_touching_resolution():
    """A third-party backend plugs in via context or explicit param."""
    custom = _DummyBackend()
    q = torch.zeros(1, 2, 4, 8)

    with attn_backend(custom):
        assert get_backend() is custom

    out = attention(q, q, q, backend=custom)
    assert out is q


def test_attention_backend_factory_lists_builtin_backends():
    assert AttentionBackendFactory.list_registered() == [
        "cuda",
        "flash",
        "torch_native",
    ]


def test_attn_backend_rejects_unknown_registered_name():
    with pytest.raises(ValueError, match="Unknown component: 'unknown'"):
        with attn_backend("unknown"):
            pass


def test_attn_backend_context_with_class():
    default = get_backend()
    with attn_backend(CudaBackend):
        assert isinstance(get_backend(), CudaBackend)
    assert get_backend() is default


def test_attn_backend_context_with_instance():
    custom = CudaBackend()
    default = get_backend()
    with attn_backend(custom):
        assert get_backend() is custom
    assert get_backend() is default


def test_cudabackend_is_context_manager():
    default = get_backend()
    with CudaBackend():
        assert isinstance(get_backend(), CudaBackend)
    assert get_backend() is default
