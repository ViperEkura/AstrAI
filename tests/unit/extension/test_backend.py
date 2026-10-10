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
from types import SimpleNamespace

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
    dispatch,
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


def test_context_beats_set_op_backend():
    """An explicit attn_backend() context wins over set_op."""
    dispatch.set_op("attention", "torch_native")
    with attn_backend("cuda"):
        assert isinstance(get_backend(), CudaBackend)
        assert isinstance(get_backend(use_default=False), CudaBackend)


def test_set_op_backend_used_without_context():
    dispatch.set_op("attention", "torch_native")
    assert isinstance(get_backend(), TorchNativeBackend)
    assert isinstance(get_backend(use_default=False), TorchNativeBackend)


def test_explicit_backend_mismatch_raises():
    dispatch.set_op("attention", None)
    q = torch.zeros(1, 2, 4, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="Explicitly-set backend"):
        with attn_backend("cuda"):
            attention(q, q, q)  # cuda + no KV cache -> cannot handle


def test_implicit_backend_falls_back_when_incapable():
    """An implicit (set_op) backend that cannot run the call falls back."""
    dispatch.set_op("attention", "cuda")
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


def test_registered_third_party_backend_probe_receives_explicit_scale():
    class ScaleAwareBackend(_DummyBackend):
        supports_scale = True
        last_scale = None

        def supports_call(
            self, q, kv_cache, attn_mask, is_causal, fwd, *, scale=None
        ) -> bool:
            self.last_scale = scale
            return scale is None or scale > 0

    AttentionBackendFactory.register("scale_probe")(ScaleAwareBackend)
    dispatch.invalidate("attention")
    q = torch.zeros(1, 2, 4, 8)
    try:
        backend = _attn_module._instance(ScaleAwareBackend)
        resolution = dispatch.resolve(
            "attention", q, None, None, False, None, explicit="scale_probe", scale=0.25
        )
        assert resolution.record.obj is backend
        assert backend.last_scale == 0.25
        with pytest.raises(RuntimeError, match="Explicitly-set backend"):
            dispatch.resolve(
                "attention",
                q,
                None,
                None,
                False,
                None,
                explicit="scale_probe",
                scale=-0.25,
            )
        assert backend.last_scale == -0.25
    finally:
        AttentionBackendFactory._entries.pop("scale_probe")
        _attn_module._singletons.pop(ScaleAwareBackend, None)
        _attn_module._priority_backends.cache_clear()
        dispatch.invalidate("attention")


def test_default_backend_probe_preserves_legacy_signature_with_scale():
    class LegacyScaleBackend(_DummyBackend):
        supports_scale = True

    q = torch.zeros(1, 2, 4, 8)
    axes = _attn_module._axes(q, None, None, False, None, scale=0.25)
    assert LegacyScaleBackend.supports_axes(axes)


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


@pytest.mark.parametrize("q_len,kv_len", [(7, 3), (3, 7)])
@pytest.mark.parametrize("scale", [None, 0.0, -0.3, 0.2])
def test_attention_causal_scale_and_mask_share_one_contract(q_len, kv_len, scale):
    torch.manual_seed(22)
    q = torch.randn(2, q_len, 4, 8)
    k = torch.randn(2, kv_len, 2, 8)
    v = torch.randn_like(k)
    padding = torch.ones(1, kv_len, dtype=torch.bool)
    padding[:, -1] = False
    causal = torch.arange(q_len)[:, None] + kv_len - q_len >= torch.arange(kv_len)[None]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.repeat_interleave(2, dim=2).transpose(1, 2),
        v.repeat_interleave(2, dim=2).transpose(1, 2),
        attn_mask=causal & padding[:, None, None, :],
        scale=scale,
    ).transpose(1, 2)
    actual = attention(
        q, k, v, attn_mask=padding, is_causal=True, backend="torch_native", scale=scale
    )
    torch.testing.assert_close(actual, expected)


def test_flash_packed_mask_is_rejected_before_cache_mutation(monkeypatch):
    _flash_available(monkeypatch)
    flash = _attn_module._instance(_attn_module.FlashAttnBackend)
    q = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    mask = torch.ones(1, 4, dtype=torch.bool)
    assert not flash.supports_call(q, object(), mask, True, "prefill")
    with pytest.raises(ValueError, match="custom attention mask"):
        flash.forward(q, q, q, object(), 0, attn_mask=mask, fwd="prefill")


def test_cuda_paged_mask_shape_rejected_before_cache_write():
    q = torch.zeros(2, 2, 64)
    kv = torch.zeros(2, 1, 64)
    cache = SimpleNamespace(
        req_pool_indices=torch.zeros(1, dtype=torch.int32),
        req_to_token=torch.zeros(1, 4, dtype=torch.int32),
        k_buffer=torch.ones(1, 4, 1, 64),
        v_buffer=torch.ones(1, 4, 1, 64),
    )
    # A complete sequence mask has four query rows, but this packed call has
    # only two request-local query rows.
    full_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="request-local"):
        CudaBackend().forward(q, kv, kv, cache, 0, full_mask, fwd="prefill")
    assert torch.count_nonzero(cache.k_buffer - 1) == 0
    assert torch.count_nonzero(cache.v_buffer - 1) == 0
