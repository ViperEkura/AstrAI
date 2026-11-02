"""Attention Dispatcher capability and execution tests."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from astrai.extension import attention, op_backend, set_op
from astrai.extension.backend.attention import AttentionDispatcher, cuda, flash
from astrai.extension.backend.attention import torch as native
from astrai.extension.runtime.operation import ExplicitSelectionError


def _spec(q, *, cache=None, mask=None, fwd=None, scale=None):
    return AttentionDispatcher().build_spec(
        q, q, q, cache, 0, mask, False, fwd, scale=scale
    )


def test_cpu_training_uses_torch_function():
    q = torch.randn(1, 2, 4, 8)
    assert (
        AttentionDispatcher().dispatch(q, q, q, None, 0, None, False, None)
        is native.forward
    )


def test_explicit_incapable_choice_is_strict():
    q = torch.zeros(1, 2, 4, 8)
    with op_backend(attention="cuda"):
        with pytest.raises(ExplicitSelectionError):
            attention(q, q, q)


def test_process_choice_is_soft():
    q = torch.zeros(1, 2, 4, 8)
    set_op("attention", "cuda")
    try:
        assert (
            AttentionDispatcher().dispatch(q, q, q, None, 0, None, False, None)
            is native.forward
        )
    finally:
        set_op("attention", None)


def test_unknown_explicit_choice_raises():
    q = torch.zeros(1, 2, 4, 8)
    with pytest.raises(ValueError, match="Unknown attention implementation"):
        attention(q, q, q, implementation="unknown")


def test_flash_capability_checks_device_dtype_and_mask(monkeypatch):
    monkeypatch.setattr(flash, "flash_attn_available", lambda: True)
    q = torch.zeros(1, 2, 4, 8, dtype=torch.bfloat16)
    spec = _spec(q)
    assert not flash.supports(spec)
    cuda_spec = replace(spec, device=torch.device("cuda"))
    assert flash.supports(cuda_spec)
    assert not flash.supports(replace(cuda_spec, mask_kind=str(torch.bool)))


@pytest.mark.parametrize(
    "dtype,supported",
    [
        (torch.bfloat16, True),
        (torch.float16, True),
        (torch.float32, False),
    ],
)
def test_cuda_attention_dtype_capability(monkeypatch, dtype, supported):
    monkeypatch.setattr(cuda, "is_available", lambda name: True)
    q = torch.empty(2, 4, 64, dtype=dtype)
    spec = replace(_spec(q, cache=object(), fwd="decode"), device=torch.device("cuda"))
    assert cuda.supports(spec) is supported


def test_cuda_training_is_never_selected(monkeypatch):
    monkeypatch.setattr(cuda, "is_available", lambda name: True)
    q = torch.empty(2, 4, 64, dtype=torch.bfloat16, requires_grad=True)
    spec = replace(_spec(q, cache=object(), fwd="decode"), device=torch.device("cuda"))
    assert spec.needs_backward
    assert not cuda.supports(spec)
    with pytest.raises(ExplicitSelectionError):
        AttentionDispatcher().select(
            spec, "cuda", "explicit", q, q, q, object(), 0, None, False, "decode"
        )


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
        q, k, v, attn_mask=padding, is_causal=True, implementation="torch", scale=scale
    )
    torch.testing.assert_close(actual, expected)


def test_flash_packed_mask_rejected_before_cache_mutation():
    q = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    mask = torch.ones(1, 4, dtype=torch.bool)
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
    full_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="request-local"):
        cuda.forward(q, kv, kv, cache, 0, full_mask, fwd="prefill")
    assert torch.count_nonzero(cache.k_buffer - 1) == 0
    assert torch.count_nonzero(cache.v_buffer - 1) == 0
