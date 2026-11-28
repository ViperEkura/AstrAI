"""Attention Dispatcher capability and execution tests."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from astrai.extension import attention, op_backend, set_op
from astrai.extension.backend.attention import AttentionDispatcher, cuda, flash
from astrai.extension.backend.attention import torch as native
from astrai.extension.runtime.operation import ExplicitSelectionError


def _spec(q, *, mask=None, scale=None):
    return AttentionDispatcher().build_spec(q, q, q, mask, False, scale=scale)


def test_explicit_incapable_choice_is_strict():
    q = torch.zeros(1, 2, 4, 8)
    with op_backend(attention="cuda"):
        with pytest.raises(ExplicitSelectionError):
            attention(q, q, q)


def test_process_choice_is_soft():
    q = torch.zeros(1, 2, 4, 8)
    set_op("attention", "cuda")
    try:
        assert AttentionDispatcher().dispatch(q, q, q, None, False) is native.forward
    finally:
        set_op("attention", None)


def test_flash_capability_checks_device_dtype_and_mask(monkeypatch):
    monkeypatch.setattr(flash, "flash_attn_available", lambda: True)
    q = torch.zeros(1, 2, 4, 8, dtype=torch.bfloat16)
    spec = _spec(q)
    assert not flash.supports(spec)
    cuda_spec = replace(spec, device=torch.device("cuda"))
    assert flash.supports(cuda_spec)
    assert not flash.supports(replace(cuda_spec, mask_kind=str(torch.bool)))


def test_cuda_training_is_never_selected(monkeypatch):
    monkeypatch.setattr(cuda, "is_available", lambda name: True)
    q = torch.empty(2, 4, 64, dtype=torch.bfloat16, requires_grad=True)
    spec = replace(_spec(q), device=torch.device("cuda"))
    assert spec.needs_backward
    with pytest.raises(ExplicitSelectionError):
        AttentionDispatcher().select(spec, "cuda", "explicit", q, q, q, None, False)


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
    with op_backend(attention="torch"):
        actual = attention(q, k, v, mask=padding, is_causal=True, scale=scale)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "batch,stride,scale,backward", [(1, 1, None, False), (3, 2, -0.25, True)]
)
def test_torch_single_query_gqa_matches_dense(batch, stride, scale, backward):
    torch.manual_seed(23)
    q, k, v = (
        torch.randn(batch, length, heads * stride, 8)[:, :, ::stride].requires_grad_(
            backward
        )
        for length, heads in ((1, 6), (7, 2), (7, 2))
    )
    with op_backend(attention="torch"), torch.set_grad_enabled(backward):
        actual = attention(q, k, v, is_causal=True, scale=scale)

    # The single bottom-right causal query sees every key.
    group_size = q.size(2) // k.size(2)
    q64 = q.double().transpose(1, 2)
    k64, v64 = (
        x.double().transpose(1, 2).repeat_interleave(group_size, dim=1) for x in (k, v)
    )
    scores = q64 @ k64.transpose(-1, -2)
    factor = q.size(-1) ** -0.5 if scale is None else scale
    expected = ((scores * factor).softmax(dim=-1) @ v64).transpose(1, 2).float()
    sdpa = torch.nn.functional.scaled_dot_product_attention(
        q64.float(), k64.float(), v64.float(), scale=scale
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, sdpa)
    if backward:
        upstream = torch.randn_like(actual)
        torch.testing.assert_close(
            torch.autograd.grad(actual, (q, k, v), upstream),
            torch.autograd.grad(expected, (q, k, v), upstream),
        )


def test_single_token_causal_shard_preserves_sdpa_contract():
    # Ring attention uses the original sequence axis and causal flag to decide
    # which remote ranks are visible, even when each shard has only one token.
    q = torch.zeros(1, 1, 4, 8)
    k = torch.zeros(1, 1, 2, 8)
    with (
        op_backend(attention="torch"),
        patch.object(
            torch.nn.functional,
            "scaled_dot_product_attention",
            wraps=torch.nn.functional.scaled_dot_product_attention,
        ) as ring_entry,
    ):
        attention(q, k, k, is_causal=True)
    ring_entry.assert_called_once()
    assert ring_entry.call_args.args[0].size(-2) == 1
    assert ring_entry.call_args.kwargs["is_causal"]


def test_torch_single_query_gqa_respects_head_mask():
    q = torch.zeros(1, 1, 4, 1)
    k = torch.zeros(1, 4, 2, 1)
    v = torch.arange(8, dtype=q.dtype).view_as(k)
    # Each query head selects its own key, even within a shared KV head.
    mask = torch.eye(4, dtype=torch.bool)[None, :, None, :]
    with op_backend(attention="torch"):
        actual = attention(q, k, v, mask=mask, is_causal=True)
    expected = torch.tensor([0.0, 2.0, 5.0, 7.0]).view_as(q)
    torch.testing.assert_close(actual, expected)


def test_flash_packed_mask_rejected_before_cache_mutation():
    q = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="custom attention mask"):
        flash.forward(q, q, q, attn_mask=mask)


def test_attention_requires_offsets_for_packed_inputs():
    q = torch.zeros(2, 4, 8)
    with pytest.raises(ValueError, match="cu_seqlens_q"):
        attention(q, q, q)
    with pytest.raises(TypeError):
        attention(q, q, q, None)


@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize(
    "missing", ["k_cache", "v_cache", "cache_seqlens", "page_table"]
)
def test_partial_cache_arguments_never_fall_back_to_contiguous(rank, missing):
    q = torch.randn(1, 2, 8) if rank == 3 else torch.randn(1, 1, 2, 8)
    cache = torch.randn(1, 4, 2, 8)
    args = dict(
        k_cache=cache,
        v_cache=cache.clone(),
        cache_seqlens=torch.zeros(1, dtype=torch.int32),
        page_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    before = cache.clone()
    args.pop(missing)
    with pytest.raises(ValueError, match="cache requires"):
        attention(q, q, q, **args)
    torch.testing.assert_close(cache, before)


@pytest.mark.parametrize("argument", ["cu_seqlens_k", "scale", "cu_seqlens_q"])
def test_cache_rejects_incompatible_options_before_writing(argument):
    q = torch.randn(1, 2, 2, 8)
    kc, vc = torch.randn(1, 4, 2, 8), torch.randn(1, 4, 2, 8)
    before = kc.clone(), vc.clone()
    value = 0.5 if argument == "scale" else torch.tensor([0, 2], dtype=torch.int32)
    with pytest.raises(ValueError):
        attention(
            q,
            q,
            q,
            k_cache=kc,
            v_cache=vc,
            cache_seqlens=torch.zeros(1, dtype=torch.int32),
            page_table=torch.zeros(1, 1, dtype=torch.int32),
            **{argument: value},
        )
    torch.testing.assert_close(kc, before[0])
    torch.testing.assert_close(vc, before[1])


@pytest.mark.parametrize("argument", ["cu_seqlens_q", "cu_seqlens_k"])
def test_dense_rejects_offsets(argument):
    q = torch.randn(1, 2, 2, 8)
    with pytest.raises(ValueError, match="dense attention"):
        attention(q, q, q, **{argument: torch.tensor([0, 2], dtype=torch.int32)})


def test_varlen_rejects_mask_and_mixed_layouts():
    q = torch.randn(2, 2, 8)
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    with pytest.raises(ValueError, match="explicit mask"):
        attention(
            q, q, q, cu_seqlens_q=offsets, mask=torch.ones(1, 2, dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="matching"):
        attention(q, q[None], q[None], cu_seqlens_q=offsets)


def test_cache_takes_priority_over_packed_layout():
    q = torch.randn(2, 2, 8)
    k, v = torch.randn_like(q), torch.randn_like(q)
    kc, vc = torch.zeros(2, 4, 2, 8), torch.zeros(2, 4, 2, 8)
    # THD without offsets means one new token per request when a cache is present.
    with op_backend(attention="torch"):
        out = attention(
            q,
            k,
            v,
            k_cache=kc,
            v_cache=vc,
            cache_seqlens=torch.zeros(2, dtype=torch.int32),
            page_table=torch.tensor([[1], [0]], dtype=torch.int32),
            is_causal=True,
        )
    torch.testing.assert_close(out, v)
    torch.testing.assert_close(kc.flip(0)[:, 0], k)
    torch.testing.assert_close(vc.flip(0)[:, 0], v)


@pytest.mark.parametrize("rank", [3, 4])
def test_spec_separates_layout_from_gradient_requirement(rank):
    q = torch.randn(1, 2, 2, 8) if rank == 4 else torch.randn(2, 2, 8)
    k = torch.randn_like(q, requires_grad=True)
    dispatcher = AttentionDispatcher()
    spec = dispatcher.build_spec(q, k, q, None, False)
    assert spec.mode == ("dense" if rank == 4 else "varlen")
    assert spec.needs_backward and spec.phase == "train"
    with torch.no_grad():
        inference = dispatcher.build_spec(q, k, q, None, False)
    assert inference.mode == spec.mode
    assert not inference.needs_backward and inference.phase == "infer"
