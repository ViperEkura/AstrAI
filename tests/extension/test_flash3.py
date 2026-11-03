"""FA3 interface regressions without requiring a Hopper runner."""

from types import SimpleNamespace

import torch

from astrai.extension.backend.attention import flash3
from astrai.model.masking import PackedDocumentMask


def test_flash3_4d_dense_and_document_packed_forward(monkeypatch):
    calls = []

    def dense(q, k, v, **kwargs):
        calls.append(("dense", q.shape, k.shape, kwargs))
        return q

    def varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        calls.append(
            (
                "varlen",
                q.shape,
                k.shape,
                cu_q.tolist(),
                cu_k.tolist(),
                max_q,
                max_k,
                kwargs,
            )
        )
        return q

    monkeypatch.setattr(
        flash3,
        "_flash3",
        SimpleNamespace(flash_attn_func=dense, flash_attn_varlen_func=varlen),
    )
    q = torch.randn(2, 4, 2, 8)
    k = torch.randn(2, 4, 1, 8)
    v = torch.randn(2, 4, 1, 8)

    out = flash3.forward(q, k, v, None, 0, is_causal=True)
    torch.testing.assert_close(out, q)
    assert calls.pop(0) == (
        "dense",
        q.shape,
        k.shape,
        {"causal": True, "softmax_scale": None},
    )

    packed = PackedDocumentMask(
        doc_ids=torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0]]),
        cu_seqlens=torch.tensor([0, 2, 4, 8], dtype=torch.int32),
        max_seqlen=4,
    )
    out = flash3.forward(q, k, v, None, 0, attn_mask=packed)
    torch.testing.assert_close(out, q)
    assert calls.pop(0) == (
        "varlen",
        torch.Size([8, 2, 8]),
        torch.Size([8, 1, 8]),
        [0, 2, 4, 8],
        [0, 2, 4, 8],
        4,
        4,
        {"causal": True, "softmax_scale": None},
    )


def test_flash3_3d_packed_inference_forward(monkeypatch):
    captured = {}

    def varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        captured.update(
            q=q, k=k, v=v, cu_q=cu_q, cu_k=cu_k, max_q=max_q, max_k=max_k, kwargs=kwargs
        )
        return q

    monkeypatch.setattr(
        flash3, "_flash3", SimpleNamespace(flash_attn_varlen_func=varlen)
    )
    q = torch.randn(3, 2, 8)
    k = torch.randn(3, 1, 8)
    v = torch.randn(3, 1, 8)
    cache = SimpleNamespace(
        k_buffer=torch.zeros(1, 5, 1, 8),
        v_buffer=torch.zeros(1, 5, 1, 8),
        out_cache_loc=torch.tensor([0, 1, 3]),
        req_to_token=torch.tensor([[0, 1, 2], [3, 4, 0]]),
        req_pool_indices=torch.tensor([0, 1]),
        max_len=3,
        seq_lens=torch.tensor([3, 2]),
        qo_indptr=torch.tensor([0, 2, 3], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 3, 5], dtype=torch.int32),
    )
    cache.k_buffer[0, 2] = 2
    cache.k_buffer[0, 4] = 4

    out = flash3.forward(q, k, v, cache, 0, is_causal=True, fwd="prefill")
    torch.testing.assert_close(out, q)
    torch.testing.assert_close(captured["k"][[0, 1, 3]], k)
    assert captured["k"].shape == (5, 1, 8)
    assert captured["cu_q"].tolist() == [0, 2, 3]
    assert captured["cu_k"].tolist() == [0, 3, 5]
    assert (captured["max_q"], captured["max_k"]) == (2, 3)
    assert captured["kwargs"] == {"causal": True, "softmax_scale": None}


def test_flash3_supports_only_its_two_layouts(monkeypatch):
    monkeypatch.setattr(flash3, "available", lambda device: True)
    spec = SimpleNamespace(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        ndim=4,
        fwd=None,
        mask_kind="packed_doc",
        kv_cache_kind="none",
    )
    assert flash3.supports(spec)
    spec.ndim = 3
    spec.fwd = "prefill"
    spec.mask_kind = "none"
    spec.kv_cache_kind = "paged"
    assert flash3.supports(spec)
    spec.mask_kind = "torch.bool"
    assert not flash3.supports(spec)
