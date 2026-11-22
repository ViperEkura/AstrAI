"""CPU checks for packed Torch attention mask semantics."""

import pytest
import torch
import torch.nn.functional as F

from astrai.model.components.attention import _model_cache_attention
from astrai.model.kv_cache import DecodeKVCache, PrefillKVCache


def _packed_inputs():
    torch.manual_seed(39)
    q = torch.randn(5, 2, 8)
    k = torch.randn(5, 1, 8)
    v = torch.randn_like(k)
    table = torch.arange(12, dtype=torch.int32).reshape(2, 6)
    cache = PrefillKVCache(
        k_buffer=torch.randn(1, 12, 1, 1, 8),
        v_buffer=torch.randn(1, 12, 1, 1, 8),
        page_table=table,
        cache_seqlens=torch.tensor([2, 2], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
    )
    return q, k, v, cache


def _expected(q, cache, mask, causal):
    if mask.ndim == 2:
        mask = mask[:, None, None, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    result = []
    for batch, (start, stop, kv_len) in enumerate(((0, 2, 4), (2, 5, 5))):
        q_len = stop - start
        tokens = cache.page_table[batch, :kv_len].long()
        key = cache.k_buffer[0, tokens, 0].repeat(1, 2, 1)
        value = cache.v_buffer[0, tokens, 0].repeat(1, 2, 1)
        source = mask[0 if mask.size(0) == 1 else batch]
        missing = False if mask.dtype == torch.bool else float("-inf")
        full = torch.full((source.size(0), q_len, kv_len), missing, dtype=mask.dtype)
        for row in range(q_len):
            source_row = 0 if source.size(-2) == 1 else row
            if source_row < source.size(-2):
                n_keys = min(kv_len, source.size(-1))
                full[:, row, :n_keys] = source[:, source_row, :n_keys]
        if causal:
            visible = torch.arange(kv_len)[None, :] <= (
                kv_len - q_len + torch.arange(q_len)[:, None]
            )
            if mask.dtype == torch.bool:
                full = full & visible
            else:
                full = full.masked_fill(~visible, float("-inf"))
        out = F.scaled_dot_product_attention(
            q[start:stop].transpose(0, 1)[None],
            key.transpose(0, 1)[None],
            value.transpose(0, 1)[None],
            attn_mask=full[None],
        )
        result.append(out[0].transpose(0, 1))
    return torch.cat(result)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mask_kind", ["bool_short", "additive_short", "key_only"])
def test_packed_short_mask_matches_explicit_extent(causal, mask_kind):
    q, k, v, cache = _packed_inputs()
    if mask_kind == "bool_short":
        mask = torch.tensor([[[[True, False, True], [False, True, True]]]])
    elif mask_kind == "additive_short":
        mask = torch.tensor([[[[0.0, -1.0, float("-inf")], [0.0, -0.3, 0.0]]]])
    else:
        mask = torch.tensor([[True, False, True]])
    out = _model_cache_attention(q, k, v, cache, 0, mask, causal)
    torch.testing.assert_close(out, _expected(q, cache, mask, causal))
    # The third query row is absent from the short non-broadcast mask.
    if mask_kind != "key_only":
        assert torch.count_nonzero(out[-1]) == 0


def test_packed_mask_query_axis_is_request_local():
    q, k, v, cache = _packed_inputs()
    mask = torch.tensor(
        [
            [
                [True, False, True, False, True, True],
                [False, True, False, True, True, True],
                [True, True, True, False, False, True],
            ]
        ]
    )
    out = _model_cache_attention(q, k, v, cache, 0, mask, False)
    torch.testing.assert_close(out, _expected(q, cache, mask, False))


def test_oversized_query_mask_rejected_before_cache_write():
    q, k, v, cache = _packed_inputs()
    before_k = cache.k_buffer.clone()
    before_v = cache.v_buffer.clone()
    # Six query rows exceed the five packed query tokens. The caller must
    # provide request-local query rows instead of an ambiguous full mask.
    mask = torch.ones(2, 6, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="request-local"):
        _model_cache_attention(q, k, v, cache, 0, mask, False)
    torch.testing.assert_close(cache.k_buffer, before_k)
    torch.testing.assert_close(cache.v_buffer, before_v)


def test_decode_mask_query_axis_rejected_before_cache_write():
    q, k, v, cache = _packed_inputs()
    cache = DecodeKVCache(
        cache.k_buffer,
        cache.v_buffer,
        cache.page_table,
        torch.tensor([3, 4], dtype=torch.int32),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
    )
    before_k = cache.k_buffer.clone()
    before_v = cache.v_buffer.clone()
    # Decode has one query row per request even when the batch has two tokens.
    mask = torch.ones(2, 2, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="request-local"):
        _model_cache_attention(q[[1, 4]], k[[1, 4]], v[[1, 4]], cache, 0, mask, False)
    torch.testing.assert_close(cache.k_buffer, before_k)
    torch.testing.assert_close(cache.v_buffer, before_v)
