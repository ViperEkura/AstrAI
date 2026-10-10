"""CPU checks for packed Torch attention mask semantics."""

import pytest
import torch
import torch.nn.functional as F

from astrai.extension.backend.attention.torch import TorchNativeBackend
from astrai.model.kv_cache import PrefillKVCache


def _packed_inputs():
    torch.manual_seed(39)
    q = torch.randn(5, 2, 8)
    k = torch.randn(5, 1, 8)
    v = torch.randn_like(k)
    table = torch.arange(12, dtype=torch.int32).reshape(2, 6)
    cache = PrefillKVCache(
        k_buffer=torch.randn(1, 12, 1, 8),
        v_buffer=torch.randn(1, 12, 1, 8),
        req_to_token=table,
        req_pool_indices=torch.arange(2, dtype=torch.int32),
        seq_lens=torch.tensor([4, 5], dtype=torch.int32),
        max_len=6,
        kv_indptr=torch.tensor([0, 4, 9], dtype=torch.int32),
        out_cache_loc=torch.tensor([2, 3, 8, 9, 10]),
        qo_indptr=torch.tensor([0, 2, 5], dtype=torch.int32),
        q_tile_to_batch=torch.zeros(2, dtype=torch.int32),
        q_tile_to_index=torch.zeros(2, dtype=torch.int32),
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
        tokens = cache.req_to_token[batch, :kv_len].long()
        key = cache.k_buffer[0, tokens].repeat(1, 2, 1)
        value = cache.v_buffer[0, tokens].repeat(1, 2, 1)
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
    out = TorchNativeBackend().forward(
        q, k, v, cache, 0, attn_mask=mask, is_causal=causal, fwd="prefill"
    )
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
    out = TorchNativeBackend().forward(q, k, v, cache, 0, attn_mask=mask, fwd="prefill")
    torch.testing.assert_close(out, _expected(q, cache, mask, False))


def test_oversized_query_mask_rejected_before_cache_write():
    q, k, v, cache = _packed_inputs()
    before_k = cache.k_buffer.clone()
    before_v = cache.v_buffer.clone()
    # Six query rows exceed the five packed query tokens. The caller must
    # provide request-local query rows instead of an ambiguous full mask.
    mask = torch.ones(2, 6, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="request-local"):
        TorchNativeBackend().forward(q, k, v, cache, 0, attn_mask=mask, fwd="prefill")
    torch.testing.assert_close(cache.k_buffer, before_k)
    torch.testing.assert_close(cache.v_buffer, before_v)


def test_decode_mask_query_axis_rejected_before_cache_write():
    q, k, v, cache = _packed_inputs()
    cache.qo_indptr = torch.tensor([0, 1, 2], dtype=torch.int32)
    cache.out_cache_loc = torch.tensor([3, 10])
    before_k = cache.k_buffer.clone()
    before_v = cache.v_buffer.clone()
    # Decode has one query row per request even when the batch has two tokens.
    mask = torch.ones(2, 2, 6, dtype=torch.bool)
    with pytest.raises(ValueError, match="request-local"):
        TorchNativeBackend().forward(
            q[[1, 4]], k[[1, 4]], v[[1, 4]], cache, 0, attn_mask=mask, fwd="decode"
        )
    torch.testing.assert_close(cache.k_buffer, before_k)
    torch.testing.assert_close(cache.v_buffer, before_v)
