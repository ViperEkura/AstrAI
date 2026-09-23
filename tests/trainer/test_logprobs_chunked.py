"""Equivalence tests for the chunked no-grad log-prob path.

:func:`astrai.trainer.strategy.get_logprobs` defers the lm_head projection
and computes per-token log-probs in row chunks whenever it runs under
``torch.no_grad``; with gradients enabled (or for models that cannot skip
their lm_head) it keeps the original full-tensor path.  Both paths must
agree numerically (up to bf16 GEMM tiling noise) for every reduction.
"""

import torch

from astrai.trainer import strategy as strategy_module
from astrai.trainer.strategy import get_logprobs
from tests.helpers import make_model


def _make_batch(device, seq_len=24, vocab_floor=3):
    torch.manual_seed(7)
    input_ids = torch.randint(vocab_floor, 200, (2, seq_len), device=device)
    attn_mask = torch.ones(2, 1, seq_len, seq_len, dtype=torch.bool, device=device)
    attn_mask = attn_mask.tril()
    loss_mask = torch.ones(2, seq_len, dtype=torch.bool, device=device)
    loss_mask[:, : seq_len // 3] = False
    return input_ids, attn_mask, loss_mask


def _full_reference(model, batch, reduction):
    """Force the full-tensor path by enabling grad (values are identical;
    only the autograd graph differs)."""
    input_ids, attn_mask, loss_mask = batch
    with torch.enable_grad():
        return get_logprobs(model, input_ids, attn_mask, loss_mask, reduction)


def test_chunked_matches_full_path_all_reductions(device):
    model, _ = make_model(device)
    batch = _make_batch(device)
    # Force multiple chunks: 2 rows of vocab*2 model-dtype bytes per chunk.
    vocab = model.lm_head.weight.shape[0]
    budget = vocab * model.lm_head.weight.element_size() * 2
    for reduction in ("none", "sum", "mean"):
        with torch.no_grad():
            chunked = strategy_module._CHUNK_LOGIT_BYTES
            strategy_module._CHUNK_LOGIT_BYTES = budget
            try:
                out = get_logprobs(model, *batch, reduction)
            finally:
                strategy_module._CHUNK_LOGIT_BYTES = chunked
        ref = _full_reference(model, batch, reduction)
        torch.testing.assert_close(
            out["logprobs"], ref["logprobs"], rtol=1e-4, atol=1e-4
        )


def test_chunked_path_engages_only_without_grad(device, monkeypatch):
    model, _ = make_model(device)
    batch = _make_batch(device)
    calls = []
    original = strategy_module._chunked_token_logprobs

    def spy(hidden_states, weight, targets):
        calls.append(hidden_states.shape)
        return original(hidden_states, weight, targets)

    monkeypatch.setattr(strategy_module, "_chunked_token_logprobs", spy)

    with torch.no_grad():
        get_logprobs(model, *batch, "none")
    assert len(calls) == 1

    calls.clear()
    _full_reference(model, batch, "none")
    assert calls == []


def test_forward_skip_lm_head_returns_hidden_only(device):
    model, _ = make_model(device)
    input_ids = torch.randint(3, 200, (2, 16), device=device)

    full = model(input_ids)
    skipped = model(input_ids, skip_lm_head=True)

    assert full["logits"] is not None
    assert skipped["logits"] is None
    torch.testing.assert_close(skipped["hidden_states"], full["hidden_states"])


def test_kwarg_rejecting_wrapper_falls_back(device):
    """A model whose forward rejects ``skip_lm_head`` still gets correct
    log-probs via the full-tensor path under no_grad."""

    model, _ = make_model(device)

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.lm_head = inner.lm_head

        def forward(self, input_ids, attn_mask=None):
            return self.inner(input_ids, attn_mask)

    wrapped = _Wrapper(model).to(device)
    batch = _make_batch(device)

    with torch.no_grad():
        out = get_logprobs(wrapped, *batch, "none")
    ref = _full_reference(model, batch, "none")
    torch.testing.assert_close(out["logprobs"], ref["logprobs"], rtol=1e-4, atol=1e-4)
