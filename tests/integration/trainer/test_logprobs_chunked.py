"""Equivalence tests for the chunked no-grad log-prob path.

:func:`astrai.trainer.strategy.get_logprobs` defers the lm_head projection
and computes per-token log-probs in row chunks whenever it runs under
``torch.no_grad``; with gradients enabled (or for models that cannot skip
their lm_head) it keeps the original full-tensor path.  Both paths must
agree numerically (up to bf16 GEMM tiling noise) for every reduction.
"""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from astrai.parallel.setup import find_free_port
from astrai.trainer.strategy import get_logprobs, rollout_token_logprobs
from astrai.trainer.strategy import ops as strategy_module
from tests.support.models import make_model


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

    monkeypatch.setattr("astrai.trainer.strategy.ops._chunked_token_logprobs", spy)

    with torch.no_grad():
        get_logprobs(model, *batch, "none")
    assert len(calls) == 1

    calls.clear()
    _full_reference(model, batch, "none")
    assert calls == []


def test_rollout_logprobs_score_unpadded_runs(device):
    model, _ = make_model(device)
    torch.manual_seed(5)
    batch, group, prompt_len, response_len = 2, 3, 6, 5
    prompts = torch.randint(3, 200, (batch, prompt_len), device=device)
    prompt_mask = torch.ones(batch, prompt_len, dtype=torch.bool, device=device)
    prompt_mask[0, :2] = False
    prompts[~prompt_mask] = 0
    responses = torch.randint(3, 200, (batch, group, response_len), device=device)
    masks = torch.ones(batch, group, response_len, dtype=torch.bool, device=device)
    masks[0, 1, 3:] = False
    masks[1, 2] = False
    responses[~masks] = 0
    seen = []
    hook = model.register_forward_pre_hook(lambda _, args: seen.append(args[0].numel()))
    with torch.no_grad():
        out = rollout_token_logprobs(model, prompts, prompt_mask, responses, masks)
    hook.remove()
    assert seen == [55]  # five runs packed in rows of 11, not 6 padded rows
    for b in range(batch):
        prompt = prompts[b][prompt_mask[b]]
        for g in range(group):
            response = responses[b, g][masks[b, g]]
            n = response.numel()
            assert torch.all(out["logprobs"][b, g, n:] == 0)
            if n == 0:
                continue
            ids = torch.cat([prompt, response])[None]
            with torch.no_grad():
                logits = model(ids[:, :-1])["logits"].float()
            ref = torch.log_softmax(logits, -1)[0, prompt.numel() - 1 :]
            ref = ref.gather(-1, response[:, None]).squeeze(-1)
            torch.testing.assert_close(
                out["logprobs"][b, g, :n], ref, rtol=1e-5, atol=1e-5
            )


def test_packed_rollout_logprobs_keep_the_grad_chunked_choice(device, monkeypatch):
    model, _ = make_model(device)
    torch.manual_seed(6)
    prompts = torch.randint(3, 200, (1, 4), device=device)
    prompt_mask = torch.ones(1, 4, dtype=torch.bool, device=device)
    responses = torch.randint(3, 200, (1, 2, 5), device=device)
    masks = torch.ones(1, 2, 5, dtype=torch.bool, device=device)
    masks[0, 1, 2:] = False
    responses[~masks] = 0
    calls = []
    original = strategy_module._chunked_token_logprobs_grad

    def spy(hidden_states, weight, targets):
        calls.append(hidden_states.shape[1])
        return original(hidden_states, weight, targets)

    monkeypatch.setattr("astrai.trainer.strategy.ops._chunked_token_logprobs_grad", spy)
    full = rollout_token_logprobs(model, prompts, prompt_mask, responses, masks)
    assert calls == []
    chunked = rollout_token_logprobs(
        model, prompts, prompt_mask, responses, masks, grad_chunked=True
    )
    assert calls == [7]
    torch.testing.assert_close(
        full["logprobs"], chunked["logprobs"], rtol=1e-5, atol=1e-5
    )
    full["logprobs"].sum().backward()


def _ddp_packed_worker(rank, world_size, port):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(0)
        model, _ = make_model("cpu")
        wrapped = DistributedDataParallel(model)
        prompts = torch.randint(3, 200, (1, 4))
        prompt_mask = torch.tensor([[False, False, True, True]])
        prompts[~prompt_mask] = 0
        responses = torch.randint(3, 200, (1, 3, 5))
        masks = torch.ones(1, 3, 5, dtype=torch.bool)
        masks[0, :2, 2:] = False
        responses[~masks] = 0
        ref = rollout_token_logprobs(model, prompts, prompt_mask, responses, masks)
        for grad_chunked in (False, True):
            seen = []
            hook = model.register_forward_pre_hook(
                lambda _, args: seen.append(args[0].numel())
            )
            out = rollout_token_logprobs(
                wrapped, prompts, prompt_mask, responses, masks, grad_chunked
            )
            hook.remove()
            assert seen == [18]  # runs of 4, 4 and 7 in two rows of 9
            torch.testing.assert_close(
                out["logprobs"], ref["logprobs"], rtol=1e-5, atol=1e-5
            )
            model.zero_grad()
            out["logprobs"].sum().backward()
            assert model.lm_head.weight.grad is not None
    finally:
        dist.destroy_process_group()


def test_packed_rollout_logprobs_run_through_ddp():
    mp.spawn(_ddp_packed_worker, args=(2, find_free_port()), nprocs=2)


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
