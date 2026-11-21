"""Equivalence tests for the chunked no-grad log-prob path.

:func:`astrai.trainer.strategy.get_logprobs` defers the lm_head projection
and computes per-token log-probs in row chunks whenever it runs under
``torch.no_grad``; with gradients enabled (or for models that cannot skip
their lm_head) it keeps the original full-tensor path.  Both paths must
agree numerically (up to bf16 GEMM tiling noise) for every reduction.
"""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from astrai.extension import is_available, op_backend
from astrai.inference.core.engine_core import EngineCore
from astrai.parallel.setup import find_free_port
from astrai.trainer.backend import ColocatedBackend
from astrai.trainer.callbacks.optimization import GradientCheckpointingCallback
from astrai.trainer.context import TrainContext
from astrai.trainer.strategy import get_logprobs, rollout_token_logprobs
from astrai.trainer.strategy import ops as strategy_module
from tests.support.capabilities import requires_nccl, skip_no_cuda
from tests.support.models import make_model, make_seeded_gqa_model
from tests.support.tokenizers import FakeTokenizer


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


def test_chunked_matches_full_path_all_reductions(cpu_device):
    model, _ = make_model(cpu_device)
    batch = _make_batch(cpu_device)
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


def test_chunked_path_engages_only_without_grad(cpu_device, monkeypatch):
    model, _ = make_model(cpu_device)
    batch = _make_batch(cpu_device)
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


def test_rollout_logprobs_score_unpadded_runs(cpu_device):
    model, _ = make_model(cpu_device)
    torch.manual_seed(5)
    batch, group, prompt_len, response_len = 2, 3, 6, 5
    prompts = torch.randint(3, 200, (batch, prompt_len), device=cpu_device)
    prompt_mask = torch.ones(batch, prompt_len, dtype=torch.bool, device=cpu_device)
    prompt_mask[0, :2] = False
    prompts[~prompt_mask] = 0
    responses = torch.randint(3, 200, (batch, group, response_len), device=cpu_device)
    masks = torch.ones(batch, group, response_len, dtype=torch.bool, device=cpu_device)
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


def test_packed_rollout_logprobs_keep_the_grad_chunked_choice(cpu_device, monkeypatch):
    model, _ = make_model(cpu_device)
    torch.manual_seed(6)
    prompts = torch.randint(3, 200, (1, 4), device=cpu_device)
    prompt_mask = torch.ones(1, 4, dtype=torch.bool, device=cpu_device)
    responses = torch.randint(3, 200, (1, 2, 5), device=cpu_device)
    masks = torch.ones(1, 2, 5, dtype=torch.bool, device=cpu_device)
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
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(0)
        model, _ = make_model(device)
        wrapped = DistributedDataParallel(model)
        prompts = torch.randint(3, 200, (1, 4), device=device)
        prompt_mask = torch.tensor([[False, False, True, True]], device=device)
        prompts[~prompt_mask] = 0
        responses = torch.randint(3, 200, (1, 3, 5), device=device)
        masks = torch.ones(1, 3, 5, dtype=torch.bool, device=device)
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


@pytest.mark.cuda
@pytest.mark.distributed
@requires_nccl(2)
def test_packed_rollout_logprobs_run_through_ddp():
    mp.spawn(_ddp_packed_worker, args=(2, find_free_port()), nprocs=2)


def test_forward_skip_lm_head_returns_hidden_only(cpu_device):
    model, _ = make_model(cpu_device)
    input_ids = torch.randint(3, 200, (2, 16), device=cpu_device)

    full = model(input_ids)
    skipped = model(input_ids, skip_lm_head=True)

    assert full["logits"] is not None
    assert skipped["logits"] is None
    torch.testing.assert_close(skipped["hidden_states"], full["hidden_states"])


def test_kwarg_rejecting_wrapper_falls_back(cpu_device):
    """A model whose forward rejects ``skip_lm_head`` still gets correct
    log-probs via the full-tensor path under no_grad."""

    model, _ = make_model(cpu_device)

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.lm_head = inner.lm_head

        def forward(self, input_ids, attn_mask=None):
            return self.inner(input_ids, attn_mask)

    wrapped = _Wrapper(model).to(cpu_device)
    batch = _make_batch(cpu_device)

    with torch.no_grad():
        out = get_logprobs(wrapped, *batch, "none")
    ref = _full_reference(model, batch, "none")
    torch.testing.assert_close(out["logprobs"], ref["logprobs"], rtol=1e-4, atol=1e-4)


@pytest.mark.cuda
@skip_no_cuda
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_fixed_attention_rollout_and_checkpointed_replay(cuda_graph):
    if not (is_available("attention") and torch.cuda.is_bf16_supported()):
        pytest.skip("Native BF16 attention is required")

    model = make_seeded_gqa_model(
        "cuda", dtype=torch.bfloat16, max_position_embeddings=384
    )
    model.eval()
    prompts = [list(range(8, 25)), list(range(26, 315))]
    with op_backend(attention="cuda_consistent"):
        core = EngineCore(
            model,
            FakeTokenizer(),
            max_batch_size=4,
            max_seq_len=384,
            enable_cuda_graph=cuda_graph,
            generation_config={"eos_token_id": []},
        )
    try:
        assert core.cuda_graph_enabled is cuda_graph
        results = ColocatedBackend(core).generate(
            [prompt for prompt in prompts for _ in range(2)],
            max_tokens=17,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            return_details=True,
            return_logprobs=True,
            seeds=[1, 2, 3, 4],
        )
    finally:
        core.stop()

    assert all(
        not result.error_reason and len(result.token_ids) == 17 for result in results
    )
    prompt_ids = torch.tensor(
        [[0] * (len(prompts[1]) - len(prompts[0])) + prompts[0], prompts[1]],
        device="cuda",
    )
    response_ids = torch.tensor(
        [result.token_ids for result in results], device="cuda"
    ).view(2, 2, 17)
    assert all(len(result.logprobs) == 17 for result in results)
    mask = torch.ones_like(response_ids, dtype=torch.bool)
    # Replay also supports a trimmed response from the same rollout.
    mask[0, 1, 7:] = False
    args = (model, prompt_ids, prompt_ids.ne(0), response_ids, mask)

    with op_backend(attention="cuda_consistent"):
        with torch.no_grad():
            expected = rollout_token_logprobs(*args)["logprobs"]
        full = rollout_token_logprobs(*args)["logprobs"]
        checkpointing = GradientCheckpointingCallback([type(model.model.layers[0])])
        context = TrainContext(model=model)
        checkpointing.on_train_begin(context)
        chunked = rollout_token_logprobs(*args, grad_chunked=True)["logprobs"]

    # Compare replay paths separately from generation-side GEMM numerics.
    for replayed in (full, chunked):
        torch.testing.assert_close(replayed, expected, atol=1e-5, rtol=0)
        assert torch.count_nonzero(replayed[~mask]) == 0

    # Backward runs outside the original selection scope.
    full_grads = torch.autograd.grad(full[mask].mean(), tuple(model.parameters()))
    chunked[mask].mean().backward()
    for parameter, reference in zip(model.parameters(), full_grads):
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference, atol=2e-3, rtol=2e-2)
