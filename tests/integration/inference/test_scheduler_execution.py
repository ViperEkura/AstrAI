"""Tests for scheduler concurrency."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from astrai.extension import CudaBackend, TorchNativeBackend, get_backend
from astrai.inference import Scheduler
from astrai.inference.contracts import SchedulerOutput
from astrai.inference.core.request import Request
from astrai.inference.worker.model_runner import (
    DecodeSteadyState,
    GPUModelRunner,
)
from astrai.inference.worker.pending import (
    PendingExecution,
    ResultRing,
)
from astrai.model.autoregressive_lm import AutoRegressiveLM
from tests.support.models import make_rollout_config
from tests.support.scheduler import (
    _make_contract_scheduler,
    _make_real_scheduler,
    _MultiPatch,
)
from tests.support.tokenizers import FakeTokenizer


def test_step_splits_decode_batch_by_request_backend():
    scheduler = _make_contract_scheduler()
    observed = []

    def submit(requests, return_logprobs=False, *, plan):
        observed.append(
            (type(get_backend()), [request.request_id for request in requests])
        )
        return PendingExecution(
            snapshot=plan,
            sampled_identities=tuple(r.identity for r in requests),
            tokens=torch.tensor([7] * len(requests), dtype=torch.long),
        )

    scheduler._executor.submit_decode = MagicMock(side_effect=submit)

    torch_task = Request("torch", [1], backend=TorchNativeBackend())
    cuda_task = Request("cuda", [1], backend=CudaBackend())
    for request in (torch_task, cuda_task):
        request.input_tokens = 1
        request.output_ids = [1]
        request.num_computed_tokens = 1
        scheduler._metrics.register(request.request_id)

    produced, aborted = scheduler._stepper.step([torch_task, cuda_task])

    assert aborted == []
    assert produced == [torch_task, cuda_task]
    assert torch_task.output_ids == [1, 7]
    assert cuda_task.output_ids == [1, 7]
    assert not scheduler._planned
    assert observed == [
        (TorchNativeBackend, ["torch"]),
        (CudaBackend, ["cuda"]),
    ]


def test_step_batches_ragged_prefill_with_shared_cache_start():
    scheduler = _make_contract_scheduler()
    short = Request("short", [1, 2, 3])
    long = Request("long", [4, 5, 6, 7, 8])

    def prefill(requests, *, plan, **kwargs):
        # Worker packing order deliberately differs from scheduler order.
        packed = sorted(requests, key=lambda r: r.request_id)
        return packed, PendingExecution(
            snapshot=plan,
            sampled_identities=tuple(r.identity for r in packed),
            tokens=torch.tensor([11, 12], dtype=torch.long),
        )

    scheduler._executor.execute_prefill = MagicMock(side_effect=prefill)

    produced, aborted = scheduler._stepper.step([short, long])

    assert aborted == []
    # ``produced`` is now the live set (callers drive it): same members,
    # caller order, not just the requests that sampled this step.
    assert set(produced) == {long, short}
    assert produced == [short, long]
    args, kwargs = scheduler._executor.execute_prefill.call_args
    assert [r.identity for r in args[0]] == [short.identity, long.identity]
    assert kwargs["start_pos"] == 0
    assert kwargs["num_tokens"] == [3, 5]
    assert kwargs["plan"].policy_version == scheduler.policy_version
    assert long.output_ids == [11]
    assert short.output_ids == [12]


def test_execute_prefill_packs_ragged_prompts_and_selects_last_logits():
    executor = object.__new__(GPUModelRunner)
    executor.device = torch.device("cpu")
    executor.kv_manager = MagicMock()
    executor.kv_manager.bind.return_value = MagicMock()
    executor._workspace = MagicMock()
    executor._workspace.max_batch_size = 16  # Add max_batch_size for validation
    all_logits = torch.arange(42, dtype=torch.float32).reshape(6, 7)

    def fake_model(ids, *, position_ids, kv_cache, fwd, logits_positions):
        return {"logits": all_logits[logits_positions]}

    executor.model = MagicMock(side_effect=fake_model)
    task_b = Request("b", [20, 21, 22, 23, 24]).execution("prefill", 1, 4)
    task_a = Request("a", [10, 11, 12]).execution("prefill", 1, 2)
    plan = SchedulerOutput(1, 9, (task_b, task_a))
    executor._submit_sample = MagicMock(
        return_value=PendingExecution(
            snapshot=plan,
            sampled_identities=(task_a.identity, task_b.identity),
            tokens=torch.tensor([101, 102]),
        )
    )

    requests, pending = executor.execute_prefill(
        [task_b, task_a], start_pos=1, plan=plan
    )

    assert requests == [task_a, task_b]
    assert [(r.identity, r.token_id) for r in pending.commit().results] == [
        (task_b.identity, 102),
        (task_a.identity, 101),
    ]
    model_args, model_kwargs = executor.model.call_args
    assert model_args[0].tolist() == [11, 12, 21, 22, 23, 24]
    assert model_kwargs["position_ids"].tolist() == [1, 2, 1, 2, 3, 4]
    assert model_kwargs["logits_positions"].tolist() == [1, 5]
    executor.kv_manager.bind.assert_called_once_with(
        ["a", "b"], executor._workspace, start_pos=1, seq_ends=[3, 5]
    )
    sample_args, sample_kwargs = executor._submit_sample.call_args
    torch.testing.assert_close(sample_args[0], all_logits[[1, 5]])
    assert sample_args[1] == [task_a, task_b]
    assert sample_args[2] is False
    assert sample_kwargs == {"plan": plan}


def test_decode_does_not_reuse_previous_batch_state():
    executor = object.__new__(GPUModelRunner)
    executor.device = torch.device("cpu")
    executor.kv_manager = MagicMock()
    executor.kv_manager.bind_was_steady = True
    executor.kv_manager.bind.return_value = MagicMock()
    executor._graph_supported = False
    executor._graph_ctx = SimpleNamespace(enabled=False)
    executor._pending = None
    executor._result_ring = ResultRing(16, executor.device)

    workspace = MagicMock()
    workspace.max_batch_size = 16
    workspace.position_ids = torch.tensor([2], dtype=torch.long)
    workspace.fill_input_ids.return_value = torch.tensor([7], dtype=torch.long)
    workspace.decode_mask.return_value = torch.ones(1, 1, 9, dtype=torch.bool)
    executor._workspace = workspace
    executor.model = MagicMock(
        return_value={"logits": torch.zeros(1, 1, 10, dtype=torch.float32)}
    )

    old_info = object()
    new_info = SimpleNamespace(has_freq=False)
    executor._decode_cache = DecodeSteadyState(("old",), [2], old_info)
    request = Request("new", list(range(8)), temperature=0)
    request.output_ids = [7]
    request = request.execution("decode", 8, 1)
    plan = SchedulerOutput(1, 2, (request,))
    pending_result = PendingExecution(
        snapshot=plan,
        sampled_identities=(request.identity,),
        tokens=torch.tensor([3], dtype=torch.long),
    )
    executor._submit_sample = MagicMock(return_value=pending_result)

    with patch(
        "astrai.inference.worker.model_runner._build_sampling_batch_info",
        return_value=new_info,
    ):
        assert executor.execute_decode([request], plan=plan) == [3]

    assert workspace.position_ids.tolist() == [8]
    assert executor._decode_cache.task_sig == (request.identity,)
    executor._submit_sample.assert_called_once()
    args, kwargs = executor._submit_sample.call_args
    assert args[1:] == ([request], False)
    assert kwargs["info"] is new_info


def test_decode_fills_input_ids_from_device_on_matching_signature():
    """Steady-state decode copies cached device tokens, skipping the host."""
    executor = object.__new__(GPUModelRunner)
    executor.device = torch.device("cpu")
    executor.kv_manager = MagicMock()
    executor.kv_manager.bind_was_steady = True
    executor.kv_manager.bind.return_value = MagicMock()
    executor._graph_supported = False
    executor._graph_ctx = SimpleNamespace(enabled=False)

    workspace = MagicMock()
    workspace.max_batch_size = 16
    workspace.position_ids = torch.tensor([2], dtype=torch.long)
    workspace.fill_input_ids_from_device.return_value = torch.tensor(
        [9], dtype=torch.long
    )
    executor._workspace = workspace
    executor.model = MagicMock(
        return_value={"logits": torch.zeros(1, 1, 10, dtype=torch.float32)}
    )

    info = SimpleNamespace(has_freq=False)
    tokens = torch.tensor([3], dtype=torch.long)
    request = Request("t1", list(range(8)), temperature=0)
    request.output_ids = [7]
    request = request.execution("decode", 3, 1)
    plan = SchedulerOutput(1, 2, (request,))
    executor._decode_cache = DecodeSteadyState(
        (request.identity,), [2], info, last_tokens=tokens
    )
    executor._pending = None
    executor._result_ring = ResultRing(16, executor.device)
    executor._submit_sample = MagicMock(
        return_value=PendingExecution(
            snapshot=plan, sampled_identities=(request.identity,), tokens=tokens
        )
    )

    with patch(
        "astrai.inference.worker.model_runner._build_sampling_batch_info",
        return_value=info,
    ):
        assert executor.execute_decode([request], plan=plan) == [3]

    workspace.fill_input_ids.assert_not_called()
    workspace.fill_input_ids_from_device.assert_called_once_with(tokens)
    assert workspace.position_ids.tolist() == [3]
    assert executor._decode_cache.task_sig == (request.identity,)
    assert executor._decode_cache.last_tokens is tokens


def test_steady_decode_submit_path_is_sync_free(device):
    """Gate: no hidden device-to-host syncs on the steady decode hot path.

    After the first decode step of an unchanged batch (the steady state the
    serving loop lives in), one ``execute_decode`` call must not resolve any
    device predicate or scalar to host: ``Tensor.item``/``any``/``all``
    probes are patched to raise, and only ``tolist`` stays legal (it is
    the single intentional commit point that yields the sampled tokens).
    """
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30, 40], [5, 6, 7, 8]]
        # First run_batch call warms everything the steady state touches
        # (prefill, first decode, sampling-info build) so the instrumented
        # second batch enters steady state from its first step.
        scheduler.run_batch(prompts, max_tokens=2, temperature=0.7, top_k=10)
        torch.manual_seed(1234)

        banned = (
            torch.Tensor.item,
            torch.Tensor.any,
            torch.Tensor.all,
        )
        calls: list = []

        def _spy(name, orig):
            def _impl(self, *a, **k):
                calls.append(name)
                return orig(self, *a, **k)

            return _impl

        patched = tuple(
            patch.object(torch.Tensor, name, _spy(name, orig))
            for name, orig in zip(("item", "any", "all"), banned)
        )
        with _MultiPatch(patched):
            scheduler.run_batch(prompts, max_tokens=2, temperature=0.7, top_k=10)
        assert not calls, f"steady decode resolved device predicates on host: {calls}"
    finally:
        scheduler.stop()


def test_run_batch_greedy_reproducible_across_calls(device):
    """Greedy decode is bit-identical across invocations of one scheduler.

    Guards the submit/commit split without RNG in play: argmax tokens must
    be stable when the same batch runs twice through the same engine.

    Stochastic reproducibility across run_batch calls is deliberately NOT
    asserted: bf16 GEMM reductions are not bit-deterministic under
    different kernel interleavings, and the async result relay changes the
    host/GPU overlap between calls. Same-seed replay is only guaranteed
    within identical execution conditions.
    """
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30, 40], [7, 8, 9]]
        first = scheduler.run_batch(prompts, max_tokens=6, temperature=0)
        second = scheduler.run_batch(prompts, max_tokens=6, temperature=0)
        assert first == second
        assert all(len(ids) == 6 for ids in first)
    finally:
        scheduler.stop()


def test_paged_pool_selected_via_scheduler_kwargs(device):
    """page_size/kv_tokens reach the BlockPool: paged strategy + prefix cache."""
    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    scheduler = Scheduler(
        model=model,
        tokenizer=FakeTokenizer(),
        max_batch_size=2,
        max_seq_len=64,
        enable_cuda_graph=False,
        page_size=4,
        kv_tokens=128,
    )
    try:
        assert not scheduler._cache.contiguous
        assert scheduler._cache.page_size == 4
        assert scheduler._cache.n_tokens == 128
        # End-to-end: generation works on the paged path.
        out = scheduler.run_batch([[10, 11, 12, 13, 14]], max_tokens=3, temperature=0.0)
        assert len(out[0]) == 3
    finally:
        scheduler.stop()


def test_chunked_prefill_matches_whole_prompt_greedy_tokens(device):
    """Chunked prefill is token-identical to whole-prompt prefill.

    Same model, same prompts, greedy: a small budget forces each prompt
    through multiple continuation chunks (KV-only forwards) before its
    final chunk samples; the sampled sequence must equal the unchunked
    reference exactly.  Also asserts the budget actually split the work
    (chunked forward count > reference forward count).
    """
    torch.manual_seed(0)

    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()

    # Greedy over a random init can argmax the stop id and finish early,
    # which breaks the exact-length gate below on a torch build whose init
    # draw differs.  This test gates prefill-chunking parity, not stopping:
    # the tokenizer stops nothing, so every request runs to max_tokens on
    # every build.
    tokenizer = FakeTokenizer()
    tokenizer.stop_ids = []

    prompts = [[10, 11, 12, 13, 14, 15, 16, 17], [40, 41, 42, 43], [5, 6, 7]]

    ref_sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_seq_len=64,
        enable_cuda_graph=False,
    )
    fwd_calls = []
    orig_prefill = ref_sched._executor.execute_prefill

    def counting_prefill(requests, **kw):
        fwd_calls.append(
            sum(kw.get("num_tokens") or [len(r.prompt_ids) for r in requests])
        )
        return orig_prefill(requests, **kw)

    ref_sched._executor.execute_prefill = counting_prefill
    try:
        reference = ref_sched.run_batch(
            [list(p) for p in prompts], max_tokens=5, temperature=0.0
        )
    finally:
        ref_sched.stop()

    chunked = Scheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_seq_len=64,
        enable_cuda_graph=False,
        token_budget=3,  # every prompt needs >=2 windows
    )
    try:
        # Sanity: budget must actually be in force.
        assert chunked._stepper._token_budget == 3
        chunked_result = chunked.run_batch(
            [list(p) for p in prompts], max_tokens=5, temperature=0.0
        )
    finally:
        chunked.stop()

    assert chunked_result == reference, (chunked_result, reference)
    assert all(len(ids) == 5 for ids in chunked_result)


def test_chunked_prefill_budget_caps_forward_tokens(device):
    """No prefill forward under a budget carries more than budget tokens."""
    torch.manual_seed(0)

    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    # Same deflake as the parity gate above: stopping is orthogonal to the
    # budget being tested, and a random-init argmax onto the stop id would
    # truncate the exact-length assert on some torch builds.
    tokenizer = FakeTokenizer()
    tokenizer.stop_ids = []
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_seq_len=64,
        enable_cuda_graph=False,
        token_budget=4,
    )
    seen = []
    orig = scheduler._executor.execute_prefill

    def spy(requests, **kw):
        seen.append(sum(kw.get("num_tokens") or []))
        return orig(requests, **kw)

    scheduler._executor.execute_prefill = spy
    try:
        out = scheduler.run_batch(
            [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]], max_tokens=2, temperature=0.0
        )
        assert len(out[0]) == 2
    finally:
        scheduler.stop()
    assert seen, "no prefill forwards observed"
    assert max(seen) <= 4, seen
    assert len(seen) >= 3, f"expected multiple chunks, got {seen}"
