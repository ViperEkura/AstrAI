"""Complete-batch parity and ownership gates for the native collector."""

from copy import deepcopy

import pytest
import torch

from astrai.inference.sampling_rng import request_seed, sampling_uniform
from astrai.trainer.backend import ColocatedBackend, ReplicaBackend
from astrai.trainer.rollout import RolloutGenerator, SamplingParams
from tests.support.inference import make_cpu_model, make_cpu_scheduler
from tests.support.models import make_model
from tests.support.tokenizers import FakeTokenizer


def _deterministic_model():
    model = make_cpu_model()

    def forward(ids, *, logits_positions=None, **kwargs):
        ids = ids.reshape(-1)
        if logits_positions is not None:
            ids = ids[logits_positions]
        logits = torch.full((ids.numel(), 256), -10.0, device=ids.device)
        logits.scatter_(1, ((ids + 1) % 256).unsqueeze(1), 10.0)
        return {"logits": logits}

    model.forward = forward
    return model


def _assert_drained(scheduler):
    assert not scheduler.engine_core.pending
    assert (
        not scheduler._planned and not scheduler._pending_order and not scheduler._ready
    )
    assert not scheduler._states and scheduler._kv_manager.request_count == 0
    assert all(not slot["in_use"] for slot in scheduler._executor._result_ring._slots)


@pytest.mark.parametrize("frequency", [0.0, 0.2])
@pytest.mark.parametrize("stop", [[], [8]])
@pytest.mark.parametrize("budget", [None, 2])
def test_fixed_batch_tokens_logprobs_finish_and_pipeline_order(frequency, stop, budget):
    results, pipelines = [], []
    prompts = [[1, 2], [3, 4, 5], [6], [196]]
    for overlap in (False, True):
        tokenizer = FakeTokenizer()
        tokenizer.stop_ids = stop
        scheduler = make_cpu_scheduler(
            _deterministic_model(),
            tokenizer,
            max_batch_size=4,
            enable_overlap=overlap,
            token_budget=budget,
        )
        pipeline = []
        original = scheduler.engine_core.submit

        def observe(plan):
            pipeline.append(len(scheduler.engine_core.pending))
            assert plan.return_logprobs
            return original(plan)

        scheduler.engine_core.submit = observe
        try:
            results.append(
                scheduler.run_batch(
                    prompts,
                    max_tokens=8,
                    temperature=0,
                    frequency_penalty=frequency,
                    return_logprobs=True,
                    return_details=True,
                )
            )
            _assert_drained(scheduler)
            pipelines.append(pipeline)
        finally:
            scheduler.stop()
    assert results[0] == results[1]
    assert all(value == 0 for value in pipelines[0])
    if frequency:
        assert all(value == 0 for value in pipelines[1])
    else:
        assert 1 in pipelines[1] and max(pipelines[1]) == 1


@pytest.mark.parametrize("group", [2, 4, 8])
def test_real_kv_model_grouped_sampling_mask_logprob_parity(group):
    torch.manual_seed(3407)
    original = make_cpu_model()
    tokenizer = FakeTokenizer(with_chat_template=True)
    tokenizer.stop_ids = []
    outputs = []
    for overlap in (False, True):
        model = deepcopy(original).train()
        scheduler = make_cpu_scheduler(
            model,
            tokenizer,
            max_batch_size=group * 2,
            enable_overlap=overlap,
            max_seq_len=64,
        )
        generator = RolloutGenerator(
            ColocatedBackend(scheduler),
            tokenizer,
            SamplingParams(
                group_size=group, max_tokens=6, temperature=1, top_p=1, top_k=0
            ),
        )
        try:
            torch.manual_seed(118)
            outputs.append(generator.generate({"instruction": ["a", "abcd"]}))
            assert model.training
            _assert_drained(scheduler)
        finally:
            scheduler.stop()
    for name in (
        "prompts",
        "prompt_mask",
        "responses",
        "response_mask",
        "logprobs_old",
    ):
        torch.testing.assert_close(
            getattr(outputs[1], name), getattr(outputs[0], name), rtol=0, atol=0
        )
    assert outputs[0].response_texts == outputs[1].response_texts
    assert outputs[0].finish_reasons == outputs[1].finish_reasons
    assert outputs[0].policy_version == outputs[1].policy_version == 0


def test_repeated_batches_and_version_updates_release_all_owners():
    tokenizer = FakeTokenizer()
    tokenizer.stop_ids = []
    scheduler = make_cpu_scheduler(
        _deterministic_model(), tokenizer, max_batch_size=2, enable_overlap=True
    )
    try:
        for version in range(1, 101):
            result = scheduler.run_batch(
                [[1], [10]],
                max_tokens=8,
                temperature=0,
                return_logprobs=True,
                return_details=True,
            )
            assert result[0].token_ids == list(range(2, 10))
            _assert_drained(scheduler)
            assert scheduler.update_weights(version) == version
    finally:
        scheduler.stop()


def test_overlapped_forward_failure_fences_before_release(monkeypatch):
    model = _deterministic_model()
    original = model.forward
    tokenizer = FakeTokenizer()
    tokenizer.stop_ids = []
    scheduler = make_cpu_scheduler(
        model, tokenizer, max_batch_size=2, enable_overlap=True
    )
    calls = []
    forward_count = 0
    synchronize = scheduler._executor.synchronize

    def failing(*args, **kwargs):
        nonlocal forward_count
        forward_count += 1
        if forward_count == 3:
            assert scheduler.engine_core.pending
            raise RuntimeError("injected decode failure")
        return original(*args, **kwargs)

    def fence():
        calls.append("fence")
        return synchronize()

    monkeypatch.setattr(model, "forward", failing)
    monkeypatch.setattr(scheduler._executor, "synchronize", fence)
    try:
        with pytest.raises(RuntimeError, match="injected decode failure"):
            scheduler.run_batch([[1], [10]], max_tokens=8, temperature=0)
        assert calls
        _assert_drained(scheduler)
        assert scheduler.update_weights(1) == 1
    finally:
        scheduler.stop()


def test_replica_backend_passes_overlap_option(monkeypatch):
    captured = {}

    class Scheduler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("astrai.trainer.backend.Scheduler", Scheduler)
    ReplicaBackend(
        make_cpu_model(),
        FakeTokenizer(),
        "cpu",
        enable_overlap=True,
        enable_cuda_graph=False,
    )
    assert captured["enable_overlap"] is True


@pytest.mark.parametrize("overlap", [False, True])
def test_explicit_head_dimension_real_kv_matches_full_forward(overlap):
    torch.manual_seed(731)
    model, _ = make_model(
        "cpu",
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        use_qk_norm=True,
    )
    tokenizer = FakeTokenizer()
    tokenizer.stop_ids = []
    prompts = [[3, 7, 8], [9, 11, 12, 13]]
    expected = []
    with torch.no_grad():
        for prompt in prompts:
            ids = list(prompt)
            for _ in range(4):
                logits = model(torch.tensor([ids]))["logits"][0, -1]
                ids.append(logits.argmax().item())
            expected.append(ids[len(prompt) :])
    scheduler = make_cpu_scheduler(
        model, tokenizer, max_batch_size=2, enable_overlap=overlap
    )
    try:
        actual = scheduler.run_batch(prompts, max_tokens=4, temperature=0)
        assert actual == expected
        _assert_drained(scheduler)
    finally:
        scheduler.stop()


@pytest.mark.parametrize("group", [2, 4, 8])
@pytest.mark.parametrize("budget", [None, 16])
def test_request_seeded_variable_eos_survives_overlap_compaction_and_reordering(
    group, budget, monkeypatch
):
    torch.manual_seed(3407)
    source = make_cpu_model()
    tokenizer = FakeTokenizer(with_chat_template=True)
    tokenizer.stop_ids = list(range(64))
    prompts = ["a", "abcd", "xyz", "longer"]
    results, lengths = [], []

    def forbidden(*args, **kwargs):
        raise AssertionError("seeded requests must not consume shared RNG")

    monkeypatch.setattr(torch, "multinomial", forbidden)
    for overlap, order in (
        (False, [0, 1, 2, 3]),
        (True, [3, 1, 0, 2]),
    ):
        scheduler = make_cpu_scheduler(
            deepcopy(source),
            tokenizer,
            max_batch_size=group * 4,
            enable_overlap=overlap,
            max_seq_len=64,
            token_budget=budget,
        )
        generator = RolloutGenerator(
            ColocatedBackend(scheduler),
            tokenizer,
            SamplingParams(group_size=group, max_tokens=12, seed=118),
        )
        try:
            raw = generator.generate({"instruction": [prompts[i] for i in order]})
            result = {}
            for row, index in enumerate(order):
                result[index] = []
                for response in range(group):
                    mask = raw.response_mask[row, response]
                    n = mask.sum().item()
                    lengths.append(n)
                    result[index].append(
                        (
                            raw.responses[row, response][mask].tolist(),
                            raw.logprobs_old[row, response][mask].tolist(),
                            raw.finish_reasons[row][response],
                        )
                    )
            results.append(result)
            _assert_drained(scheduler)
        finally:
            scheduler.stop()
    assert min(lengths) < max(lengths)
    assert any(item[2] == "stop" for group in results[0].values() for item in group)
    for index in range(4):
        for baseline, candidate in zip(results[0][index], results[1][index]):
            assert baseline[0] == candidate[0]
            assert baseline[2] == candidate[2]
            torch.testing.assert_close(
                torch.tensor(baseline[1]),
                torch.tensor(candidate[1]),
                rtol=1e-5,
                atol=1e-5,
            )


def test_sampling_seed_and_position_are_stable_without_request_identity():
    seed = request_seed(118, 7, [1, 2, 3], 0)
    assert seed != request_seed(118, 7, [1, 2, 3], 1)
    assert seed != request_seed(118, 8, [1, 2, 3], 0)
    draws = [sampling_uniform(seed, position) for position in range(256)]
    assert all(0 < draw < 1 for draw in draws)
    assert len(set(draws)) == 256
    assert draws == [sampling_uniform(seed, position) for position in range(256)]
    assert 0.4 < sum(draws) / len(draws) < 0.6
    with pytest.raises(ValueError, match="seed"):
        sampling_uniform(True, 0)
    with pytest.raises(ValueError, match="position"):
        sampling_uniform(seed, -1)
