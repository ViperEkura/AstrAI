"""Unit tests for the online rollout module."""

import threading

import pytest
import torch

from astrai.inference.scheduler import InferenceScheduler
from astrai.inference.task import GenerationResult
from astrai.trainer.backend import ColocatedBackend, P2PCopyPublisher, ReplicaBackend
from astrai.trainer.rollout import (
    BaseRewardModel,
    RawRollout,
    RolloutEvaluator,
    RolloutGenerator,
    RolloutResult,
    RolloutRunner,
    RolloutVersionError,
    SamplingParams,
)
from tests.conftest import skip_lt2_cuda
from tests.helpers import FakeExecutor, FakeTokenizer, make_model


class ConstantRewardModel(BaseRewardModel):
    """Returns a constant reward for every response."""

    def __init__(self, value: float = 1.0):
        self.value = value

    def score(self, prompts, responses):
        B = len(prompts)
        G = len(responses[0]) if B else 0
        return torch.full((B, G), float(self.value))


class BadShapeRewardModel(BaseRewardModel):
    def score(self, prompts, responses):
        return torch.zeros(len(prompts))


class NonFiniteRewardModel(BaseRewardModel):
    def score(self, prompts, responses):
        B = len(prompts)
        G = len(responses[0]) if B else 0
        return torch.full((B, G), float("nan"))


def _make_scheduler(model, tokenizer, max_batch_size=8, max_len=128):
    return InferenceScheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_seq_len=max_len,
    )


def _make_instruction_batch(n=2):
    """Build a batch of instruction+input prompts as lists of strings."""
    instructions = [f"Tell me about topic {i}" for i in range(n)]
    inputs = [f"context {i}" for i in range(n)]
    return {"instruction": instructions, "input": inputs}


def _blocking_hook(original, started, release):
    """Wrap a hook so it signals ``started`` then blocks until ``release``."""

    def hook(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    return hook


def _assert_interleaved(first, second, *, started, release, finished):
    """Run ``first`` until it blocks, then assert ``second`` cannot finish
    while ``first`` holds the lock; release, join both, and surface errors."""
    errors = []

    def run_safely(fn):
        def run():
            try:
                fn()
            except BaseException as exc:
                errors.append(exc)

        return run

    first_thread = threading.Thread(target=run_safely(first))
    second_thread = threading.Thread(target=run_safely(second))
    first_thread.start()
    assert started.wait(timeout=5)
    second_thread.start()
    assert not finished.wait(timeout=0.1)

    release.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []


def test_raw_rollout_fields():
    r = RawRollout(
        prompts=torch.zeros(2, 4, dtype=torch.long),
        prompt_mask=torch.ones(2, 4, dtype=torch.bool),
        responses=torch.zeros(2, 3, 5, dtype=torch.long),
        response_mask=torch.ones(2, 3, 5, dtype=torch.bool),
        logprobs_old=torch.zeros(2, 3, 5),
    )
    assert r.prompts.shape == (2, 4)
    assert r.responses.shape == (2, 3, 5)
    assert r.policy_version == 0
    assert r.prompt_texts == []
    assert r.response_texts == []


def test_rollout_result_inherits_raw_rollout_fields():
    r = RolloutResult(
        prompts=torch.zeros(2, 4, dtype=torch.long),
        prompt_mask=torch.ones(2, 4, dtype=torch.bool),
        responses=torch.zeros(2, 3, 5, dtype=torch.long),
        response_mask=torch.ones(2, 3, 5, dtype=torch.bool),
        logprobs_old=torch.zeros(2, 3, 5),
        rewards=torch.zeros(2, 3),
    )
    assert r.rewards.shape == (2, 3)
    assert r.prompts.shape == (2, 4)
    assert r.responses.shape == (2, 3, 5)
    assert r.prompt_mask.shape == (2, 4)
    # RolloutResult must carry every RawRollout field.
    raw_fields = {f for f in RawRollout.__dataclass_fields__}
    assert raw_fields.issubset(set(RolloutResult.__dataclass_fields__))


def test_base_reward_model_is_abstract():
    with pytest.raises(TypeError):
        BaseRewardModel()


def _make_generator(device, **kw):
    model, _ = make_model(device, max_position_embeddings=128)
    tokenizer = FakeTokenizer(with_chat_template=True)
    scheduler = _make_scheduler(
        model,
        tokenizer,
        max_batch_size=kw.get("max_batch_size", 8),
        max_len=kw.get("max_position_embeddings", 128),
    )
    generator = RolloutGenerator(
        backend=ColocatedBackend(scheduler),
        tokenizer=tokenizer,
        params=SamplingParams(
            max_tokens=kw.get("max_tokens", 8),
            group_size=kw.get("group_size", 2),
            temperature=kw.get("temperature", 1.0),
            top_k=kw.get("top_k", 0),
            top_p=kw.get("top_p", 1.0),
        ),
    )
    return generator, model


def test_rollout_generator_shapes(device):
    gen, _ = _make_generator(device, group_size=3, max_tokens=5)
    batch = _make_instruction_batch(n=2)
    r = gen.generate(batch)
    assert r.responses.shape == (2, 3, 5)
    assert r.response_mask.shape == (2, 3, 5)
    assert r.logprobs_old.shape == (2, 3, 5)
    assert r.prompt_mask.shape == r.prompts.shape
    assert len(r.prompt_texts) == 2
    assert len(r.response_texts) == 2
    assert len(r.response_texts[0]) == 3
    assert r.policy_version == 0


def test_rollout_generator_uses_eval_and_restores_mode(device):
    gen, model = _make_generator(device, group_size=1, max_tokens=2)
    model.train()
    seen_training = []
    original = gen.backend.scheduler.run_batch

    def recording_run_batch(*args, **kwargs):
        seen_training.append(model.training)
        return original(*args, **kwargs)

    gen.backend.scheduler.run_batch = recording_run_batch
    gen.generate(_make_instruction_batch(n=1))
    assert seen_training == [False]
    assert model.training is True


def test_generate_params_override_training_defaults(device):
    """A per-call SamplingParams overrides group size and token budget."""
    gen, _ = _make_generator(device, group_size=2, max_tokens=8)
    batch = _make_instruction_batch(n=2)

    default = gen.generate(batch)
    assert default.responses.shape[1] == 2

    override = gen.generate(
        batch, SamplingParams(group_size=1, max_tokens=1, temperature=1.0)
    )
    assert override.responses.shape[:2] == (2, 1)
    assert override.responses.shape[2] <= 1
    # The generator's training defaults are untouched by the override.
    assert gen.params.group_size == 2
    assert gen.generate(batch).responses.shape[1] == 2


def test_rollout_evaluator_reports_reward_metrics_and_leaves_cache(device):
    """The val evaluator scores under its own params; the replay cache,
    its cadence counter, and the cache key stay exactly as they were."""
    runner, _ = _make_runner(device, group_size=2, max_tokens=4, rollout_interval=10)
    batch = _make_instruction_batch(n=2)

    result, is_fresh = runner(batch)
    assert is_fresh
    cache_before = runner._cache
    steps_before = runner._steps_since_rollout

    evaluator = RolloutEvaluator(
        generator=runner.generator,
        reward_model=ConstantRewardModel(2.0),
        params=SamplingParams(group_size=1, max_tokens=2, temperature=0.0),
    )
    metrics = evaluator.evaluate(batch)

    assert metrics["reward_mean"] == pytest.approx(2.0)
    assert metrics["reward_std"] == pytest.approx(0.0)
    assert metrics["num_responses"] == 2.0  # B=2 prompts x G=1 override
    assert metrics["response_len_mean"] <= 2.0
    assert runner._cache is cache_before
    assert runner._cache_key == runner._batch_key(batch)
    assert runner._steps_since_rollout == steps_before


def _make_replica_pair():
    """A training model on cuda:0 and a diverged replica on cuda:1."""
    train_model, _ = make_model("cuda:0", max_position_embeddings=128)
    replica_model, _ = make_model("cuda:1", max_position_embeddings=128)
    with torch.no_grad():
        for p in replica_model.parameters():
            p.add_(1.0)
    return train_model, replica_model


@skip_lt2_cuda
def test_replica_backend_generation_and_output_device():
    """A replica on cuda:1 generates under its own scheduler and the
    generator lands rollout tensors on the training device."""
    train_model, replica_model = _make_replica_pair()
    replica_model.load_state_dict(train_model.state_dict())
    tokenizer = FakeTokenizer(with_chat_template=True)
    backend = ReplicaBackend(
        model=replica_model,
        tokenizer=tokenizer,
        device="cuda:1",
        max_batch_size=8,
        max_seq_len=128,
    )
    generator = RolloutGenerator(
        backend=backend,
        tokenizer=tokenizer,
        params=SamplingParams(group_size=2, max_tokens=4),
        output_device=torch.device("cuda", 0),
    )

    rollout = generator.generate(_make_instruction_batch(n=2))

    assert rollout.responses.shape[:2] == (2, 2)
    assert rollout.responses.device == torch.device("cuda", 0)
    assert rollout.logprobs_old.device == torch.device("cuda", 0)
    # The replica never toggles the shared training model: it stays eval.
    assert replica_model.training is False


@skip_lt2_cuda
def test_p2p_publisher_copies_weights_and_advances_version():
    train_model, replica_model = _make_replica_pair()
    tokenizer = FakeTokenizer(with_chat_template=True)
    backend = ReplicaBackend(
        model=replica_model,
        tokenizer=tokenizer,
        device="cuda:1",
        max_batch_size=8,
        max_seq_len=128,
    )
    publisher = P2PCopyPublisher(backend)

    with torch.no_grad():
        for p in train_model.parameters():
            p.mul_(2.0).add_(1.0)
    publisher.publish(3, train_model)

    assert backend.policy_version == 3
    for (name, src), (name2, dst) in zip(
        train_model.state_dict().items(), backend.model.state_dict().items()
    ):
        assert name == name2
        assert torch.equal(dst.to(src.device), src)
    # A second publish reuses the cached pairs and keeps versions monotone.
    publisher.publish(4, train_model)
    assert backend.policy_version == 4


@skip_lt2_cuda
def test_online_optimizer_step_syncs_replica_atomically():
    """strategy.optimizer_step advances the replica's weights and version
    inside one commit — the cross-GPU rollout contract."""
    from astrai.trainer.strategy import GRPOStrategy
    from tests.helpers import make_frozen

    train_model, replica_model = _make_replica_pair()
    tokenizer = FakeTokenizer(with_chat_template=True)
    backend = ReplicaBackend(
        model=replica_model,
        tokenizer=tokenizer,
        device="cuda:1",
        max_batch_size=8,
        max_seq_len=128,
    )
    runner = RolloutRunner(
        generator=RolloutGenerator(
            backend=backend,
            tokenizer=tokenizer,
            params=SamplingParams(group_size=2, max_tokens=4),
            output_device=torch.device("cuda", 0),
        ),
        reward_model=ConstantRewardModel(),
        rollout_interval=2,
    )
    strategy = GRPOStrategy(
        model=train_model,
        device="cuda:0",
        old_model=None,
        ref_model=make_frozen(train_model, "cuda:0"),
        clip_eps=0.2,
        kl_coef=0.01,
        group_size=2,
        model_fn=None,
        executor=FakeExecutor(),
    )
    strategy.set_rollout_runner(runner)
    strategy.set_weight_publishers([P2PCopyPublisher(backend)])

    for p in train_model.parameters():
        p.grad = torch.ones_like(p)
    optimizer = torch.optim.SGD(train_model.parameters(), lr=0.1)
    before = next(train_model.parameters()).detach().clone()
    strategy.optimizer_step(optimizer)

    assert not torch.equal(next(train_model.parameters()), before)
    assert backend.policy_version == 1
    assert runner.policy_version == 1
    for (name, src), (name2, dst) in zip(
        train_model.state_dict().items(), backend.model.state_dict().items()
    ):
        assert name == name2
        assert torch.equal(dst.to(src.device), src)


def test_rollout_generator_serializes_generation_and_policy_update(device):
    gen, _ = _make_generator(device, group_size=1, max_tokens=2)
    generation_started = threading.Event()
    allow_generation_to_finish = threading.Event()
    update_finished = threading.Event()
    gen._generate_eval = _blocking_hook(
        gen._generate_eval, generation_started, allow_generation_to_finish
    )

    _assert_interleaved(
        lambda: gen.generate(_make_instruction_batch(n=1)),
        lambda: gen.apply_weight_update(1, lambda _version: update_finished.set()),
        started=generation_started,
        release=allow_generation_to_finish,
        finished=update_finished,
    )

    assert update_finished.is_set()
    assert gen.policy_version == 1


def test_apply_weight_update_hands_derived_version_inside_lock(device):
    gen, _ = _make_generator(device, group_size=1, max_tokens=2)
    seen = {}

    def record(policy_version):
        # The target version is derived under the lock and passed in; the
        # live version has not moved yet because the commit follows the
        # update — weight publishers rely on exactly this ordering.
        seen["arg"] = policy_version
        seen["live_during"] = gen.backend.scheduler.policy_version

    gen.apply_weight_update(None, record)

    assert seen == {"arg": 1, "live_during": 0}
    assert gen.policy_version == 1


def test_rollout_generator_serializes_direct_scheduler_update(device):
    gen, _ = _make_generator(device, group_size=1, max_tokens=2)
    generation_started = threading.Event()
    allow_generation_to_finish = threading.Event()
    update_finished = threading.Event()
    gen._generate_eval = _blocking_hook(
        gen._generate_eval, generation_started, allow_generation_to_finish
    )
    rollout = []

    def update_scheduler_directly():
        gen.backend.scheduler.update_weights(1)
        update_finished.set()

    _assert_interleaved(
        lambda: rollout.append(gen.generate(_make_instruction_batch(n=1))),
        update_scheduler_directly,
        started=generation_started,
        release=allow_generation_to_finish,
        finished=update_finished,
    )

    assert rollout[0].policy_version == 0
    assert gen.policy_version == 1


def test_rollout_generator_keeps_generation_start_version(device):
    gen, _ = _make_generator(device, group_size=1, max_tokens=2)
    original_run_batch = gen.backend.scheduler.run_batch

    def update_after_generation(*args, **kwargs):
        result = original_run_batch(*args, **kwargs)
        gen.backend.scheduler.update_weights(1)
        return result

    gen.backend.scheduler.run_batch = update_after_generation

    rollout = gen.generate(_make_instruction_batch(n=1))

    assert rollout.policy_version == 0
    assert gen.policy_version == 1


def test_rollout_generator_mask_matches_responses(device):
    """Positions beyond a response's length are pad (mask False)."""
    gen, _ = _make_generator(device, group_size=2, max_tokens=6)
    batch = _make_instruction_batch(n=2)
    r = gen.generate(batch)
    for i in range(2):
        for g in range(2):
            real = r.response_mask[i, g].sum().item()
            assert r.responses[i, g, real:].sum() == 0
            if real < r.logprobs_old.size(-1):
                assert torch.all(r.logprobs_old[i, g, real:] == 0)


def test_rollout_generator_logprobs_are_nonpositive(device):
    """Behaviour-policy logprobs of sampled tokens should be <= 0."""
    gen, _ = _make_generator(device, group_size=2, max_tokens=4)
    batch = _make_instruction_batch(n=1)
    r = gen.generate(batch)
    for i in range(1):
        for g in range(2):
            mask = r.response_mask[i, g]
            lp = r.logprobs_old[i, g][mask]
            assert torch.all(lp <= 1e-5)


def test_rollout_generator_rejects_failed_requests(device):
    gen, _ = _make_generator(device, group_size=2, max_tokens=4)

    def failed_run_batch(*_args, **kwargs):
        assert kwargs["return_details"] is True
        return [
            GenerationResult([1], [-0.1], "length"),
            GenerationResult([], [], "rejected", "kv_cache_allocation_failed"),
        ]

    gen.backend.scheduler.run_batch = failed_run_batch

    with pytest.raises(
        RuntimeError,
        match="Rollout generation failed: request 1: kv_cache_allocation_failed",
    ):
        gen.generate(_make_instruction_batch(n=1))


def test_rollout_generator_instruction_role_mapping(device):
    """instruction -> system, input -> user, output -> assistant."""
    gen, _ = _make_generator(device, group_size=1, max_tokens=4)
    batch = {
        "instruction": ["Be helpful"],
        "input": ["What is 2+2?"],
        "output": ["Four"],
    }
    r = gen.generate(batch)
    text = r.prompt_texts[0]
    assert "SYSTEM: Be helpful" in text
    assert "USER: What is 2+2?" in text
    assert "ASSISTANT: Four" in text


def test_rollout_generator_messages_format(device):
    """Rollout also accepts pre-built messages."""
    gen, _ = _make_generator(device, group_size=2, max_tokens=4)
    batch = {
        "messages": [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Goodbye"}],
        ]
    }
    r = gen.generate(batch)
    assert r.responses.shape[0] == 2
    assert len(r.prompt_texts) == 2
    assert "Hello" in r.prompt_texts[0] or "USER" in r.prompt_texts[0]


def test_rollout_generator_bad_batch_raises(device):
    """Batch without messages or instruction raises a clear error."""
    gen, _ = _make_generator(device)
    with pytest.raises(
        ValueError, match="must contain either 'messages' or 'instruction'"
    ):
        gen.generate({"input_ids": torch.zeros(2, 4, dtype=torch.long)})


def _make_runner(device, **kw):
    generator, model = _make_generator(
        device,
        group_size=kw.get("group_size", 2),
        max_tokens=kw.get("max_tokens", 8),
        max_batch_size=kw.get("max_batch_size", 8),
        max_len=kw.get("max_position_embeddings", 128),
    )
    rm = ConstantRewardModel(1.0)
    return (
        RolloutRunner(
            generator=generator,
            reward_model=rm,
            rollout_interval=kw.get("rollout_interval", 2),
            max_policy_lag=kw.get("max_policy_lag"),
        ),
        model,
    )


def test_rollout_runner_shapes(device):
    runner, _ = _make_runner(device, group_size=3, max_tokens=5)
    batch = _make_instruction_batch(n=2)
    r, is_fresh = runner(batch)
    assert is_fresh
    assert r.responses.shape == (2, 3, 5)
    assert r.response_mask.shape == (2, 3, 5)
    assert r.rewards.shape == (2, 3)
    assert r.logprobs_old.shape == (2, 3, 5)
    assert len(r.prompt_texts) == 2
    assert len(r.response_texts) == 2
    assert len(r.response_texts[0]) == 3


def test_rollout_runner_cache_returns_stale_flag(device):
    runner, _ = _make_runner(device, rollout_interval=10)
    batch = _make_instruction_batch()
    r1, fresh1 = runner(batch)
    r2, fresh2 = runner(batch)
    assert r1 is r2
    assert fresh1 is True
    assert fresh2 is False


def test_rollout_runner_evaluate_leaves_cache_untouched(device):
    runner, _ = _make_runner(device, rollout_interval=10)
    batch = _make_instruction_batch()
    cached, _ = runner(batch)

    eval_batch = _make_instruction_batch(n=1)
    result = runner.evaluate(eval_batch)

    assert result.rewards.shape == result.responses.shape[:2]
    replayed, fresh = runner(batch)
    assert replayed is cached
    assert fresh is False
    assert runner._steps_since_rollout == 0


def test_rollout_runner_tags_generation_version_and_preserves_cached_behavior(device):
    runner, _ = _make_runner(device, rollout_interval=100)
    batch = _make_instruction_batch(n=1)

    first, first_fresh = runner(batch)
    assert first_fresh is True
    assert first.policy_version == 0

    assert runner.update_weights(1) == 1
    cached, cached_fresh = runner(batch)
    assert cached is first
    assert cached_fresh is False
    assert cached.policy_version == 0

    runner.clear_cache()
    refreshed, refreshed_fresh = runner(batch)
    assert refreshed_fresh is True
    assert refreshed.policy_version == 1


def test_rollout_runner_rejects_future_generation_version(device):
    runner, _ = _make_runner(device, rollout_interval=2)
    raw = runner.generator.generate(_make_instruction_batch(n=1))
    raw.policy_version = runner.policy_version + 1
    runner.generator.generate = lambda _batch: raw

    with pytest.raises(RolloutVersionError, match="future policy version"):
        runner(_make_instruction_batch(n=1))


def test_rollout_runner_rejects_result_beyond_max_policy_lag(device):
    runner, _ = _make_runner(device, rollout_interval=4, max_policy_lag=1)
    batch = _make_instruction_batch(n=1)
    result, _ = runner(batch)
    assert result.policy_version == 0

    runner.update_weights(2)
    with pytest.raises(RolloutVersionError, match="exceeds max_policy_lag=1"):
        runner(batch)


def test_rollout_runner_revalidates_version_after_async_scoring(device):
    runner, _ = _make_runner(device, rollout_interval=4, max_policy_lag=0)
    original_score = runner._score

    def score_while_policy_advances(raw):
        result = original_score(raw)
        runner.update_weights(1)
        return result

    runner._score = score_while_policy_advances

    with pytest.raises(RolloutVersionError, match="exceeds max_policy_lag=0"):
        runner(_make_instruction_batch(n=1))
    assert runner._cache is None


def test_rollout_runner_publishes_cache_before_concurrent_policy_update(device):
    runner, _ = _make_runner(device, rollout_interval=4, max_policy_lag=1)
    final_validation_started = threading.Event()
    allow_final_validation_to_finish = threading.Event()
    update_finished = threading.Event()
    rollout_finished = threading.Event()
    validation_calls = 0
    original_validate = runner._validate_policy_version

    def blocking_validate(result, *, live_version=None):
        nonlocal validation_calls
        validation_calls += 1
        original_validate(result, live_version=live_version)
        if validation_calls == 3:
            final_validation_started.set()
            assert allow_final_validation_to_finish.wait(timeout=5)

    runner._validate_policy_version = blocking_validate

    def produce_rollout():
        runner(_make_instruction_batch(n=1))
        rollout_finished.set()

    _assert_interleaved(
        produce_rollout,
        lambda: runner.apply_weight_update(1, lambda _version: update_finished.set()),
        started=final_validation_started,
        release=allow_final_validation_to_finish,
        finished=update_finished,
    )

    assert rollout_finished.is_set()
    assert update_finished.is_set()
    assert runner._cache is not None
    assert runner._cache.policy_version == 0
    assert runner.policy_version == 1


def test_rollout_runner_derives_default_policy_lag_from_interval(device):
    runner, _ = _make_runner(device, rollout_interval=4)
    assert runner.max_policy_lag == 3


def _interleave_before_snapshot(runner, callback):
    """Wrap ``with_policy_snapshot`` so ``callback`` runs just before a
    named snapshot callback enters the generator/scheduler locks."""
    original_snapshot = runner.generator.with_policy_snapshot

    def wrapper(inspect):
        if inspect.__name__ == "reuse":
            callback()
        return original_snapshot(inspect)

    runner.generator.with_policy_snapshot = wrapper


def test_rollout_runner_reuse_reads_cache_inside_the_snapshot(device):
    """The reuse decision must observe the cache under the policy snapshot
    (regression: the cache was read outside the lock, so a concurrent
    commit between the read and the lock silently handed the trainer a
    stale rollout — a lost update)."""
    import dataclasses

    runner, _ = _make_runner(device, rollout_interval=100)
    batch = _make_instruction_batch(n=1)
    first, _ = runner(batch)
    assert first.policy_version == 0

    def concurrent_refresh():
        runner.update_weights(1)
        runner._cache = dataclasses.replace(first, policy_version=1)
        runner._steps_since_rollout = 0

    _interleave_before_snapshot(runner, concurrent_refresh)
    result, fresh = runner(batch)
    assert fresh is False
    assert result is not first
    assert result.policy_version == 1


def test_rollout_runner_recovers_when_cache_cleared_before_reuse_snapshot(device):
    """A cache clear between the reuse decision and the snapshot must
    trigger a fresh rollout instead of an assertion failure (regression:
    ``assert cached is not None`` fired because the object was captured
    outside the lock)."""
    runner, _ = _make_runner(device, rollout_interval=100)
    batch = _make_instruction_batch(n=1)
    first, _ = runner(batch)

    _interleave_before_snapshot(runner, runner.clear_cache)
    result, fresh = runner(batch)
    assert fresh is True
    assert result is not first


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rollout_interval": 0}, "rollout_interval must be positive"),
        ({"max_policy_lag": -1}, "max_policy_lag must be non-negative"),
    ],
)
def test_rollout_runner_rejects_invalid_version_window(device, kwargs, message):
    generator, _ = _make_generator(device)
    with pytest.raises(ValueError, match=message):
        RolloutRunner(generator, ConstantRewardModel(), **kwargs)


def test_rollout_runner_refreshes_for_different_batch(device):
    runner, _ = _make_runner(device, rollout_interval=100)
    r1, fresh1 = runner(_make_instruction_batch(n=1))
    batch2 = {"instruction": ["Different prompt"], "input": [""]}
    r2, fresh2 = runner(batch2)
    assert fresh1 is True
    assert fresh2 is True
    assert r2 is not r1


@pytest.mark.parametrize("reward_model", [BadShapeRewardModel, NonFiniteRewardModel])
def test_rollout_runner_rejects_invalid_rewards(device, reward_model):
    generator, _ = _make_generator(device, group_size=2, max_tokens=2)
    runner = RolloutRunner(generator, reward_model(), rollout_interval=1)
    with pytest.raises(ValueError):
        runner(_make_instruction_batch(n=1))


def test_rollout_runner_step_triggers_new_rollout(device):
    runner, _ = _make_runner(device, rollout_interval=2)
    batch = _make_instruction_batch()
    r1, fresh1 = runner(batch)
    assert fresh1 is True
    runner.step()
    # interval=2 means trigger when _steps_since_rollout >= 2; 1 step not enough
    r2, fresh2 = runner(batch)
    assert r2 is r1
    assert fresh2 is False
    runner.step()
    # Now _steps_since_rollout == 2 -> re-rollout
    r3, fresh3 = runner(batch)
    assert r3 is not r1
    assert fresh3 is True


def test_rollout_runner_clear_cache_forces_rerun(device):
    runner, _ = _make_runner(device, rollout_interval=100)
    batch = _make_instruction_batch()
    r1, _ = runner(batch)
    runner.clear_cache()
    r2, fresh2 = runner(batch)
    assert r2 is not r1
    assert fresh2 is True


def test_rollout_runner_release_clears_cache_and_resumes(device):
    runner, _ = _make_runner(device, rollout_interval=100)
    batch = _make_instruction_batch(n=1)
    first, _ = runner(batch)
    assert runner._cache is first

    assert runner.release() is True
    assert runner._cache is None
    assert runner._cache_key is None
    assert runner.generator.backend.runtime_released is True
    with pytest.raises(RuntimeError, match="call resume"):
        runner(batch)

    assert runner.resume() is True
    refreshed, is_fresh = runner(batch)
    assert is_fresh is True
    assert refreshed is not first


def test_rollout_runner_step_resets_counter(device):
    runner, _ = _make_runner(device, rollout_interval=1)
    batch = _make_instruction_batch()
    r1, _ = runner(batch)
    runner.step()
    r2, fresh2 = runner(batch)
    assert r2 is not r1
    assert fresh2 is True
    # Counter reset after rollout; second call w/o step should be cached.
    r3, fresh3 = runner(batch)
    assert r3 is r2
    assert fresh3 is False
