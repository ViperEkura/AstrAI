import copy
import json
import os
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from astrai.config import ComponentSpec, TrainConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel.setup import find_free_port
from astrai.serialization import Checkpoint
from astrai.trainer.callbacks import TrainCallback
from astrai.trainer.rollout.async_round import AsyncRoundCoordinator
from astrai.trainer.rollout.nccl_transport import NCCLWeightChannel
from astrai.trainer.rollout.setup import configure_rollout
from astrai.trainer.rollout.types import ResolvedAsyncRolloutConfig, SamplingParams
from astrai.trainer.trainer import Trainer
from tests.support.rollout import (
    InstructionDataset,
    LengthRewardModel,
    instruction_collate_fn,
)
from tests.support.tokenizers import CHAT_TEMPLATE
from tests.support.trainer import (
    make_online_value_model,
)


@pytest.fixture
def online_env(base_test_env):
    env = dict(base_test_env)
    env["tokenizer"] = copy.deepcopy(env["tokenizer"])
    env["tokenizer"].set_chat_template(CHAT_TEMPLATE)
    env["tokenizer"].save_pretrained(env["test_dir"])
    return env


def _online_config(env, *, dataset_length=4, **overrides):
    defaults = dict(
        strategy="online_grpo",
        model=ComponentSpec(
            type="autoregressive_lm",
            params={"config": env["transformer_config"].to_dict()},
        ),
        precision={"compute_dtype": "float32"},
        dataset=ComponentSpec(
            type="tests.support.rollout:InstructionDataset",
            params={
                "length": dataset_length,
                "collate": "tests.support.rollout:instruction_collate_fn",
            },
        ),
        optimizer=ComponentSpec(type="torch.optim:AdamW", params={"lr": 1e-4}),
        scheduler=ComponentSpec(
            type="cosine",
            params={"warmup_steps": 1, "lr_decay_steps": 4, "min_rate": 0.05},
        ),
        ckpt_dir=os.path.join(env["test_dir"], "ckpt"),
        n_epoch=1,
        batch_per_device=2,
        ckpt_interval=100,
        grad_accum_steps=1,
        random_seed=42,
        device_type=env["device"],
        dp_mode="none",
        strategy_kwargs={"clip_eps": 0.2, "kl_coef": 0.01, "group_size": 2},
        rollout_interval=1,
        rollout_temperature=1.0,
        rollout_top_k=0,
        rollout_top_p=1.0,
        rollout_max_tokens=4,
        reward_model_fn=LengthRewardModel,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


_ONLINE_STRATEGIES = [
    pytest.param(
        "online_grpo",
        {"clip_eps": 0.2, "kl_coef": 0.01, "group_size": 2},
        None,
        id="grpo",
    ),
    pytest.param("online_dpo", {"beta": 0.1, "group_size": 2}, None, id="dpo"),
    pytest.param(
        "online_ppo",
        {
            "clip_eps": 0.2,
            "kl_coef": 0.01,
            "group_size": 2,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
        },
        True,
        id="ppo",
    ),
]

_DDP_TEST_WORLD_SIZE = int(os.environ.get("ASTRAI_DDP_TEST_WORLD_SIZE", "2"))


@pytest.mark.parametrize(
    ("strategy", "strategy_kwargs", "with_critic"), _ONLINE_STRATEGIES
)
def test_online_rollout_end_to_end(online_env, strategy, strategy_kwargs, with_critic):
    """Run one epoch of online RL rollout with KV-cache-backed generation."""
    test_dir = online_env["test_dir"]
    model_config = online_env["transformer_config"]

    config_kwargs = {"strategy": strategy, "strategy_kwargs": strategy_kwargs}
    if with_critic:
        config_kwargs["critic_model_fn"] = partial(
            make_online_value_model, model_config
        )
        config_kwargs["critic_optimizer_fn"] = make_online_optimizer
    train_config = _online_config(online_env, **config_kwargs)

    trainer = Trainer(train_config)
    trainer.train(param_path=test_dir)

    checkpoint_dir = os.path.join(test_dir, "ckpt", "epoch_0_step_2")
    assert os.path.isdir(checkpoint_dir)
    checkpoint = Checkpoint.load(checkpoint_dir)
    assert checkpoint.meta["policy_version"] == 2
    if with_critic:
        assert "value_model" in checkpoint.extra
        assert "value_optimizer" in checkpoint.extra
    else:
        assert "value_model" not in checkpoint.extra


def test_online_config_rejects_contradictory_policy_lag(online_env):
    """rollout_max_policy_lag below rollout_interval - 1 guarantees a fatal
    RolloutVersionError mid-training; it must fail at config time instead."""
    with pytest.raises(ValueError, match="rollout_max_policy_lag=0"):
        _online_config(online_env, rollout_interval=3, rollout_max_policy_lag=0)

    # lag == interval - 1 (including the derived default) stays valid.
    config = _online_config(online_env, rollout_interval=3, rollout_max_policy_lag=2)
    assert config.rollout_max_policy_lag == 2
    config = _online_config(online_env, rollout_interval=3)
    assert config.rollout_max_policy_lag is None

    # Offline strategies never consult the rollout window.
    config = _online_config(
        online_env, strategy="sft", rollout_interval=3, rollout_max_policy_lag=0
    )
    assert config.rollout_max_policy_lag == 0


def test_rollout_val_overrides_drop_unset_fields(online_env):
    """Only explicitly set validation fields enter the replace() dict."""
    config = _online_config(online_env, rollout_val_temperature=0.0)
    assert config.rollout_val_overrides() == {"temperature": 0.0}

    config = _online_config(
        online_env,
        rollout_val_temperature=0.0,
        rollout_val_top_p=0.95,
        rollout_val_top_k=0,
        rollout_val_max_tokens=256,
        rollout_val_group_size=1,
    )
    assert config.rollout_val_overrides() == {
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 0,
        "max_tokens": 256,
        "group_size": 1,
    }

    # Unset fields stay legal (inherit) while out-of-range values are
    # rejected at config time; temperature=0 (greedy) is val-only legal.
    with pytest.raises(ValueError, match="rollout_val_top_p"):
        _online_config(online_env, rollout_val_top_p=0.0)
    with pytest.raises(ValueError, match="rollout_val_group_size"):
        _online_config(online_env, rollout_val_group_size=0)


def test_async_round_rejects_moe_before_starting_workers(online_env):
    config = _online_config(
        online_env,
        rollout_mode="async_round",
        rollout_devices=["cuda:1", "cuda:2", "cuda:3", "cuda:4"],
        rollout_interval=1,
        rollout_max_policy_lag=1,
        device_type="cuda",
        dp_mode="none",
        grad_accum_steps=1,
    )
    model = torch.nn.Linear(2, 2)
    inference_model = SimpleNamespace(
        config=SimpleNamespace(ffn_type="moe", max_position_embeddings=8)
    )
    context = SimpleNamespace(
        strategy=SimpleNamespace(supports_online=lambda: True),
        executor=SimpleNamespace(model_for_inference=lambda _model: inference_model),
        model=model,
        checkpoint=None,
        optimizer_step=0,
    )
    tokenizer_cls = SimpleNamespace(from_pretrained=lambda _path: object())
    with pytest.raises(ValueError, match="dense models only"):
        configure_rollout(
            context,
            config,
            param_path="unused",
            strategy_kwargs={"group_size": 2},
            create_ref_model=lambda **_kwargs: None,
            validate=lambda _executor: None,
            engine_core_cls=object,
            tokenizer_cls=tokenizer_cls,
        )


@pytest.mark.distributed
@pytest.mark.cuda
@pytest.mark.skipif(
    torch.cuda.device_count() < _DDP_TEST_WORLD_SIZE,
    reason=f"{_DDP_TEST_WORLD_SIZE} CUDA devices are required",
)
def test_ddp_online_grpo_end_to_end(online_env):
    """Run real rollout and optimizer steps on the requested DDP replicas."""
    test_dir = online_env["test_dir"]

    dataset = InstructionDataset(length=40)
    train_config = _online_config(
        online_env,
        dataset_length=40,
        batch_per_device=1,
        device_type="cuda",
        dp_size=_DDP_TEST_WORLD_SIZE,
        dp_mode="ddp",
    )

    Trainer(train_config).train(param_path=test_dir)

    expected_steps = (len(dataset) + _DDP_TEST_WORLD_SIZE - 1) // _DDP_TEST_WORLD_SIZE
    assert Path(test_dir, "ckpt", f"epoch_0_step_{expected_steps}").is_dir()


@pytest.mark.distributed
@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="four CUDA devices required")
def test_async_round_ddp_two_learners_two_workers(online_env):
    test_dir = online_env["test_dir"]
    config = _online_config(
        online_env,
        device_type="cuda",
        dp_size=2,
        dp_mode="ddp",
        rollout_mode="async_round",
        rollout_devices=["cuda:2", "cuda:3"],
        random_seed=3407,
        rollout_temperature=0.7,
        rollout_top_p=0.9,
        ckpt_interval=5000,
    )
    Trainer(config).train(param_path=test_dir)
    checkpoint = Checkpoint.load(os.path.join(test_dir, "ckpt", "epoch_1_step_1"))
    assert checkpoint.meta["policy_version"] == 1
    assert checkpoint.consumed_samples == 4


@pytest.mark.distributed
@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="five CUDA devices required")
def test_async_round_online_grpo_five_gpus(online_env, monkeypatch):
    """Four simultaneous replicas feed one optimizer update per round."""
    test_dir = online_env["test_dir"]

    seen = []
    pools = []
    original_init = AsyncRoundCoordinator.__init__
    original_collect = AsyncRoundCoordinator.collect_round

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        pools.append(self)

    def tracked_collect(self, handle):
        result = original_collect(self, handle)
        seen.append(result.policy_version)
        return result

    monkeypatch.setattr(AsyncRoundCoordinator, "__init__", tracked_init)
    monkeypatch.setattr(AsyncRoundCoordinator, "collect_round", tracked_collect)
    config = _online_config(
        online_env,
        dataset_length=10,
        batch_per_device=4,
        device_type="cuda",
        rollout_mode="async_round",
        rollout_devices=["cuda:1", "cuda:2", "cuda:3", "cuda:4"],
        rollout_max_policy_lag=1,
        random_seed=3407,
        rollout_temperature=0.7,
        rollout_top_p=0.9,
        ckpt_interval=5000,
    )
    Trainer(config).train(param_path=test_dir)

    checkpoint = Checkpoint.load(os.path.join(test_dir, "ckpt", "epoch_1_step_3"))
    assert checkpoint.meta["policy_version"] == 3
    assert checkpoint.consumed_samples == 10
    assert seen == [0, 0, 1]
    assert len({process.pid for process in pools[0]._processes}) == 4
    assert set(pools[0].worker_cuda_graph_enabled) == {
        "cuda:1",
        "cuda:2",
        "cuda:3",
        "cuda:4",
    }
    assert all(
        isinstance(enabled, bool)
        for enabled in pools[0].worker_cuda_graph_enabled.values()
    )

    resume_config = replace(config, n_epoch=2)
    Trainer(resume_config).train(
        param_path=os.path.join(test_dir, "ckpt", "epoch_1_step_3"), resume=True
    )
    resumed = Checkpoint.load(os.path.join(test_dir, "ckpt", "epoch_2_step_6"))
    assert resumed.meta["policy_version"] == 6
    assert resumed.consumed_samples == 20


@pytest.mark.distributed
@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="five CUDA devices required")
def test_async_round_version_commit_failure_checkpoint_resumes_once(
    online_env, monkeypatch
):
    test_dir = online_env["test_dir"]
    config = _online_config(
        online_env,
        dataset_length=8,
        batch_per_device=4,
        device_type="cuda",
        rollout_mode="async_round",
        rollout_devices=["cuda:1", "cuda:2", "cuda:3", "cuda:4"],
        rollout_max_policy_lag=1,
        random_seed=3407,
        rollout_temperature=0.7,
        rollout_top_p=0.9,
        ckpt_interval=5000,
    )
    original_publication = NCCLWeightChannel.mark_committed

    def fail_after_commit(self, version):
        if version == 1:
            raise RuntimeError("injected version commit failure")
        return original_publication(self, version)

    monkeypatch.setattr(NCCLWeightChannel, "mark_committed", fail_after_commit)
    with pytest.raises(RuntimeError, match="injected version commit failure"):
        Trainer(config).train(param_path=test_dir)
    checkpoint_path = os.path.join(test_dir, "ckpt", "epoch_0_step_1")
    checkpoint = Checkpoint.load(checkpoint_path)
    assert checkpoint.meta["policy_version"] == 1
    assert checkpoint.meta["optimizer_step"] == 1
    assert checkpoint.consumed_samples == 4

    monkeypatch.setattr(NCCLWeightChannel, "mark_committed", original_publication)
    Trainer(config).train(param_path=checkpoint_path, resume=True)
    resumed = Checkpoint.load(os.path.join(test_dir, "ckpt", "epoch_1_step_2"))
    assert resumed.meta["policy_version"] == 2
    assert resumed.meta["optimizer_step"] == 2
    assert resumed.consumed_samples == 8


def make_online_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-4)


def make_graph_rollout_model(config):
    return AutoRegressiveLM(config).to(dtype=torch.bfloat16)


@pytest.mark.distributed
@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="five CUDA devices required")
def test_async_seeds_match_after_worker_repartition_and_resume(online_env):
    config = replace(
        online_env["transformer_config"], hidden_size=128, intermediate_size=256
    )
    model_fn = partial(make_graph_rollout_model, config)
    source = model_fn().to("cuda:0").eval()
    batch = instruction_collate_fn([InstructionDataset()[i % 4] for i in range(8)])
    results = []
    for devices in (
        ["cuda:1", "cuda:2"],
        ["cuda:1", "cuda:2", "cuda:3", "cuda:4"],
        ["cuda:1", "cuda:2"],
    ):
        per_worker = 8 // len(devices)
        pool = AsyncRoundCoordinator(
            config=ResolvedAsyncRolloutConfig(
                learner_device="cuda:0",
                devices=tuple(devices),
                params=SamplingParams(
                    group_size=2, max_tokens=6, temperature=0.7, top_k=0, top_p=1.0
                ),
                max_seq_len=64,
                max_prompts_per_worker=per_worker,
                startup_timeout_s=300.0,
                generation_timeout_s=600.0,
                weight_timeout_s=600.0,
                max_policy_lag=1,
                random_seed=42,
                sample_cursor=10,
            ),
            source=source,
            model_fn=model_fn,
            param_path=online_env["test_dir"],
            reward_model=LengthRewardModel(),
            policy_version=0,
            model_dtype=torch.bfloat16,
        )
        try:
            results.append(pool.collect_round(pool.submit_round(batch)))
            assert all(pool.worker_cuda_graph_enabled.values())
        finally:
            pool.close()
        assert all(not p.is_alive() for p in pool._processes)
    for other in results[1:]:
        assert torch.equal(results[0].responses, other.responses)
        assert torch.equal(results[0].response_mask, other.response_mask)
        torch.testing.assert_close(
            results[0].logprobs_old, other.logprobs_old, atol=1e-5, rtol=1e-5
        )


class _StopAtFour(TrainCallback):
    def after_optimizer_step(self, context):
        if context.optimizer_step == 4:
            context.request_stop()


def test_online_replay_publishes_each_update_and_reference(online_env):
    """Replay publishes each update; the cursor advances once per data window."""
    test_dir = online_env["test_dir"]

    config = _online_config(online_env, rl_update_epochs=2, rl_minibatch_prompts=1)
    Trainer(config).train(param_path=test_dir)

    checkpoint_dir = os.path.join(test_dir, "ckpt", "epoch_0_step_8")
    checkpoint = Checkpoint.load(checkpoint_dir)
    # 2 batches × (2 minibatches × 2 epochs) publications, one step each.
    assert checkpoint.meta["policy_version"] == 8
    assert checkpoint.meta["optimizer_step"] == 8
    # The cursor counts data positions, not steps: a replayed batch is one
    # position however many updates it produces.
    assert checkpoint.meta["consumed_samples"] == 4
    assert "reference_model" in checkpoint.extra
    assert "rng_state" not in checkpoint.extra
    ref_keys = set(checkpoint.extra["reference_model"])
    assert ref_keys and ref_keys.issubset(checkpoint.state_dict.keys())


def test_online_accumulation_commits_complete_rollout_windows(online_env):
    """Replay updates publish separately; the data cursor moves once per window."""
    test_dir = online_env["test_dir"]

    config = _online_config(
        online_env,
        batch_per_device=1,
        grad_accum_steps=2,
        rl_update_epochs=2,
        rl_minibatch_prompts=1,
    )
    Trainer(config).train(param_path=test_dir)
    checkpoint = Checkpoint.load(os.path.join(test_dir, "ckpt", "epoch_0_step_8"))
    assert checkpoint.meta["policy_version"] == 8
    assert checkpoint.meta["optimizer_step"] == 8
    assert checkpoint.meta["consumed_samples"] == 4
    assert checkpoint.meta["epoch_sample_offset"] == 4
    with open(
        os.path.join(test_dir, "ckpt", "epoch_0_step_8", "metric.jsonl"),
        encoding="utf-8",
    ) as file:
        steps = [
            record["step"]
            for record in map(json.loads, file)
            if record["type"] == "step"
        ]
    assert steps == list(range(1, 9))


def test_online_ddp_accumulation_keeps_policy_versions_aligned(online_env):
    """All ranks publish the same replay updates, including empty group slots."""
    test_dir = online_env["test_dir"]

    config = _online_config(
        online_env,
        dp_size=2,
        dp_mode="ddp",
        backend="gloo",
        master_port=find_free_port(),
        batch_per_device=1,
        grad_accum_steps=2,
        rl_update_epochs=2,
        rl_minibatch_prompts=2,
    )
    Trainer(config).train(param_path=test_dir)
    checkpoint = Checkpoint.load(os.path.join(test_dir, "ckpt", "epoch_0_step_4"))
    assert checkpoint.meta["policy_version"] == 4
    assert checkpoint.meta["optimizer_step"] == 4
    assert checkpoint.meta["consumed_samples"] == 4


def test_online_rollout_window_resumes_after_dp_resize(online_env):
    test_dir = online_env["test_dir"]
    old = _online_config(
        online_env,
        dataset_length=8,
        batch_per_device=2,
        grad_accum_steps=2,
        rl_update_epochs=2,
        rl_minibatch_prompts=2,
    )
    Trainer(old, callbacks=[_StopAtFour()]).train(param_path=test_dir)
    first = os.path.join(test_dir, "ckpt", "epoch_0_step_4")
    saved = Checkpoint.load(first)
    assert saved.consumed_samples == 4
    assert saved.meta["policy_version"] == 4

    new = _online_config(
        online_env,
        dataset_length=8,
        dp_size=2,
        dp_mode="ddp",
        backend="gloo",
        master_port=find_free_port(),
        batch_per_device=1,
        grad_accum_steps=2,
        rl_update_epochs=2,
        rl_minibatch_prompts=2,
    )
    Trainer(new).train(param_path=first, resume=True)
    final = Checkpoint.load(os.path.join(test_dir, "ckpt", "epoch_0_step_8"))
    assert final.meta["policy_version"] == 8
    assert final.meta["optimizer_step"] == 8
    assert final.consumed_samples == 8


def test_online_ddp_flushes_tail_with_empty_rank(online_env):
    test_dir = online_env["test_dir"]
    config = _online_config(
        online_env,
        dataset_length=5,
        dp_size=2,
        dp_mode="ddp",
        backend="gloo",
        master_port=find_free_port(),
        batch_per_device=1,
        grad_accum_steps=2,
        epoch_tail_policy="flush",
        rl_minibatch_prompts=2,
    )
    Trainer(config).train(param_path=test_dir)
    checkpoint = Checkpoint.load(os.path.join(test_dir, "ckpt", "epoch_0_step_3"))
    assert checkpoint.meta["policy_version"] == 3
    assert checkpoint.consumed_samples == 5
    assert checkpoint.meta["epoch_sample_offset"] == 5
