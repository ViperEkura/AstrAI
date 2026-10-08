"""CPU/Gloo oracles for global objectives and complete update boundaries."""

from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from astrai.config import TrainConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel.executor import BaseExecutor
from astrai.parallel.topology import ParallelTopology
from astrai.serialization import Checkpoint
from astrai.trainer.callbacks import CheckpointCallback
from astrai.trainer.rollout import RolloutResult
from astrai.trainer.strategy import GRPOStrategy
from astrai.trainer.strategy.ops import rollout_token_logprobs
from astrai.trainer.train_context import TrainContext, TrainContextBuilder
from astrai.trainer.trainer import Trainer
from tests.support.models import make_rollout_config


def _actor():
    return AutoRegressiveLM(
        make_rollout_config(vocab_size=31, num_hidden_layers=1)
    ).eval()


def _batch(model, count, group):
    generator = torch.Generator().manual_seed(118)
    prompts = torch.randint(1, 31, (count, 3), generator=generator)
    responses = torch.randint(1, 31, (count, group, 5), generator=generator)
    lengths = torch.arange(count * group).reshape(count, group) % 6
    mask = torch.arange(5)[None, None, :] < lengths[:, :, None]
    rewards = torch.randn(count, group, generator=generator)
    if count > 1:
        rewards[1].fill_(1)  # zero-variance group remains in the objective
    with torch.no_grad():
        old = rollout_token_logprobs(
            model, prompts, torch.ones_like(prompts, dtype=torch.bool), responses, mask
        )["logprobs"]
    return dict(
        prompts=prompts,
        prompt_mask=torch.ones_like(prompts, dtype=torch.bool),
        responses=responses,
        masks=mask,
        rewards=rewards,
        logprobs_old=old + 0.05,
    )


def _ddp_oracle(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        singleton = None
        for peer in range(world):
            group = dist.new_group([peer])
            if peer == rank:
                singleton = group
        for aggregation in ("token", "sequence"):
            for group_size in (2, 4, 8):
                for kl in (0.0, 0.2):
                    torch.manual_seed(3407)
                    expected = _actor()
                    actual = DistributedDataParallel(deepcopy(expected))
                    reference = deepcopy(expected).requires_grad_(False)
                    with torch.no_grad():
                        reference.lm_head.weight[1].add_(0.05)
                    counts = [1 + peer % 3 for peer in range(world)]
                    batch = _batch(expected, sum(counts), group_size)
                    batch["masks"][: counts[0]].zero_()  # an empty-token rank
                    begin = sum(counts[:rank])
                    local = GRPOStrategy._slice_prompt_groups(
                        batch, begin, begin + counts[rank]
                    )
                    oracle = GRPOStrategy(
                        expected,
                        "cpu",
                        old_model=None,
                        ref_model=reference,
                        loss_aggregation=aggregation,
                        kl_coef=kl,
                        loss_process_group=singleton,
                    )
                    candidate = GRPOStrategy(
                        actual,
                        "cpu",
                        old_model=None,
                        ref_model=reference,
                        loss_aggregation=aggregation,
                        kl_coef=kl,
                        rl_microbatch_prompts=2,
                    )
                    opt_expected = torch.optim.SGD(
                        expected.parameters(), lr=0.01, momentum=0.9
                    )
                    opt_actual = torch.optim.SGD(
                        actual.parameters(), lr=0.01, momentum=0.9
                    )
                    for _ in range(3):
                        output = oracle.compute_loss_output(batch)
                        output["loss"].backward()
                        reported = 0.0
                        updates = 0
                        for update in candidate.training_updates(local):
                            seen = False
                            for micro in update:
                                seen = True
                                reported += micro["metrics"]["loss"]
                                micro["loss"].backward()
                            updates += int(seen)
                        assert updates == 1
                        assert candidate.training_global_prompts == sum(counts)
                        assert reported == pytest.approx(
                            output["metrics"]["loss"], rel=2e-5, abs=2e-6
                        )
                        for wanted, got in zip(
                            expected.parameters(), actual.module.parameters()
                        ):
                            torch.testing.assert_close(
                                got.grad, wanted.grad, rtol=3e-4, atol=3e-6
                            )
                        oracle.optimizer_step(opt_expected)
                        candidate.optimizer_step(opt_actual)
                        for wanted, got in zip(
                            expected.parameters(), actual.module.parameters()
                        ):
                            torch.testing.assert_close(
                                got, wanted, rtol=2e-5, atol=3e-6
                            )
                            torch.testing.assert_close(
                                opt_actual.state[got]["momentum_buffer"],
                                opt_expected.state[wanted]["momentum_buffer"],
                                rtol=3e-4,
                                atol=1e-5,
                            )
                        opt_expected.zero_grad()
                        opt_actual.zero_grad()
        # A rank with no prompt groups still joins the same forward/backward
        # schedule. Padding groups never enter N_global or the data cursor.
        torch.manual_seed(3407)
        expected = _actor()
        actual = DistributedDataParallel(deepcopy(expected))
        counts = [0] + [1 + peer % 3 for peer in range(1, world)]
        batch = _batch(expected, sum(counts), 4)
        batch["masks"].fill_(True)
        reference = deepcopy(expected).requires_grad_(False)
        oracle = GRPOStrategy(
            expected,
            "cpu",
            old_model=None,
            ref_model=reference,
            loss_process_group=singleton,
        )
        oracle.compute_loss(batch).backward()
        begin = sum(counts[:rank])
        local = GRPOStrategy._slice_prompt_groups(batch, begin, begin + counts[rank])
        candidate = GRPOStrategy(
            actual, "cpu", old_model=None, ref_model=reference, rl_microbatch_prompts=1
        )
        for update in candidate.training_updates(local):
            for output in update:
                output["loss"].backward()
        for wanted, got in zip(expected.parameters(), actual.module.parameters()):
            torch.testing.assert_close(got.grad, wanted.grad, rtol=3e-4, atol=3e-6)
        actual.zero_grad()
        local["masks"].zero_()
        assert all(list(update) == [] for update in candidate.training_updates(local))
        assert all(param.grad is None for param in actual.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 8])
def test_global_gradient_and_multi_update_oracle(tmp_path, world):
    mp.spawn(
        _ddp_oracle,
        args=(world, "file://" + str(tmp_path / "rendezvous")),
        nprocs=world,
        join=True,
    )


class _Runner:
    def __init__(self):
        self.calls = self.steps = self.policy_version = 0

    def __call__(self, batch):
        self.calls += 1
        return RolloutResult(
            prompts=batch["prompts"],
            prompt_mask=batch["prompt_mask"],
            responses=batch["responses"],
            response_mask=batch["masks"],
            rewards=batch["rewards"],
            logprobs_old=batch["logprobs_old"],
        ), True

    def apply_weight_update(self, version, update):
        result = update(self.policy_version + 1)
        self.policy_version += 1
        return result

    def step(self):
        self.steps += 1


def _config(tmp_path, **extra):
    values = dict(
        strategy="grpo",
        model_fn=_actor,
        dataset=torch.utils.data.TensorDataset(torch.arange(6)),
        optimizer_fn=lambda m: torch.optim.AdamW(m.parameters(), lr=0.001),
        scheduler_fn=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0
        ),
        device_type="cpu",
        dp_mode="none",
        batch_per_device=3,
        n_epoch=1,
        ckpt_dir=str(tmp_path),
        ckpt_interval=1,
    )
    values.update(extra)
    return TrainConfig(**values)


def test_trainer_round_tail_empty_and_checkpoint_counts(tmp_path, monkeypatch):
    cfg = _config(
        tmp_path, rl_microbatch_prompts=1, rl_minibatch_prompts=2, rl_update_epochs=2
    )
    executor = BaseExecutor()
    model, optimizer, scheduler = executor.prepare(
        _actor, cfg.optimizer_fn, cfg.scheduler_fn
    )
    full = _batch(model, 6, 4)
    full["masks"].fill_(True)
    batches = [
        GRPOStrategy._slice_prompt_groups(full, 0, 3),
        GRPOStrategy._slice_prompt_groups(full, 3, 5),
        GRPOStrategy._slice_prompt_groups(full, 5, 6),
    ]
    batches[-1]["masks"].zero_()
    strategy = GRPOStrategy(
        model,
        "cpu",
        old_model=None,
        ref_model=deepcopy(model).requires_grad_(False),
        executor=executor,
        rl_microbatch_prompts=1,
        rl_minibatch_prompts=2,
        rl_update_epochs=2,
    )
    runner = _Runner()
    strategy.set_rollout_runner(runner)
    context = TrainContext(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        executor=executor,
        config=cfg,
        strategy=strategy,
        dataloader=batches,
        model_config=model.config.to_dict(),
        optimizer_steps=0,
    )

    class Builder:
        def __init__(self, config):
            pass

        def with_param_path(self, *args, **kwargs):
            return self

        def build(self):
            return context

    class Probe:
        steps = []
        rounds = []

        def after_optimizer_step(self, context):
            self.steps.append(context.optimizer_step)

        def on_batch_end(self, context):
            self.rounds.append((context.optimizer_step, context.consumed_samples))
            if context.consumed_samples == 5:
                self.before_empty = deepcopy(context.model.state_dict())
                self.before_optimizer = deepcopy(context.optimizer.state_dict())

    monkeypatch.setattr("astrai.trainer.trainer.TrainContextBuilder", Builder)
    trainer = Trainer(cfg)
    probe = Probe()
    trainer.callbacks = [probe, CheckpointCallback(str(tmp_path), 1)]
    trainer._trainer_loop()
    assert runner.calls == 3 and runner.steps == 6 and runner.policy_version == 6
    assert probe.steps == list(range(1, 7))
    assert probe.rounds == [(4, 3), (6, 5), (6, 6)]
    assert context.metrics["empty_update"] == 1
    assert context.scheduler.scheduler.last_epoch == 6
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, probe.before_empty[key], rtol=0, atol=0)
    for key, value in context.optimizer.state_dict()["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(
                tensor, probe.before_optimizer["state"][key][name], rtol=0, atol=0
            )
    checkpoint = Checkpoint.load_any(tmp_path / "epoch_0_step_6")
    assert checkpoint.consumed_samples == 6
    assert checkpoint.meta["optimizer_steps"] == checkpoint.meta["policy_version"] == 6
    # The short, completed second round has cursor 5, not floor(5 / B) * B.
    tail = Checkpoint(
        state_dict=checkpoint.state_dict,
        config=checkpoint.config,
        consumed_samples=5,
        meta={**checkpoint.meta, "consumed_samples": 5},
    )
    tail.save(tmp_path / "tail")
    builder = TrainContextBuilder(cfg).with_param_path(
        str(tmp_path / "tail"), resume=True
    )
    builder._topology = ParallelTopology(1)
    state = builder._load_preloaded_state()
    resumed = builder._create_context(state, executor)
    assert resumed.consumed_samples == 5 and resumed.optimizer_step == 6


def test_moe_aux_requires_its_own_microbatch_reduction(tmp_path, monkeypatch):
    model = _actor()
    batch = _batch(model, 2, 2)
    strategy = GRPOStrategy(
        model,
        "cpu",
        old_model=None,
        ref_model=deepcopy(model),
        rl_microbatch_prompts=1,
        moe_aux_loss_coef=0.1,
    )
    original = rollout_token_logprobs

    def with_aux(*args, **kwargs):
        output = original(*args, **kwargs)
        output["aux_loss"] = model.lm_head.weight.square().mean()
        return output

    monkeypatch.setattr("astrai.trainer.strategy.grpo.rollout_token_logprobs", with_aux)
    with pytest.raises(NotImplementedError, match="router-statistics"):
        for update in strategy.training_updates(batch):
            list(update)
    strategy.moe_aux_loss_coef = 0
    assert sum(len(list(update)) for update in strategy.training_updates(batch)) == 2
    strategy.rl_microbatch_prompts = None
    strategy.moe_aux_loss_coef = 0.1
    (update,) = strategy.training_updates(batch)
    (output,) = update
    assert output["metrics"]["moe_aux_loss_weighted"] == pytest.approx(
        0.1 * output["metrics"]["moe_aux_loss"]
    )


def test_grpo_requires_explicit_update_microbatch_config(tmp_path):
    with pytest.raises(ValueError, match="grad_accum_steps=1"):
        _config(tmp_path, grad_accum_steps=2)
    with pytest.raises(ValueError, match="GRPO only"):
        _config(tmp_path, strategy="seq", rl_microbatch_prompts=1)
    with pytest.raises(ValueError, match="rl_microbatch_prompts"):
        _config(tmp_path, rl_microbatch_prompts=0)


def test_builder_uses_the_models_actual_gradient_reducer_group(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DEVICE", "cpu")
    cfg = _config(tmp_path)
    builder = TrainContextBuilder(cfg)
    builder._topology = ParallelTopology(1)
    model = _actor()
    reducer_group = object()
    model.process_group = reducer_group
    context = TrainContext(config=cfg, model=model, executor=BaseExecutor())
    builder._create_strategy(context, context.executor)
    assert context.strategy.loss_process_group is reducer_group
