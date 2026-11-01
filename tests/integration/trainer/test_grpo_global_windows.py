"""CPU/Gloo oracles for GRPO global objectives and learner microbatches."""

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from astrai.config import ComponentSpec, TrainConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel.executor import BaseExecutor
from astrai.trainer.callbacks import TrainCallback
from astrai.trainer.context import TrainContext, TrainContextBuilder
from astrai.trainer.rollout.batching import slice_batch
from astrai.trainer.strategy import GRPOStrategy
from astrai.trainer.strategy.ops import rollout_token_logprobs
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


def _rows(batch, begin, end):
    return slice_batch(batch, slice(begin, end), batch["prompts"].shape[0])


def _chunks(local, count):
    """Two dataloader chunks, a placeholder row standing in for an empty one."""
    split = min(1, count)
    pieces = []
    for begin, end in ((0, split), (split, count)):
        if end > begin:
            pieces.append((_rows(local, begin, end), end - begin))
        else:
            pieces.append((_rows(local, 0, 1), 0))
    return pieces


def _config(path, **extra):
    values = dict(
        strategy="grpo",
        model=ComponentSpec(
            type="tests.integration.trainer.test_grpo_global_windows:_actor"
        ),
        dataset=ComponentSpec(
            type="tests.support.rollout:RandomTokenDataset",
            params={"length": 6, "max_length": 4, "vocab_size": 31},
        ),
        optimizer=ComponentSpec(type="torch.optim:AdamW", params={"lr": 0.001}),
        scheduler=ComponentSpec(
            type="torch.optim.lr_scheduler:StepLR",
            params={"step_size": 1, "gamma": 1.0},
        ),
        num_workers=0,
        pin_memory=False,
        epoch_tail_policy="flush",
        device_type="cpu",
        dp_mode="none",
        batch_per_device=3,
        n_epoch=1,
        ckpt_dir=str(path),
        ckpt_interval=1,
    )
    values.update(extra)
    return TrainConfig(**values)


class _BatchProbe(TrainCallback):
    def __init__(self):
        self.begins = self.ends = 0

    def on_batch_begin(self, context):
        self.begins += 1

    def on_batch_end(self, context):
        self.ends += 1


def _ddp_oracle(rank, world, rendezvous, path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        cases = [
            (aggregation, group_size, kl, microbatch, empty_rank)
            for aggregation in ("token", "sequence")
            for group_size in (2, 4)
            for kl in (0.0, 0.2)
            for microbatch in (None, 1, 2)
            for empty_rank in (False, True)
        ]
        for aggregation, group_size, kl, microbatch, empty_rank in cases:
            torch.manual_seed(3407)
            expected = _actor()
            actual = DistributedDataParallel(deepcopy(expected))
            reference = deepcopy(expected).requires_grad_(False)
            with torch.no_grad():
                reference.lm_head.weight[1].add_(0.05)
            # Unequal local prompt counts give unequal microbatch schedules.
            counts = [1 + peer % 3 for peer in range(world)]
            if empty_rank:
                counts[0] = 0  # a rank without prompt groups still joins
            batch = _batch(expected, sum(counts), group_size)
            if not empty_rank:
                batch["masks"][: counts[0]].zero_()  # an empty-token rank
            begin = sum(counts[:rank])
            local = (
                _rows(batch, begin, begin + counts[rank])
                if counts[rank]
                else _rows(batch, 0, 1)
            )
            oracle = GRPOStrategy(
                expected,
                "cpu",
                old_model=None,
                ref_model=reference,
                loss_aggregation=aggregation,
                kl_coef=kl,
            )
            executor = BaseExecutor()
            context = TrainContext(
                model=actual,
                optimizer=torch.optim.SGD(actual.parameters(), lr=0.01, momentum=0.9),
                executor=executor,
                strategy=GRPOStrategy(
                    actual,
                    "cpu",
                    old_model=None,
                    ref_model=reference,
                    executor=executor,
                    loss_aggregation=aggregation,
                    kl_coef=kl,
                ),
                config=_config(path, rl_microbatch_prompts=microbatch),
                world_size=world,
                rank=rank,
            )
            trainer = Trainer(context.config)
            trainer.callbacks = []
            opt_expected = torch.optim.SGD(expected.parameters(), lr=0.01, momentum=0.9)
            for step in range(3):
                output = oracle.compute_loss_output(batch)
                output["loss"].backward()
                opt_expected.step()
                opt_expected.zero_grad()
                trainer._run_global_update(
                    context,
                    _chunks(local, counts[rank]),
                    sum(counts),
                    sum(counts) * (step + 1),
                    True,
                )
                assert context.optimizer_step == step + 1
                assert context.consumed_samples == sum(counts) * (step + 1)
                assert context.loss == pytest.approx(
                    output["metrics"]["loss"], rel=2e-5, abs=2e-6
                )
                for wanted, got in zip(
                    expected.parameters(), actual.module.parameters()
                ):
                    torch.testing.assert_close(got, wanted, rtol=2e-5, atol=3e-6)
                    torch.testing.assert_close(
                        context.optimizer.state[got]["momentum_buffer"],
                        opt_expected.state[wanted]["momentum_buffer"],
                        rtol=3e-4,
                        atol=1e-5,
                    )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_global_microbatch_gradient_and_multi_update_oracle(tmp_path, world):
    mp.spawn(
        _ddp_oracle,
        args=(world, "file://" + str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=world,
        join=True,
    )


@pytest.mark.parametrize("aggregation", ["token", "sequence"])
@pytest.mark.parametrize("microbatch", [None, 1, 2])
def test_global_trainer_microbatches_match_one_full_objective(
    tmp_path, monkeypatch, aggregation, microbatch
):
    torch.manual_seed(3407)
    oracle_model = _actor()
    model = deepcopy(oracle_model)
    reference = deepcopy(oracle_model).requires_grad_(False)
    executor = BaseExecutor(grad_accum_steps=2)
    oracle = GRPOStrategy(
        oracle_model,
        "cpu",
        old_model=None,
        ref_model=reference,
        loss_aggregation=aggregation,
        kl_coef=0.2,
    )
    strategy = GRPOStrategy(
        model,
        "cpu",
        old_model=None,
        ref_model=reference,
        executor=executor,
        loss_aggregation=aggregation,
        kl_coef=0.2,
    )
    forwards = []
    original = strategy.compute_loss_output

    def counted(batch):
        forwards.append(batch["prompts"].shape[0])
        return original(batch)

    monkeypatch.setattr(strategy, "compute_loss_output", counted)
    context = TrainContext(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.001),
        executor=executor,
        strategy=strategy,
        config=_config(tmp_path, grad_accum_steps=2, rl_microbatch_prompts=microbatch),
    )
    trainer = Trainer(context.config)
    probe = _BatchProbe()
    trainer.callbacks = [probe]
    expected_optimizer = torch.optim.AdamW(oracle_model.parameters(), lr=0.001)
    batch = _batch(oracle_model, 3, 4)
    chunks = [(_rows(batch, 0, 1), 1), (_rows(batch, 1, 3), 2)]
    for step in range(3):
        output = oracle.compute_loss_output(batch)
        output["loss"].backward()
        expected_optimizer.step()
        expected_optimizer.zero_grad()
        forwards.clear()
        trainer._run_global_update(context, chunks, 3, 3 * (step + 1), True)
        # Each prompt-group microbatch runs its own forward/backward.
        assert forwards == {None: [1, 2], 1: [1, 1, 1], 2: [1, 2]}[microbatch]
        assert context.optimizer_step == step + 1
        assert context.consumed_samples == 3 * (step + 1)
        assert context.loss == pytest.approx(
            output["metrics"]["loss"], rel=2e-5, abs=2e-6
        )
        for expected, actual in zip(oracle_model.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-6)
            for name, value in expected_optimizer.state[expected].items():
                torch.testing.assert_close(
                    context.optimizer.state[actual][name], value, rtol=3e-4, atol=3e-6
                )
    # Batch callbacks still observe dataloader chunks, not microbatches.
    assert probe.begins == probe.ends == 3 * len(chunks)


def test_rl_microbatch_prompts_config_validation(tmp_path):
    assert _config(tmp_path).rl_microbatch_prompts is None
    assert _config(tmp_path, rl_microbatch_prompts=2).rl_microbatch_prompts == 2
    with pytest.raises(ValueError, match="GRPO only"):
        _config(tmp_path, strategy="seq", rl_microbatch_prompts=1)
    with pytest.raises(ValueError, match="positive"):
        _config(tmp_path, rl_microbatch_prompts=0)
    with pytest.raises(ValueError, match="strategy_kwargs"):
        _config(tmp_path, strategy_kwargs={"rl_microbatch_prompts": 1})


class _BuiltPastMoeCheck(Exception):
    pass


@pytest.mark.parametrize("coefficient", [0.1, 0.0])
def test_moe_aux_loss_rejects_learner_microbatches(tmp_path, monkeypatch, coefficient):
    cfg = _config(tmp_path, rl_microbatch_prompts=1, moe_aux_loss_coef=coefficient)
    builder = TrainContextBuilder(cfg)
    monkeypatch.setattr(
        builder,
        "_load_preloaded_state",
        lambda: SimpleNamespace(model_config={"ffn_type": "moe"}),
    )

    def stop():
        raise _BuiltPastMoeCheck

    monkeypatch.setattr(builder, "_create_executor", stop)
    if coefficient:
        with pytest.raises(NotImplementedError, match="router-statistics"):
            builder.build()
    else:
        with pytest.raises(_BuiltPastMoeCheck):
            builder.build()
