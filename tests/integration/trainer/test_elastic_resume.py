"""The checkpoint cursor must describe global data, independent of DP layout."""

from copy import deepcopy

import pytest
import torch

from astrai.dataset import GlobalBatchSampler
from astrai.parallel.setup import find_free_port
from astrai.parallel.topology import ParallelTopology
from astrai.serialization import Checkpoint
from astrai.trainer.callbacks import TrainCallback
from astrai.trainer.context.resume import PreloadedState, ResumeLoader
from astrai.trainer.trainer import Trainer
from tests.support.trainer import create_train_config


class _StopAfterOneStep(TrainCallback):
    def after_optimizer_step(self, context):
        if context.optimizer_step == 1:
            context.request_stop()


class _TokenDataset(torch.utils.data.Dataset):
    def __init__(self, length=9, max_length=8, vocab_size=100):
        self.length = length
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        tokens = (torch.arange(self.max_length) + index) % self.vocab_size
        return {
            "input_ids": tokens,
            "target_ids": (tokens + 1) % self.vocab_size,
        }


class _MaskedTokenDataset(_TokenDataset):
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        batch["loss_mask"] = torch.arange(self.max_length) < (index % 5 + 1)
        batch["position_ids"] = torch.arange(self.max_length)
        return batch


def _global_samples(length, dp, batch, accum, offset=0, policy="flush"):
    dataset = range(length)
    samplers = [
        GlobalBatchSampler(dataset, dp, rank, batch, accum, 47, policy, 0, offset)
        for rank in range(dp)
    ]
    rank_batches = [iter(sampler) for sampler in samplers]
    observed = []
    for start, count, microbatches in samplers[0].windows():
        window = [None] * count
        for slot in range(microbatches):
            for rank, sampler in enumerate(samplers):
                indices = next(rank_batches[rank])
                real = sampler.real_count(start, slot)
                assert len(indices) == real or (real == 0 and len(indices) == 1)
                for local, sample in enumerate(indices[:real]):
                    position = slot * dp * batch + rank * batch + local
                    window[position] = sample
        assert None not in window
        observed.extend(window)
    return observed


@pytest.mark.parametrize(
    "old_dp,old_batch,old_accum,new_dp,new_batch,new_accum",
    [(4, 2, 2, 8, 1, 2), (8, 1, 2, 4, 2, 2)],
)
def test_global_sampler_is_stable_across_dp_and_batch_changes(
    old_dp, old_batch, old_accum, new_dp, new_batch, new_accum
):
    length = 35
    global_batch = old_dp * old_batch * old_accum
    assert global_batch == new_dp * new_batch * new_accum
    original = _global_samples(length, old_dp, old_batch, old_accum)
    resumed = _global_samples(length, new_dp, new_batch, new_accum, offset=global_batch)
    assert original[global_batch:] == resumed
    assert sorted(original) == list(range(length))
    assert len(set(original)) == length


@pytest.mark.parametrize("policy,expected", [("drop", []), ("flush", list(range(3)))])
def test_epoch_tail_smaller_than_dp_uses_only_real_samples(policy, expected):
    observed = _global_samples(3, 8, 1, 2, policy=policy)
    assert sorted(observed) == expected


def _contract_config(tmp_path, dp, batch, accum, mode="ddp"):
    return create_train_config(
        _TokenDataset(),
        str(tmp_path),
        device="cpu",
        dp_size=dp,
        dp_mode=mode,
        backend="gloo",
        master_port=find_free_port(),
        n_epoch=1,
        batch_per_device=batch,
        grad_accum_steps=accum,
        epoch_tail_policy="flush",
        ckpt_interval=100,
    )


def _take_contract(config, checkpoint):
    state = PreloadedState()
    loader = ResumeLoader(
        config,
        "unused",
        True,
        ParallelTopology(config.dp_size),
        lambda _: None,
    )
    loader._take_cursor(state, checkpoint, {"optimizer": False, "scheduler": False})
    return state


def test_elastic_contract_and_rejections(tmp_path):
    old = _contract_config(tmp_path, 4, 2, 2)
    new = _contract_config(tmp_path, 8, 1, 2)
    meta = {
        "resume_contract_version": 1,
        "optimizer_step": 1,
        "dataset_len": 35,
        "epoch_sample_offset": 16,
        "random_seed": old.random_seed,
        "cp_size": old.cp_size,
        "tp_size": old.tp_size,
        "rollout_mode": old.rollout_mode,
        "strategy": old.strategy,
        "rl_update_epochs": old.rl_update_epochs,
        "rl_minibatch_prompts": old.rl_minibatch_prompts,
        "dp_size": 4,
        "dp_mode": "ddp",
        "batch_per_device": 2,
        "grad_accum_steps": 2,
        "epoch_tail_policy": "flush",
    }
    checkpoint = Checkpoint(epoch=0, consumed_samples=16, meta=meta)
    assert _take_contract(new, checkpoint).epoch_sample_offset == 16

    bad = deepcopy(checkpoint)
    bad.meta["dp_mode"] = "fsdp"
    with pytest.raises(ValueError, match="FSDP"):
        _take_contract(new, bad)
    with pytest.raises(ValueError, match="global effective batch"):
        _take_contract(_contract_config(tmp_path, 8, 2, 2), checkpoint)
    bad = deepcopy(checkpoint)
    bad.meta["random_seed"] += 1
    with pytest.raises(ValueError, match="random_seed"):
        _take_contract(new, bad)
    legacy = Checkpoint(
        epoch=0,
        consumed_samples=16,
        meta={"dp_size": 4, "epoch_tail_policy": "flush"},
    )
    for config in (old, new):
        with pytest.raises(ValueError, match="migrate checkpoint metadata"):
            _take_contract(config, legacy)


def test_cpu_ddp_short_tail_and_single_rank_resume(tmp_path):
    """Gloo ranks including an empty tail rank finish the same collectives."""
    tail_dir = tmp_path / "tail"
    Trainer(_contract_config(tail_dir, 2, 1, 2)).train()
    tail = Checkpoint.load(str(tail_dir / "epoch_0_step_3"))
    assert tail.consumed_samples == 9
    assert tail.meta["epoch_sample_offset"] == 9

    baseline_dir = tmp_path / "baseline"
    Trainer(_contract_config(baseline_dir, 1, 2, 2, mode="none")).train()
    baseline = Checkpoint.load(str(baseline_dir / "epoch_0_step_3"))

    old = _contract_config(tmp_path, 2, 1, 2)
    Trainer(old, callbacks=[_StopAfterOneStep()]).train()
    first = tmp_path / "epoch_0_step_1"
    saved = Checkpoint.load(str(first))
    assert saved.meta["epoch_sample_offset"] == 4
    assert saved.consumed_samples == 4

    new = _contract_config(tmp_path, 1, 2, 2, mode="none")
    Trainer(new).train(param_path=str(first), resume=True)
    final = Checkpoint.load(str(tmp_path / "epoch_0_step_3"))
    assert final.meta["epoch_sample_offset"] == 9
    assert final.consumed_samples == 9
    assert final.meta["optimizer_step"] == 3
    for key in baseline.state_dict:
        torch.testing.assert_close(
            baseline.state_dict[key], final.state_dict[key], rtol=1e-5, atol=1e-6
        )
    for key in tail.state_dict:
        torch.testing.assert_close(
            baseline.state_dict[key], tail.state_dict[key], rtol=1e-5, atol=1e-6
        )

    reverse_dir = tmp_path / "reverse"
    Trainer(
        _contract_config(reverse_dir, 1, 2, 2, mode="none"),
        callbacks=[_StopAfterOneStep()],
    ).train()
    reverse_first = reverse_dir / "epoch_0_step_1"
    Trainer(_contract_config(reverse_dir, 2, 1, 2)).train(
        param_path=str(reverse_first), resume=True
    )
    reverse = Checkpoint.load(str(reverse_dir / "epoch_0_step_3"))
    assert reverse.consumed_samples == 9
    for key in baseline.state_dict:
        torch.testing.assert_close(
            baseline.state_dict[key], reverse.state_dict[key], rtol=1e-5, atol=1e-6
        )

    wrong_length = create_train_config(
        _TokenDataset(length=10),
        str(tmp_path / "wrong_length"),
        device="cpu",
        dp_size=1,
        dp_mode="none",
        backend="gloo",
        n_epoch=1,
        batch_per_device=2,
        grad_accum_steps=2,
        epoch_tail_policy="flush",
    )
    with pytest.raises(ValueError, match="dataset length changed"):
        Trainer(wrong_length).train(param_path=str(first), resume=True)


def test_sft_global_token_weighting_matches_single_rank(tmp_path):
    def config(directory, dp, batch, mode):
        return create_train_config(
            _MaskedTokenDataset(length=5),
            str(directory),
            device="cpu",
            strategy="sft",
            dp_size=dp,
            dp_mode=mode,
            backend="gloo",
            master_port=find_free_port(),
            n_epoch=1,
            batch_per_device=batch,
            grad_accum_steps=2,
            epoch_tail_policy="flush",
            ckpt_interval=100,
        )

    single = tmp_path / "single_sft"
    distributed = tmp_path / "distributed_sft"
    Trainer(config(single, 1, 2, "none")).train()
    Trainer(config(distributed, 2, 1, "ddp")).train()
    expected = Checkpoint.load(str(single / "epoch_0_step_2"))
    actual = Checkpoint.load(str(distributed / "epoch_0_step_2"))
    for key in expected.state_dict:
        torch.testing.assert_close(
            expected.state_dict[key], actual.state_dict[key], rtol=1e-5, atol=1e-6
        )
