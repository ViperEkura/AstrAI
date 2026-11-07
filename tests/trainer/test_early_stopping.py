import os
from dataclasses import replace

import pytest

from astrai.trainer.trainer import Trainer
from tests.support.checkpoint import load_checkpoint_meta


def test_early_stopping_simulation(
    base_test_env, early_stopping_dataset, train_config_factory, cpu_device
):
    """Simulate early stopping behavior"""
    train_config = train_config_factory(
        dataset=early_stopping_dataset,
        test_dir=base_test_env["test_dir"],
        device=cpu_device,
        n_epoch=2,
        ckpt_interval=1,
        grad_accum_steps=2,
    )

    trainer = Trainer(train_config)

    with pytest.raises(RuntimeError, match="Simulated early stopping"):
        trainer.train()

    # Resume from latest checkpoint
    load_dir = os.path.join(base_test_env["test_dir"], "epoch_0_step_1")
    # A fresh dataset is opened on resume. Disable the injected
    # one-shot failure for the continuation.
    dataset = replace(
        train_config.dataset,
        params={**train_config.dataset.params, "stop_after": None},
    )
    trainer = Trainer(replace(train_config, dataset=dataset))
    trainer.train(param_path=load_dir, resume=True)

    # Verify checkpoint was saved at expected step
    load_dir = os.path.join(base_test_env["test_dir"], "epoch_2_step_4")
    meta = load_checkpoint_meta(load_dir)
    assert meta["consumed_samples"] == 20
