import os

from astrai.trainer.trainer import Trainer
from tests.support.checkpoint import load_checkpoint_meta


def test_early_stopping_simulation(
    base_test_env, early_stopping_dataset, train_config_factory, device
):
    """Simulate early stopping behavior"""
    train_config = train_config_factory(
        model_fn=lambda: base_test_env["model"],
        dataset=early_stopping_dataset,
        test_dir=base_test_env["test_dir"],
        device=device,
        n_epoch=2,
        ckpt_interval=1,
        grad_accum_steps=2,
    )

    trainer = Trainer(train_config)

    try:
        trainer.train()
    except Exception:
        pass

    # Resume from latest checkpoint
    load_dir = os.path.join(base_test_env["test_dir"], "epoch_0_step_1")
    trainer = Trainer(train_config)
    trainer.train(param_path=load_dir, resume=True)

    # Verify checkpoint was saved at expected step
    load_dir = os.path.join(base_test_env["test_dir"], "epoch_1_step_5")
    meta = load_checkpoint_meta(load_dir)
    assert meta["consumed_samples"] == 20
