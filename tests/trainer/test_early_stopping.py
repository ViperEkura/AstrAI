import os

import numpy as np
import torch

from astrai.config.train_config import TrainConfig
from astrai.trainer.schedule import SchedulerFactory
from astrai.trainer.trainer import Trainer


def test_early_stopping_simulation(base_test_env, early_stopping_dataset):
    """Simulate early stopping behavior"""

    def optimizer_fn(model):
        return torch.optim.AdamW(model.parameters())

    def scheduler_fn(optim):
        return SchedulerFactory.create(
            "cosine", optim, warmup_steps=10, lr_decay_steps=10, min_rate=0.05
        )

    train_config = TrainConfig(
        strategy="seq",
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        model_fn=lambda: base_test_env["model"],
        dataset=early_stopping_dataset,
        ckpt_dir=base_test_env["test_dir"],
        log_dir=os.path.join(base_test_env["test_dir"], "logs"),
        n_epoch=2,
        batch_per_device=2,
        ckpt_interval=1,
        grad_accum_steps=2,
        random_seed=np.random.randint(1e4),
        device_type=base_test_env["device"],
    )

    trainer = Trainer(train_config)

    # Should handle early stopping gracefully
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
    import json

    with open(os.path.join(load_dir, "meta.json")) as f:
        meta = json.load(f)
    assert meta["consumed_samples"] == 20
