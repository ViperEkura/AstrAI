import pytest

from astrai.trainer import Trainer


def test_training_runs_with_various_batch_sizes(
    base_test_env, random_dataset, train_config_factory
):
    """Training should complete for a range of batch sizes without error."""
    for batch_per_device in [1, 2, 4]:
        train_config = train_config_factory(
            model_fn=lambda: base_test_env["model"],
            dataset=random_dataset,
            test_dir=base_test_env["test_dir"],
            device=base_test_env["device"],
            batch_per_device=batch_per_device,
        )
        trainer = Trainer(train_config)
        trainer.train()


@pytest.mark.slow
def test_gradient_accumulation_runs(
    base_test_env, random_dataset, train_config_factory
):
    """Training with gradient accumulation should complete."""
    train_config = train_config_factory(
        model_fn=lambda: base_test_env["model"],
        dataset=random_dataset,
        test_dir=base_test_env["test_dir"],
        device=base_test_env["device"],
        batch_per_device=2,
        grad_accum_steps=4,
    )
    trainer = Trainer(train_config)
    trainer.train()
