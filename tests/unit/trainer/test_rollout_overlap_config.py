from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from torch import nn

from astrai.config import ComponentSpec, TrainConfig
from astrai.trainer.rollout.setup import (
    ResolvedAsyncRolloutConfig,
    configure_rollout,
)
from astrai.trainer.rollout.types import SamplingParams
from tests.support.tokenizers import FakeTokenizer


def _config(**overrides):
    values = dict(
        strategy="online_grpo",
        model=ComponentSpec(
            type="torch.nn:Linear", params={"in_features": 2, "out_features": 2}
        ),
        dataset=ComponentSpec(type="tests.support.rollout:RandomTokenDataset"),
        optimizer=ComponentSpec(type="torch.optim:SGD", params={"lr": 0.01}),
        scheduler=ComponentSpec(
            type="torch.optim.lr_scheduler:StepLR", params={"step_size": 1}
        ),
        reward_model_fn=object,
        device_type="cpu",
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_rollout_overlap_default_and_disabled_config_roundtrip():
    assert _config().rollout_enable_overlap is True
    config = _config(rollout_enable_overlap=False)
    saved = config.to_dict()
    assert saved["rollout_enable_overlap"] is False
    restored = TrainConfig(**{**saved, "reward_model_fn": object})
    assert restored.rollout_enable_overlap is False


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("placement", ["colocated", "replica", "async"])
def test_rollout_overlap_option_reaches_every_backend(enabled, placement, monkeypatch):
    """An explicit opt-out reaches training, validation and async workers."""
    config = _config(
        rollout_enable_overlap=enabled,
        rollout_device="cpu" if placement == "replica" else None,
        rollout_val_device="cpu" if placement != "async" else None,
    )
    model = nn.Linear(2, 2)
    model.config = SimpleNamespace(max_position_embeddings=32)
    strategy = SimpleNamespace(
        group_size=2,
        supports_online=lambda: True,
        set_rollout_runner=Mock(),
        set_weight_publishers=Mock(),
    )
    context = SimpleNamespace(
        model=model,
        executor=SimpleNamespace(model_for_inference=lambda model: model),
        strategy=strategy,
        checkpoint=None,
        optimizer_step=3,
        consumed_samples=0,
    )
    engine = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("astrai.trainer.backend.EngineCore", engine)
    coordinator = Mock()
    monkeypatch.setattr(
        "astrai.trainer.rollout.setup.AsyncRoundCoordinator", coordinator
    )
    if placement == "async":
        config.rollout_mode = "async_round"
        # Exercise wiring without querying GPUs or starting worker processes.
        monkeypatch.setattr(
            "astrai.trainer.rollout.setup.resolve_async_rollout",
            lambda *args, **kwargs: ResolvedAsyncRolloutConfig(
                learner_device="cpu",
                devices=("cpu",),
                params=SamplingParams(group_size=2),
                max_seq_len=32,
                max_prompts_per_worker=1,
                startup_timeout_s=30,
                generation_timeout_s=30,
                weight_timeout_s=30,
                max_policy_lag=1,
                random_seed=3407,
                sample_cursor=0,
            ),
        )
    configure_rollout(
        context,
        config,
        "tokenizer",
        {},
        Mock(return_value=model),
        lambda _: None,
        engine,
        SimpleNamespace(from_pretrained=lambda _: FakeTokenizer()),
    )
    if placement == "async":
        assert coordinator.call_args.kwargs["enable_overlap"] is enabled
        assert context.async_rollout.coordinator is coordinator.return_value
        engine.assert_not_called()
    else:
        runner = strategy.set_rollout_runner.call_args.args[0]
        assert runner.generator.backend.engine_core.enable_overlap is enabled
        assert (
            context.val_evaluator.generator.backend.engine_core.enable_overlap
            is enabled
        )
        assert engine.call_count == 2
        coordinator.assert_not_called()
