from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from astrai.config import TrainConfig
from astrai.factory import BaseFactory
from astrai.trainer.session import TrainSession
from astrai.trainer.strategy import BaseStrategy, StrategyFactory
from astrai.trainer.train_context import TrainContextBuilder


class NoCopyDataset(Dataset):
    def __deepcopy__(self, memo):
        raise AssertionError("Dataset must not be copied into checkpoint metadata")

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return index


def _config(**overrides):
    values = dict(
        model_fn=lambda: torch.nn.Linear(1, 1),
        strategy="seq",
        dataset=NoCopyDataset(),
        optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler_fn=lambda optimizer: None,
        device_type="cpu",
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_checkpoint_snapshot_does_not_copy_runtime_objects():
    snapshot = _config().to_dict()
    assert snapshot["strategy"] == "seq"
    assert snapshot["batch_per_device"] == 4
    assert snapshot["collate_fn"] is None
    assert snapshot["val_dataset"] is None
    assert "dataset" not in snapshot
    assert "model_fn" not in snapshot
    assert "optimizer_fn" not in snapshot


def test_registered_custom_strategy_and_unknown_options():
    @StrategyFactory.register("custom_refactor_test")
    class CustomStrategy(BaseStrategy):
        def __init__(self, model, device, custom_scale=1.0, **kwargs):
            super().__init__(model, device, **kwargs)
            self.custom_scale = custom_scale

    config = _config(strategy="custom_refactor_test")
    assert config.strategy == "custom_refactor_test"
    model = torch.nn.Linear(1, 1)
    strategy = StrategyFactory.create_checked(
        "custom_refactor_test", model=model, device="cpu", custom_scale=2.0
    )
    assert strategy.custom_scale == 2.0
    with pytest.raises(TypeError, match="unknown arguments"):
        StrategyFactory.create_checked(
            "custom_refactor_test", model=model, device="cpu", custm_scale=2.0
        )


def test_legacy_factory_warns_for_dropped_kwargs_and_checked_path_raises():
    class Component:
        def __init__(self, value):
            self.value = value

    class Factory(BaseFactory[Component]):
        pass

    Factory.register("component")(Component)
    with pytest.warns(DeprecationWarning, match="ignored arguments"):
        assert Factory.create("component", value=1, typo=2).value == 1
    with pytest.raises(TypeError):
        Factory.create_checked("component", value=1, typo=2)


def test_train_session_cleans_up_when_start_callback_fails(monkeypatch):
    import astrai.trainer.session as session_module

    events = []
    monkeypatch.setattr(
        session_module, "register_signal_handlers", lambda _: events.append("register")
    )
    monkeypatch.setattr(
        session_module,
        "unregister_signal_handlers",
        lambda: events.append("unregister"),
    )
    rollout = SimpleNamespace(close=lambda: events.append("close"))
    context = SimpleNamespace(
        async_rollout=rollout,
        executor=SimpleNamespace(use_distributed=False),
    )

    def callback(name, _):
        events.append(name)
        if name == "on_train_begin":
            raise RuntimeError("startup failed")

    with pytest.raises(RuntimeError, match="startup failed"):
        with TrainSession(lambda: context, callback):
            pass
    assert events == [
        "register",
        "on_train_begin",
        "on_error",
        "close",
        "on_train_end",
        "unregister",
    ]


def test_builder_closes_rollout_after_configuration_failure(monkeypatch):
    import astrai.trainer.train_context as context_module

    closed = []
    rollout = SimpleNamespace(close=lambda: closed.append(True))
    context = SimpleNamespace(async_rollout=None)
    builder = TrainContextBuilder(_config())
    monkeypatch.setattr(context_module, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        context_module, "build_topology", lambda **_: SimpleNamespace(tp_size=1)
    )
    monkeypatch.setattr(
        builder, "_load_preloaded_state", lambda: SimpleNamespace(model_config={})
    )
    monkeypatch.setattr(builder, "_create_executor", lambda: object())
    monkeypatch.setattr(builder, "_validate_rollout_configuration", lambda _: None)
    monkeypatch.setattr(builder, "_create_context", lambda *_: context)
    monkeypatch.setattr(builder, "_prepare_model", lambda *_: None)
    monkeypatch.setattr(builder, "_restore_optimizer_state", lambda *_: None)
    monkeypatch.setattr(builder, "_get_datasets", lambda: ([], None))
    monkeypatch.setattr(builder, "_create_dataloaders", lambda *_: None)
    monkeypatch.setattr(builder, "_create_strategy", lambda *_: {})

    def fail_configure(ctx, _):
        ctx.async_rollout = rollout
        raise RuntimeError("rollout setup failed")

    monkeypatch.setattr(builder, "_configure_rollout", fail_configure)
    with pytest.raises(RuntimeError, match="rollout setup failed"):
        builder.build()
    assert closed == [True]
