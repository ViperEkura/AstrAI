from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

import astrai.trainer.context.builder as builder_module
from astrai.config import ComponentSpec, TrainConfig
from astrai.factory import BaseFactory
from astrai.trainer import Trainer
from astrai.trainer import components as dataset_component
from astrai.trainer.components import build_scheduler_component
from astrai.trainer.context import TrainContext, TrainContextBuilder
from astrai.trainer.strategy import BaseStrategy, StrategyFactory
from tests.support.checkpoint import load_checkpoint_meta
from tests.support.rollout import RandomTokenDataset


class NoCopyDataset(Dataset):
    def __init__(self, length=1):
        self.length = length

    def __deepcopy__(self, memo):
        raise AssertionError("Dataset must not be copied into checkpoint metadata")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index


def custom_dataset_factory():
    return NoCopyDataset(length=3), None


def _config(**overrides):
    values = dict(
        model=ComponentSpec(
            type="torch.nn:Linear", params={"in_features": 1, "out_features": 1}
        ),
        strategy="seq",
        dataset=ComponentSpec(
            type="tests.trainer.test_trainer:NoCopyDataset", params={}
        ),
        optimizer=ComponentSpec(type="torch.optim:SGD", params={"lr": 0.1}),
        scheduler=ComponentSpec(
            type="torch.optim.lr_scheduler:StepLR",
            params={"step_size": 1, "gamma": 1.0},
        ),
        device_type="cpu",
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_checkpoint_snapshot_does_not_copy_runtime_objects():
    class RuntimeFactory:
        def __call__(self):
            raise AssertionError("Snapshot must not invoke runtime factories")

        def __deepcopy__(self, memo):
            raise AssertionError("Snapshot must not copy runtime factories")

    snapshot = _config(
        reward_model_fn=RuntimeFactory(),
        critic_model_fn=RuntimeFactory(),
        critic_optimizer_fn=RuntimeFactory(),
    ).to_dict()
    assert snapshot["strategy"] == "seq"
    assert snapshot["batch_per_device"] == 4
    assert snapshot["dataset"]["type"].endswith(":NoCopyDataset")
    assert snapshot["dataset"]["params"] == {}
    assert "model_fn" not in snapshot
    assert "optimizer_fn" not in snapshot
    assert "reward_model_fn" not in snapshot
    assert "critic_model_fn" not in snapshot
    assert "critic_optimizer_fn" not in snapshot


def test_component_specs_resolve_without_trainer_branches():
    config = _config(
        model=ComponentSpec(
            type="torch.nn:Linear", params={"in_features": 2, "out_features": 2}
        ),
        optimizer=ComponentSpec(type="torch.optim:SGD", params={"lr": 0.1}),
        scheduler=ComponentSpec(
            type="torch.optim.lr_scheduler:StepLR",
            params={"step_size": "$total_steps", "gamma": 0.5},
        ),
    )
    builder = TrainContextBuilder(config)
    model = builder._instantiate_model({})
    optimizer = builder._instantiate_optimizer(model)
    scheduler = builder._instantiate_scheduler(optimizer, dataset_len=9)

    assert isinstance(model, torch.nn.Linear)
    assert optimizer.param_groups[0]["params"][0] is model.weight
    assert scheduler.step_size == 2
    assert config.to_dict()["scheduler"]["params"]["step_size"] == "$total_steps"
    assert isinstance(config.dataset, ComponentSpec)
    assert TrainContext(config=config).dataset is None


def test_scheduler_counts_only_unpadded_global_windows():
    config = _config(
        dp_size=2,
        batch_per_device=2,
        grad_accum_steps=2,
        epoch_tail_policy="drop",
    )
    optimizer = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.1)
    spec = ComponentSpec(
        type="torch.optim.lr_scheduler:StepLR",
        params={"step_size": "$total_steps", "gamma": 0.5},
    )
    scheduler = build_scheduler_component(spec, optimizer, 7, config)
    assert scheduler.step_size == 0


def test_builder_creates_and_splits_context_dataset():
    created = []
    original_load = dataset_component.load_dataset_component

    def track_load(kind, path, params):
        dataset = original_load(kind, path, params)
        created.append(dataset)
        return dataset

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(dataset_component, "load_dataset_component", track_load)
    try:
        config = _config(
            dataset=ComponentSpec(
                type="tests.trainer.test_trainer:NoCopyDataset",
                params={"length": 10},
            ),
            val_split=0.2,
        )
        context = TrainContext(config=config)
        TrainContextBuilder(config)._create_datasets(context)
    finally:
        monkeypatch.undo()

    assert created == [context.dataset.dataset]
    assert context.dataset_len == 8
    assert len(context.val_dataset) == 2
    assert context.val_dataset.dataset is created[0]


def test_builder_resolves_custom_dataset():
    config = _config(
        dataset=ComponentSpec(
            type="tests.trainer.test_trainer:custom_dataset_factory",
        )
    )
    context = TrainContext(config=config)
    TrainContextBuilder(config)._create_datasets(context)
    assert context.dataset_len == 3
    assert context.val_dataset is None
    assert isinstance(context.dataset, NoCopyDataset)


def test_scheduler_uses_dataset_owned_by_context(monkeypatch):
    seen = []
    config = _config()
    context = TrainContext(dataset=NoCopyDataset(), config=config)
    builder = TrainContextBuilder(config)
    monkeypatch.setattr(builder_module, "get_current_device", lambda: "cpu")
    monkeypatch.setattr(
        builder,
        "_instantiate_scheduler",
        lambda optimizer, length: seen.append(length),
    )

    class Executor:
        def prepare(self, model_fn, optimizer_fn, scheduler_fn, **kwargs):
            return None, None, scheduler_fn(object())

    builder._prepare_model(context, Executor(), SimpleNamespace(state_dict=None))
    assert seen == [context.dataset_len]


def test_registered_custom_strategy_and_unknown_options(monkeypatch):
    monkeypatch.setattr(StrategyFactory, "_entries", dict(StrategyFactory._entries))
    monkeypatch.setattr(
        StrategyFactory, "_capabilities", dict(StrategyFactory._capabilities)
    )
    monkeypatch.setattr(
        "astrai.config.train_config._STRATEGY_CAPABILITIES",
        StrategyFactory._capabilities,
    )

    @StrategyFactory.register("custom_refactor_test")
    class CustomStrategy(BaseStrategy):
        def __init__(self, model, device, custom_scale=1.0, **kwargs):
            super().__init__(model, device, **kwargs)
            self.custom_scale = custom_scale

    config = _config(strategy="custom_refactor_test")
    assert config.strategy == "custom_refactor_test"
    model = torch.nn.Linear(1, 1)
    strategy = StrategyFactory.create(
        "custom_refactor_test", model=model, device="cpu", custom_scale=2.0
    )
    assert strategy.custom_scale == 2.0
    with pytest.raises(TypeError, match="unknown arguments"):
        StrategyFactory.create(
            "custom_refactor_test", model=model, device="cpu", custm_scale=2.0
        )


def test_factory_preserves_arguments_and_rejects_unknown_options():
    class Component:
        def __init__(self, value):
            self.value = value

    class Factory(BaseFactory[Component]):
        pass

    Factory.register("component")(Component)
    assert Factory.create("component", value=1).value == 1
    with pytest.raises(TypeError):
        Factory.create("component", value=1, typo=2)
    config = {"value": 1, "other_component_option": 2}
    assert Factory.create_from_config("component", config).value == 1
    assert Factory.create_from_config("component", config, value=3).value == 3
    with pytest.raises(TypeError):
        Factory.create_from_config("component", config, typo=2)
    assert config == {"value": 1, "other_component_option": 2}


def test_builder_closes_rollout_after_configuration_failure(monkeypatch):
    closed = []
    rollout = SimpleNamespace(close=lambda: closed.append(True))
    context = SimpleNamespace(rollout=None, dataset=[], val_dataset=None)
    builder = TrainContextBuilder(_config())
    monkeypatch.setattr(builder_module, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        builder_module, "build_topology", lambda **_: SimpleNamespace(tp_size=1)
    )
    monkeypatch.setattr(
        builder, "_load_preloaded_state", lambda: SimpleNamespace(model_config={})
    )
    monkeypatch.setattr(builder, "_create_executor", lambda: object())
    monkeypatch.setattr(builder, "_validate_rollout_configuration", lambda _: None)
    monkeypatch.setattr(builder, "_create_context", lambda *_: context)
    monkeypatch.setattr(builder, "_prepare_model", lambda *_: None)
    monkeypatch.setattr(builder, "_restore_optimizer_state", lambda *_: None)
    monkeypatch.setattr(builder, "_create_datasets", lambda *_: None)
    monkeypatch.setattr(builder, "_create_dataloaders", lambda *_: None)
    monkeypatch.setattr(builder, "_create_strategy", lambda *_: {})

    def fail_configure(ctx, _):
        ctx.rollout = rollout
        raise RuntimeError("rollout setup failed")

    monkeypatch.setattr(builder, "_configure_rollout", fail_configure)
    with pytest.raises(RuntimeError, match="rollout setup failed"):
        builder.build()
    assert closed == [True]


@pytest.mark.parametrize(
    "batch_per_device,grad_accum_steps", [(1, 1), (2, 1), (4, 1), (2, 4)]
)
def test_training_commits_small_epoch(
    base_test_env, train_config_factory, batch_per_device, grad_accum_steps
):
    config = train_config_factory(
        dataset=RandomTokenDataset(length=8),
        test_dir=base_test_env["test_dir"],
        device=base_test_env["device"],
        batch_per_device=batch_per_device,
        grad_accum_steps=grad_accum_steps,
    )
    Trainer(config).train()
    meta = load_checkpoint_meta(base_test_env["test_dir"])
    assert meta["consumed_samples"] == 8
    assert meta["optimizer_step"] == 8 // (batch_per_device * grad_accum_steps)
