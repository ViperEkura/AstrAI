from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

import astrai.trainer.context.builder as builder_module
import astrai.trainer.session as session_module
from astrai.config import ComponentSpec, TrainConfig
from astrai.factory import BaseFactory
from astrai.inference.core.versioning import PolicyVersionGuard
from astrai.trainer import components as dataset_component
from astrai.trainer.components import build_scheduler_component
from astrai.trainer.context import TrainContext, TrainContextBuilder
from astrai.trainer.context.metric import MetricHistory
from astrai.trainer.rollout import RolloutRunner
from astrai.trainer.rollout import worker as worker_module
from astrai.trainer.rollout.async_round import AsyncRoundCoordinator
from astrai.trainer.session import TrainSession
from astrai.trainer.strategy import BaseStrategy, StrategyFactory
from astrai.trainer.trainer import Trainer


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
            type="tests.unit.trainer.test_refactor_boundaries:NoCopyDataset", params={}
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
                type="tests.unit.trainer.test_refactor_boundaries:NoCopyDataset",
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
            type="tests.unit.trainer.test_refactor_boundaries:custom_dataset_factory",
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


def test_global_window_combines_losses_with_shared_normalizer():
    strategy = SimpleNamespace(
        loss_outputs=lambda batch: iter(
            {"loss": torch.tensor(loss), "metrics": {"task": loss}, "normalizer": 4}
            for loss in (2.0, 3.0)
        )
    )
    output = Trainer._window_loss_output(strategy, "batch", 4)
    assert output["loss"].item() == 5.0
    assert output["metrics"] == {"task": 5.0}
    assert output["normalizer"] == 4.0


@pytest.mark.parametrize("counts", [[], [2, 3]])
def test_global_window_rejects_empty_or_inconsistent_losses(counts):
    strategy = SimpleNamespace(
        loss_outputs=lambda batch: iter(
            {"loss": torch.tensor(1.0), "metrics": {}, "normalizer": count}
            for count in counts
        )
    )
    with pytest.raises(RuntimeError, match="has no loss|share a normalizer"):
        Trainer._window_loss_output(strategy, "batch", 2)


def test_registered_custom_strategy_and_unknown_options():
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


def test_train_session_cleans_up_when_start_callback_fails(monkeypatch):
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
        rollout=rollout,
        executor=SimpleNamespace(use_distributed=False),
        stop_requested=False,
        run_shutdown=lambda: events.append("shutdown_hooks"),
        metric_history=MetricHistory(),
    )

    def callback(name, _):
        events.append(name)
        if name == "on_train_begin":
            context.metric_history.stage({"type": "step"})
            raise RuntimeError("startup failed")
        if name == "on_shutdown":
            context.metric_history.commit(1, 2)
            assert not context.metric_history.snapshot().records

    with pytest.raises(RuntimeError, match="startup failed"):
        with TrainSession(lambda: context, callback):
            pass
    assert events == [
        "register",
        "on_train_begin",
        "shutdown_hooks",
        "on_shutdown",
        "close",
        "on_train_end",
        "unregister",
    ]


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


def test_rollout_worker_follows_its_parent_first(monkeypatch):
    """A replica sets its parent-death signal before touching any state."""

    class ParentDied(RuntimeError):
        pass

    monkeypatch.setattr(
        worker_module, "die_with_parent", lambda: (_ for _ in ()).throw(ParentDied())
    )
    with pytest.raises(ParentDied):
        worker_module.run_rollout_worker(None, None)


@pytest.mark.parametrize(
    ("mode", "failure"),
    [
        (mode, failure)
        for mode in ("offline", "sync", "async_round")
        for failure in (None, "optimizer", "publisher", "critic", "scheduler")
        if mode != "offline" or failure != "publisher"
    ],
)
def test_update_failure_keeps_partial_weights_unpublishable(mode, failure):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    events = []
    guard = PolicyVersionGuard(0, lambda: None, lambda: events.append("version"))

    def fail_at(phase):
        events.append(phase)
        if failure == phase:
            raise RuntimeError(phase)

    actor_step = optimizer.step

    def update():
        actor_step()
        fail_at("optimizer")

    optimizer.step = update

    class Publisher:
        def publish(self, version, source):
            assert source is model and version == 1
            assert model.weight.item() == pytest.approx(0.9)
            if mode == "sync":
                assert guard.lock._is_owned()
                assert guard.policy_version == 0
            fail_at("publisher")

    if mode == "async_round":
        coordinator = AsyncRoundCoordinator.__new__(AsyncRoundCoordinator)
        coordinator._policy_version = 0
        coordinator.weights = SimpleNamespace(
            mark_committed=lambda version: events.append("version")
        )
        rollout = RolloutRunner(coordinator=coordinator, publishers=[Publisher()])
    elif mode == "sync":
        rollout = RolloutRunner(
            generator=guard, reward_model=object(), publishers=[Publisher()]
        )
    else:
        rollout = None

    def after_update():
        assert not guard.lock._is_owned()
        if rollout is not None:
            assert rollout.policy_version == 1
        fail_at("critic")

    context = TrainContext(
        model=model,
        optimizer=optimizer,
        rollout=rollout,
        pending_samples=2,
        strategy=SimpleNamespace(after_update=after_update),
        scheduler=SimpleNamespace(step=lambda: fail_at("scheduler")),
    )
    trainer = Trainer.__new__(Trainer)
    trainer.callbacks = [
        SimpleNamespace(
            before_optimizer_step=lambda ctx: ctx.metric_history.stage({"type": "step"})
        )
    ]
    batch = SimpleNamespace(prompts=torch.ones(2, 1))

    def commit():
        if mode == "async_round":
            trainer._commit_online_update(context, batch, new_data=True)
        else:
            trainer._commit_global_update(context, new_data=True)

    if failure:
        with pytest.raises(RuntimeError, match=failure):
            commit()
    else:
        commit()
        assert events == (
            ["optimizer", "critic", "scheduler"]
            if mode == "offline"
            else ["optimizer", "publisher", "version", "critic", "scheduler"]
        )

    # Mutation is real even when the optimizer or publisher raises. Never
    # mistake a rolled-back cursor for rolled-back weights.
    assert model.weight.item() == pytest.approx(0.9)
    advanced = failure not in ("optimizer", "publisher")
    if rollout is not None:
        assert rollout.policy_version == int(advanced)
    committed = failure is None or (mode == "async_round" and advanced)
    assert context.optimizer_step == int(committed)
    assert context.consumed_samples == 2 * int(committed)
    assert context.epoch_sample_offset == (
        2 * int(committed) if mode == "async_round" else 0
    )
    assert context.checkpoint_safe == (mode == "async_round" and failure is None)
    assert len(context.metric_history.snapshot().records) == int(failure is None)
    context.metric_history.commit(99, 99)
    assert len(context.metric_history.snapshot().records) == int(failure is None)
