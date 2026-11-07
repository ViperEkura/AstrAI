"""Replay planning preserves prompt groups and the final data commit boundary."""

from types import SimpleNamespace

import pytest
import torch

from astrai.inference.core.versioning import PolicyVersionGuard
from astrai.trainer.context import TrainContext
from astrai.trainer.rollout import RolloutRunner
from astrai.trainer.rollout.async_round import AsyncRoundCoordinator
from astrai.trainer.rollout.batching import slice_batch
from astrai.trainer.step_plan import global_epoch_steps, learner_update_slices
from astrai.trainer.trainer import Trainer


@pytest.mark.parametrize("total", [1, 5, 8])
@pytest.mark.parametrize("epochs", [1, 3])
@pytest.mark.parametrize("size", [None, 2, 9])
def test_replay_plan_matches_step_count_and_visits_each_prompt(total, epochs, size):
    updates = list(learner_update_slices(total, epochs, size))
    assert len(updates) == global_epoch_steps(
        total,
        0,
        total,
        "flush",
        online=True,
        rl_update_epochs=epochs,
        rl_minibatch_prompts=size,
    )
    assert [
        i for selection, _ in updates for i in range(selection.start, selection.stop)
    ] == list(range(total)) * epochs
    assert [final for _, final in updates] == [False] * (len(updates) - 1) + [True]


def test_contiguous_batch_selection_preserves_views_and_metadata_contract():
    tensor = torch.arange(24).reshape(4, 2, 3)
    labels = ["a", "b", "c", "d"]
    batch = {"responses": tensor, "labels": labels, "constant": torch.tensor(7)}
    replay = slice_batch(batch, slice(1, 3), 4, slice_sequences=False)
    assert (
        replay["responses"].untyped_storage().data_ptr()
        == tensor.untyped_storage().data_ptr()
    )
    assert replay["labels"] is labels
    assert replay["constant"] is batch["constant"]
    selected = slice_batch(batch, slice(1, 3), 4)
    torch.testing.assert_close(selected["responses"], tensor[1:3])
    assert selected["labels"] == ["b", "c"]


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
