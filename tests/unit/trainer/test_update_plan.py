"""Replay planning preserves prompt groups and the final data commit boundary."""

import pytest
import torch

from astrai.trainer.rollout.batching import slice_batch
from astrai.trainer.step_plan import global_epoch_steps, learner_update_slices


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
