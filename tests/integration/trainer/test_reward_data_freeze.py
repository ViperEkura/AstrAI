"""Dataset preparation keeps semantic duplicates out of all held-out splits."""

import pytest

from examples.rl_reward.prepare_data import freeze_splits


def test_countdown_permutations_deduplicate_before_frozen_split_selection():
    rows = [{"nums": [2, 3, i + 7], "target": i + 12} for i in range(68)]
    rows += [{"nums": [7, 3, 2], "target": 12}]
    splits, counts = freeze_splits(rows, [], "countdown", 64, 2, 2, 3407)
    assert counts["train_source_duplicates"] == 1
    assert [len(splits[name]) for name in ("train", "dev", "test")] == [64, 2, 2]
    ids = [record["id"] for split in splits.values() for record in split]
    assert len(ids) == len(set(ids)) == 68
    repeated, _ = freeze_splits(list(reversed(rows)), [], "countdown", 64, 2, 2, 3407)
    assert splits == repeated


def test_gsm8k_preserves_original_test_and_removes_train_collisions():
    train = [{"question": f"Question {i}?", "answer": f"#### {i}"} for i in range(66)]
    test = [{"question": f"Held out {i}?", "answer": f"#### {i}"} for i in range(2)]
    test += [{"question": " Question  0? ", "answer": "#### 0"}]
    splits, counts = freeze_splits(train, test, "gsm8k", 64, 2, 2, 3407)
    assert counts["test_source_duplicates_or_train_collisions"] == 1
    assert all(record["question"].startswith("Held out") for record in splits["test"])
    assert not {r["id"] for r in splits["test"]}.intersection(
        r["id"] for r in splits["train"]
    )


def test_frozen_split_never_silently_shrinks_when_deduplication_reduces_supply():
    rows = [{"nums": [2, 3, 7], "target": 12}] * 68
    with pytest.raises(ValueError, match="insufficient"):
        freeze_splits(rows, [], "countdown", 64, 2, 2, 3407)


def test_conflicting_labels_for_identical_questions_fail_preparation():
    rows = [
        {"question": "same question", "answer": "#### 1"},
        {"question": " same  question ", "answer": "#### 2"},
    ]
    with pytest.raises(ValueError, match="conflicting labels"):
        freeze_splits(rows, [], "gsm8k", 64, 2, 2, 3407)
