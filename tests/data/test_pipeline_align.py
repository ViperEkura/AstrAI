"""Tests for preprocessing pipeline bucket alignment."""

from astrai.preprocessing.pipeline import Pipeline


def test_align_bucket_backfills_missing_mask_with_ones():
    bucket = {
        "sequence": [[1, 2], [3, 4]],
        "loss_mask": [[0, 1]],
        "chosen_mask": [[1]],
        "position_ids": [[0, 1]],
    }
    result = {"sequence": [5, 6, 7]}
    Pipeline._align_bucket(bucket, result, [5, 6, 7])
    assert bucket["loss_mask"][-1] == [1, 1, 1]
    assert bucket["chosen_mask"][-1] == [1, 1, 1]
    assert bucket["position_ids"][-1] == [0, 0, 0]
    assert bucket["sequence"] == [[1, 2], [3, 4]]


def test_align_bucket_keeps_present_keys():
    bucket = {"sequence": [[1, 2]], "loss_mask": [[0, 1]]}
    result = {"sequence": [9], "loss_mask": [1]}
    Pipeline._align_bucket(bucket, result, [9])
    assert bucket["loss_mask"] == [[0, 1]]
    assert bucket["sequence"] == [[1, 2]]
