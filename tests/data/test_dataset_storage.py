import os

import pytest
import torch

from astrai.dataset.storage import (
    JsonlStore,
    MmapStore,
    StoreFactory,
)
from astrai.serialization import (
    load_bin,
    save_bin,
)


@pytest.mark.parametrize("single_file", [False, True])
def test_jsonl_records_preserve_order_and_defer_processing(
    tmp_path, caplog, single_file
):
    (tmp_path / "b.jsonl").write_text('{"value": 2}\n', encoding="utf-8")
    first = tmp_path / "a.jsonl"
    first.write_text('\n{"value": 1}\ninvalid JSON\n', encoding="utf-8")
    processed = []

    def processor(record):
        processed.append(record["value"])
        return {"value": torch.tensor([record["value"]])}

    store = JsonlStore()
    store.load(str(first if single_file else tmp_path), processor=processor)
    assert processed == []
    assert "Failed to parse JSON line" in caplog.text
    expected = [1] if single_file else [1, 2]
    assert len(store) == len(expected)
    first.write_text('{"value": 99}\n', encoding="utf-8")
    assert [
        store.fetch_record(i, "value").item() for i in range(len(store))
    ] == expected
    assert processed == expected
    assert store.keys == ["value"]


def test_store_multi_segment_concat(base_test_env):
    test_dir = base_test_env["test_dir"]
    data_dir = os.path.join(test_dir, "multi_seg")
    os.makedirs(data_dir, exist_ok=True)

    segs = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6, 7]),
        torch.tensor([8, 9]),
    ]
    save_bin(data_dir, {"sequence": segs})

    store = StoreFactory.create("bin")
    store.load(data_dir)
    assert store.token_count == 9
    result = store.fetch(2, 7, "sequence")
    assert result.tolist() == [3, 4, 5, 6, 7]


def test_save_load_bin_roundtrip(base_test_env):
    test_dir = base_test_env["test_dir"]

    data = {
        "sequence": [torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)],
        "loss_mask": [torch.tensor([0, 1, 1, 0, 1], dtype=torch.int64)],
    }
    save_bin(test_dir, data)
    result = load_bin(test_dir)

    assert "sequence" in result
    assert "loss_mask" in result
    assert result["sequence"][0].tolist() == [1, 2, 3, 4, 5]
    assert result["loss_mask"][0].tolist() == [0, 1, 1, 0, 1]


def test_mmap_store_load_and_fetch(base_test_env):
    test_dir = base_test_env["test_dir"]
    data = {"sequence": [torch.arange(200)]}
    save_bin(test_dir, data)

    store = StoreFactory.create("bin")
    store.load(test_dir)
    assert store.token_count == 200
    assert store.num_records == 0
    assert len(store) == 0  # no window configured, no records → 0 samples
    assert "sequence" in store.keys

    result = store.fetch(10, 20, "sequence")
    assert result.tolist() == data["sequence"][0][10:20].tolist()


def test_store_fetch_out_of_bounds(base_test_env):
    test_dir = base_test_env["test_dir"]
    save_bin(test_dir, {"sequence": [torch.randint(0, 100, (50,))]})
    store = StoreFactory.create("bin")
    store.load(test_dir)
    with pytest.raises(ValueError, match="out of bounds"):
        store.fetch(-1, 10, "sequence")
    with pytest.raises(ValueError, match="out of bounds"):
        store.fetch(0, 51, "sequence")
    with pytest.raises(ValueError, match="out of bounds"):
        store.fetch(50, 50, "sequence")


def test_mmap_store_dual_mode(base_test_env):
    """MmapStore supports both fetch (stream) and fetch_record (record).

    No window configured → ``len(store)`` reflects the record count
    (2).  ``token_count`` retains the legacy stream length (128), and
    token-stream access via :meth:`fetch` is still available for
    callers that want explicit begin/end control.
    """
    test_dir = base_test_env["test_dir"]

    seq_length = 64
    dummy_data = {
        "chosen": [torch.arange(seq_length), torch.arange(seq_length)],
        "rejected": [torch.arange(seq_length), torch.arange(seq_length)],
    }
    save_bin(test_dir, dummy_data, record_keys=["chosen", "rejected"])

    store = MmapStore()
    store.load(test_dir)

    assert store.token_count == seq_length * 2
    assert store.num_records == 2
    assert len(store) == 2  # no window configured → record count

    rec0 = store.fetch_record(0, "chosen")
    assert rec0.shape == (seq_length,)

    stream = store.fetch(0, 10, "chosen")
    assert stream.shape == (10,)

    # Window-configured view of the same data uses stream sample count:
    #   token_count=128, window_size=64 → num_samples = (128-1-64)//64 + 1 = 1
    stream_view = MmapStore(window_size=seq_length, stride=seq_length)
    stream_view.load(test_dir)
    assert len(stream_view) == 1
