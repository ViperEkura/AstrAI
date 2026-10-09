import os

import pytest
import torch

from astrai.dataset.dataset import (
    DatasetFactory,
    _build_jsonl_transform,
)
from astrai.dataset.storage import (
    JsonlStore,
    MmapStore,
    StoreFactory,
    detect_format,
)
from astrai.serialization import (
    load_bin,
    save_bin,
)

SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}:{{ message['content'] }}\n{% endfor %}"
)


from tests.support.dataset import (
    _make_seq_dataset,
    _rand_seq,
    _save_test_tokenizer,
    _write_dpo_jsonl,
    _write_text_dataset,
)


def test_unloaded_sample_window_raises():
    """Store.sample_window before load raises RuntimeError."""
    store = MmapStore(window_size=64, stride=64)
    with pytest.raises(IndexError, match="Data too short"):
        store.sample_window(0)


def test_store_unloaded_len():
    """Unloaded Store has __len__ == 0."""
    store = MmapStore()
    assert len(store) == 0
    assert store.keys == []


def test_store_fetch_begin_equals_end(base_test_env):
    test_dir = base_test_env["test_dir"]
    dataset = _make_seq_dataset(test_dir, "empty_fetch", seq_length=100, window_size=32)
    result = dataset.store.fetch(10, 10, "sequence")
    assert result.numel() == 0


def test_store_fetch_before_load():
    """Store.fetch before load raises RuntimeError"""
    store = MmapStore()
    with pytest.raises(RuntimeError, match="not loaded"):
        store.fetch(0, 10, "sequence")


def test_detect_format_nonexistent_path():
    """detect_format raises FileNotFoundError for bad path"""
    with pytest.raises(FileNotFoundError, match="No supported"):
        detect_format("/nonexistent/path/xyz")


def test_detect_format_unsupported_file(base_test_env):
    """detect_format raises ValueError for unsupported file extension"""
    test_dir = base_test_env["test_dir"]
    path = os.path.join(test_dir, "data.txt")
    with open(path, "w") as f:
        f.write("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        detect_format(path)


def test_create_store_invalid_type():
    """StoreFactory.create raises ValueError for unknown type"""
    with pytest.raises(ValueError, match="Unknown component"):
        StoreFactory.create("parquet")


def test_store_multi_segment_concat(base_test_env):
    """Multi-segment data is concatenated into single tensor at load time"""
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
    """save_bin + load_bin roundtrip preserves data"""
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
    data = {"sequence": [_rand_seq(200)]}
    save_bin(test_dir, data)

    store = StoreFactory.create("bin")
    store.load(test_dir)
    assert store.token_count == 200
    assert store.num_records == 0
    assert len(store) == 0  # no window configured, no records → 0 samples
    assert "sequence" in store.keys

    result = store.fetch(10, 20, "sequence")
    assert result.tolist() == data["sequence"][0][10:20].tolist()


def test_mmap_dataset_load(base_test_env):
    test_dir = base_test_env["test_dir"]
    data = {"sequence": [_rand_seq(200)]}
    save_bin(test_dir, data)
    dataset = DatasetFactory.load("seq", test_dir, window_size=64)
    assert len(dataset) > 0
    assert dataset.token_count == 200
    assert dataset[0]["input_ids"].shape[0] == 64


def test_detect_format_bin_dir(base_test_env):
    """detect_format returns 'bin' for directory with .bin + meta.json"""
    test_dir = base_test_env["test_dir"]
    save_bin(test_dir, {"sequence": [torch.randint(0, 100, (10,))]})
    assert detect_format(test_dir) == "bin"


def test_store_fetch_multi_key(base_test_env):
    test_dir = base_test_env["test_dir"]
    save_bin(
        test_dir,
        {
            "sequence": [torch.randint(0, 100, (100,), dtype=torch.int64)],
            "loss_mask": [torch.ones(100, dtype=torch.int64)],
        },
    )
    store = StoreFactory.create("bin")
    store.load(test_dir)
    result = store.fetch(10, 20, ["sequence", "loss_mask"])
    assert isinstance(result, dict)
    assert result["sequence"].shape[0] == 10
    assert result["loss_mask"].shape[0] == 10


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


def test_dataset_load_explicit_storage_type(base_test_env):
    test_dir = base_test_env["test_dir"]
    dataset = _make_seq_dataset(test_dir, "explicit", storage_type="bin")
    assert len(dataset) > 0
    assert dataset.token_count == 200


@pytest.mark.parametrize("use_jsonl", [True, False])
def test_detect_format_data_dir(base_test_env, use_jsonl):
    """detect_format returns 'jsonl' for dirs of .jsonl or .json files."""
    test_dir = base_test_env["test_dir"]
    tokenizer_path = _save_test_tokenizer(test_dir, base_test_env["tokenizer"])
    data_dir = _write_text_dataset(
        test_dir,
        "jsonl_data" if use_jsonl else "json_data",
        tokenizer_path,
        [{"text": "hello world"}, {"text": "foo bar baz"}],
    )
    assert detect_format(data_dir) == "jsonl"


def test_jsonl_store_lazy_len_returns_record_count(base_test_env):
    """JsonlStore in lazy mode: len() returns record count, not tokens."""
    test_dir = base_test_env["test_dir"]
    records = [{"input": str(i), "chosen": "c", "rejected": "r"} for i in range(5)]
    path = _write_dpo_jsonl(test_dir, records)

    store = JsonlStore()
    store.load(path, processor=lambda r: {"chosen": torch.tensor([1, 2])})

    assert len(store) == 5
    assert store.num_records == 5


def test_jsonl_store_eager_len_returns_token_count(base_test_env):
    """JsonlStore in eager mode: num_records reflects per-record count."""
    test_dir = base_test_env["test_dir"]
    tokenizer_path = _save_test_tokenizer(test_dir, base_test_env["tokenizer"])
    data_dir = _write_text_dataset(
        test_dir,
        "jsonl_data",
        tokenizer_path,
        [{"text": "hello world"}, {"text": "foo bar"}],
        config_overrides={
            "output": {"position_ids_mode": "none"},
        },
    )

    store = JsonlStore()
    store.load(data_dir, transform=_build_jsonl_transform(data_dir))

    assert store.num_records == 2
    assert len(store.keys) > 0


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
        "chosen": [_rand_seq(seq_length), _rand_seq(seq_length)],
        "rejected": [_rand_seq(seq_length), _rand_seq(seq_length)],
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


def test_mmap_store_stream_only_no_offsets(base_test_env):
    """MmapStore without offsets: num_records == 0, stream works.

    No window configured → ``len(store)`` is 0 (no iterate units).
    ``token_count`` remains 128 for raw token slicing, and ``fetch``
    provides direct token-range access.
    """
    test_dir = base_test_env["test_dir"]

    seq_length = 128
    dummy_data = {"sequence": [_rand_seq(seq_length)]}
    save_bin(test_dir, dummy_data)

    store = StoreFactory.create("bin")
    store.load(test_dir)

    assert store.token_count == seq_length
    assert store.num_records == 0
    assert len(store) == 0

    chunk = store.fetch(0, 32, "sequence")
    assert chunk.shape == (32,)
