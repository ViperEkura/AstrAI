import json
import os

import numpy as np
import pytest
import torch

from astrai.config.preprocess_config import PipelineConfig
from astrai.dataset.dataset import DatasetFactory, SEQDataset
from astrai.dataset.storage import (
    H5Store,
    StoreFactory,
    detect_format,
)
from astrai.serialization import (
    load_bin,
    save_bin,
    save_h5,
)


def _rand_seq(length, vocab=1000):
    return torch.randint(0, vocab, (length,), dtype=torch.int64)


def _save_test_tokenizer(test_dir, tokenizer):
    tokenizer_path = os.path.join(test_dir, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    return tokenizer_path


def _write_jsonl_dataset(test_dir, tokenizer_path, records, config_overrides=None):
    data_dir = os.path.join(test_dir, "jsonl_data")
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "data.jsonl"), "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    config = {
        "tokenizer_path": tokenizer_path,
        "version": 1,
        "input": {"sections": [{"field": "text", "action": "train"}]},
        "preprocessing": {"max_seq_len": 128},
        "output": {"position_ids_mode": "continuous"},
    }
    if config_overrides:
        config.update(config_overrides)

    with open(
        os.path.join(data_dir, "dataset_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return data_dir


def _make_seq_dataset(
    test_dir, name="data", seq_length=200, train_type="seq", data=None, **load_kwargs
):
    if data is None:
        data = {"sequence": [_rand_seq(seq_length)]}
    save_h5(test_dir, name, data)
    return DatasetFactory.load(
        train_type,
        test_dir,
        window_size=load_kwargs.pop("window_size", 64),
        **load_kwargs,
    )


def test_dataset_loader_random_paths(base_test_env):
    """Test dataset loader with multiple random paths"""
    test_dir = base_test_env["test_dir"]

    num_files = np.random.randint(2, 5)
    for i in range(num_files):
        seq_length = np.random.randint(200, 400)
        dummy_data = {"sequence": [_rand_seq(seq_length) for _ in range(10)]}
        loaded_dataset = _make_seq_dataset(
            test_dir, f"data_{i}", seq_length, data=dummy_data
        )
        assert loaded_dataset is not None
        assert len(loaded_dataset) > 0

    # Test that we can get items without errors
    for i in range(len(loaded_dataset)):
        item = loaded_dataset[i]
        assert "input_ids" in item
        assert "target_ids" in item
        assert item["input_ids"].shape == item["target_ids"].shape
        assert item["input_ids"].shape[0] == 64


def test_dpo_strategy_with_random_data(base_test_env):
    """Test DPO strategy with randomized preference data"""
    test_dir = base_test_env["test_dir"]

    seq_length = np.random.randint(100, 200)
    dummy_data = {
        "chosen": [_rand_seq(seq_length)],
        "rejected": [_rand_seq(seq_length)],
        "chosen_mask": [torch.ones(seq_length, dtype=torch.bool)],
        "rejected_mask": [torch.ones(seq_length, dtype=torch.bool)],
    }
    dpo_dataset = _make_seq_dataset(
        test_dir, "dpo_data", seq_length, train_type="dpo", data=dummy_data
    )

    assert dpo_dataset is not None
    assert dpo_dataset.storage is not None
    assert len(dpo_dataset) > 0

    # Test that we can get DPO items without errors
    for i in range(min(3, len(dpo_dataset))):
        item = dpo_dataset[i]
        assert "chosen" in item
        assert "rejected" in item
        assert "chosen_mask" in item
        assert "rejected_mask" in item
        assert item["chosen"].shape == item["rejected"].shape
        assert item["chosen_mask"].shape == item["rejected_mask"].shape


def test_sft_dataset_with_random_data(base_test_env):
    """Test SFT dataset with random data"""
    test_dir = base_test_env["test_dir"]

    seq_length = np.random.randint(100, 200)
    dummy_data = {
        "sequence": [_rand_seq(seq_length)],
        "loss_mask": [torch.ones(seq_length, dtype=torch.bool)],
        "position_ids": [torch.arange(seq_length, dtype=torch.int32)],
    }
    sft_dataset = _make_seq_dataset(
        test_dir, "sft_data", seq_length, train_type="sft", data=dummy_data
    )

    assert sft_dataset is not None
    assert sft_dataset.storage is not None
    assert len(sft_dataset) > 0

    # Test that we can get SFT items without errors
    for i in range(min(3, len(sft_dataset))):
        item = sft_dataset[i]
        assert "input_ids" in item
        assert "target_ids" in item
        assert "loss_mask" in item
        assert item["input_ids"].shape == item["target_ids"].shape
        assert item["loss_mask"].shape[0] == 64


def test_dataset_with_custom_stride(base_test_env):
    """Test dataset with custom stride parameter"""
    test_dir = base_test_env["test_dir"]

    custom_stride = 32
    dataset = _make_seq_dataset(test_dir, "stride_test_data", stride=custom_stride)
    assert dataset is not None
    assert len(dataset) > 0

    default_stride_dataset = DatasetFactory.load(
        train_type="seq",
        load_path=test_dir,
        window_size=64,
    )

    assert len(dataset) > len(default_stride_dataset)


def test_dataset_count_property(base_test_env):
    test_dir = base_test_env["test_dir"]
    dataset = _make_seq_dataset(test_dir, "count_test_data")
    assert dataset.count == 200
    assert dataset.count > len(dataset)
    assert len(dataset) == (200 - 1 - 64) // 64 + 1


def test_empty_dataset_count():
    """Test count returns 0 when no data is loaded"""
    dataset = SEQDataset(window_size=64, stride=32)
    assert dataset.count == 0
    assert dataset.keys == []


def test_dataset_too_short_for_window(base_test_env):
    test_dir = base_test_env["test_dir"]
    dataset = _make_seq_dataset(test_dir, "short", seq_length=30)
    assert len(dataset) == 0
    assert dataset.count == 30


def test_unloaded_dataset_getitem_raises():
    """__getitem__ without load() should fail clearly"""
    dataset = SEQDataset(window_size=64, stride=32)
    with pytest.raises(RuntimeError, match="not loaded"):
        dataset.get_index(0)


def test_unloaded_dataset_len():
    """__len__ without load() returns 0"""
    dataset = SEQDataset(window_size=64, stride=32)
    assert len(dataset) == 0


def test_store_unloaded_len():
    """Unloaded Store has __len__ == 0"""
    store = H5Store()
    assert len(store) == 0
    assert store.keys == []


def test_store_fetch_begin_equals_end(base_test_env):
    test_dir = base_test_env["test_dir"]
    dataset = _make_seq_dataset(test_dir, "empty_fetch", seq_length=100, window_size=32)
    result = dataset.storage.fetch(10, 10, "sequence")
    assert result.numel() == 0


def test_store_fetch_before_load():
    """Store.fetch before load raises RuntimeError"""
    store = H5Store()
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
    """Multi-segment H5 data is concatenated into single tensor at load time"""
    import os

    test_dir = base_test_env["test_dir"]
    data_dir = os.path.join(test_dir, "multi_seg")
    os.makedirs(data_dir, exist_ok=True)

    segs = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5, 6, 7]),
        torch.tensor([8, 9]),
    ]
    save_h5(data_dir, "data", {"sequence": segs})

    store = StoreFactory.create("h5")
    store.load(data_dir)
    assert len(store) == 9
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
    assert len(store) == 200
    assert "sequence" in store.keys

    result = store.fetch(10, 20, "sequence")
    assert result.tolist() == data["sequence"][0][10:20].tolist()


def test_mmap_dataset_load(base_test_env):
    test_dir = base_test_env["test_dir"]
    data = {"sequence": [_rand_seq(200)]}
    save_bin(test_dir, data)
    dataset = DatasetFactory.load("seq", test_dir, window_size=64)
    assert len(dataset) > 0
    assert dataset.count == 200
    assert dataset[0]["input_ids"].shape[0] == 64


def test_normalize_empty_key():
    """_normalize with empty tensor list does not crash"""
    store = H5Store()
    store._normalize({"sequence": []})
    assert len(store) == 0
    assert store.keys == ["sequence"]


def test_normalize_mixed_empty_key():
    """_normalize with empty + non-empty keys returns min=0"""
    store = H5Store()
    store._normalize({"sequence": [torch.tensor([1, 2, 3])], "loss_mask": []})
    assert len(store) == 0
    assert set(store.keys) == {"sequence", "loss_mask"}


def test_grpo_dataset_dtype(base_test_env):
    test_dir = base_test_env["test_dir"]
    dummy_data = {
        "prompts": [torch.randint(0, 100, (100,), dtype=torch.int32)],
        "responses": [torch.randint(0, 100, (100,), dtype=torch.int32)],
        "masks": [torch.ones(100, dtype=torch.int32)],
        "rewards": [torch.ones(100, dtype=torch.float32)],
    }
    dataset = _make_seq_dataset(
        test_dir, "grpo_dtype", train_type="grpo", data=dummy_data, window_size=32
    )
    item = dataset[0]

    assert item["prompts"].dtype == torch.long
    assert item["responses"].dtype == torch.long
    assert item["masks"].dtype == torch.bool
    assert item["rewards"].dtype == torch.float32


def test_grpo_dataset_load(base_test_env):
    test_dir = base_test_env["test_dir"]
    dummy_data = {
        "prompts": [_rand_seq(200)],
        "responses": [_rand_seq(200)],
        "masks": [torch.ones(200, dtype=torch.int64)],
        "rewards": [torch.rand(200, dtype=torch.float32)],
    }
    dataset = _make_seq_dataset(
        test_dir, "grpo_test", train_type="grpo", data=dummy_data
    )
    assert len(dataset) > 0
    item = dataset[0]
    assert "prompts" in item
    assert "responses" in item
    assert "masks" in item
    assert "rewards" in item
    assert item["prompts"].shape[0] == 64
    assert item["responses"].shape[0] == 64


def test_detect_format_bin_dir(base_test_env):
    """detect_format returns 'bin' for directory with .bin + meta.json"""
    test_dir = base_test_env["test_dir"]
    save_bin(test_dir, {"sequence": [torch.randint(0, 100, (10,))]})
    assert detect_format(test_dir) == "bin"


def test_store_fetch_multi_key(base_test_env):
    test_dir = base_test_env["test_dir"]
    save_h5(
        test_dir,
        "multi_key",
        {
            "sequence": [torch.randint(0, 100, (100,), dtype=torch.int64)],
            "loss_mask": [torch.ones(100, dtype=torch.int64)],
        },
    )
    store = StoreFactory.create("h5")
    store.load(test_dir)
    result = store.fetch(10, 20, ["sequence", "loss_mask"])
    assert isinstance(result, dict)
    assert result["sequence"].shape[0] == 10
    assert result["loss_mask"].shape[0] == 10


def test_store_fetch_out_of_bounds(base_test_env):
    test_dir = base_test_env["test_dir"]
    save_h5(test_dir, "bounds", {"sequence": [torch.randint(0, 100, (50,))]})
    store = StoreFactory.create("h5")
    store.load(test_dir)
    with pytest.raises(ValueError, match="out of bounds"):
        store.fetch(-1, 10, "sequence")
    with pytest.raises(ValueError, match="out of bounds"):
        store.fetch(0, 51, "sequence")
    with pytest.raises(ValueError, match="out of bounds"):
        store.fetch(50, 50, "sequence")


def test_dataset_load_explicit_storage_type(base_test_env):
    test_dir = base_test_env["test_dir"]
    dataset = _make_seq_dataset(test_dir, "explicit", storage_type="h5")
    assert len(dataset) > 0
    assert dataset.count == 200


def _write_json_dataset(test_dir, tokenizer_path, records, config_overrides=None):
    """Write JSON (not JSONL) dataset — array of objects."""
    data_dir = os.path.join(test_dir, "json_data")
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    config = {
        "tokenizer_path": tokenizer_path,
        "version": 1,
        "input": {"sections": [{"field": "text", "action": "train"}]},
        "preprocessing": {"max_seq_len": 128, "min_chars": 0},
        "output": {"position_ids_mode": "continuous"},
    }
    if config_overrides:
        config.update(config_overrides)

    with open(
        os.path.join(data_dir, "dataset_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return data_dir


def test_detect_format_jsonl_dir(base_test_env):
    test_dir = base_test_env["test_dir"]
    tokenizer_path = _save_test_tokenizer(test_dir, base_test_env["tokenizer"])
    data_dir = _write_jsonl_dataset(
        test_dir,
        tokenizer_path,
        [{"text": "hello world"}, {"text": "foo bar baz"}],
    )
    assert detect_format(data_dir) == "jsonl"


def test_detect_format_json_dir(base_test_env):
    """detect_format returns 'jsonl' for directory with .json files."""
    test_dir = base_test_env["test_dir"]
    tokenizer_path = _save_test_tokenizer(test_dir, base_test_env["tokenizer"])
    data_dir = _write_json_dataset(
        test_dir,
        tokenizer_path,
        [{"text": "hello world"}, {"text": "foo bar baz qux"}],
    )
    assert detect_format(data_dir) == "jsonl"


def test_json_store_seq(base_test_env):
    """JsonlStore loads .json array correctly."""
    test_dir = base_test_env["test_dir"]
    tokenizer_path = _save_test_tokenizer(test_dir, base_test_env["tokenizer"])
    data_dir = _write_json_dataset(
        test_dir,
        tokenizer_path,
        [{"text": "hello world"}, {"text": "foo bar baz qux"}],
    )

    store = StoreFactory.create("jsonl")
    store.load(data_dir)
    assert len(store) > 0
    assert "sequence" in store.keys

    dataset = DatasetFactory.load("seq", data_dir, window_size=8)
    assert len(dataset) > 0
    item = dataset[0]
    assert "input_ids" in item
    assert "target_ids" in item


def test_json_store_no_tokenizer_path(base_test_env):
    """JsonlStore uses dataset dir as tokenizer_path when omitted."""
    test_dir = base_test_env["test_dir"]
    tokenizer = base_test_env["tokenizer"]
    tokenizer.set_chat_template(
        "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}\n{% endfor %}"
    )

    data_dir = os.path.join(test_dir, "self_contained")
    os.makedirs(data_dir, exist_ok=True)

    # Save tokenizer files directly in the dataset directory
    tokenizer.save_pretrained(data_dir)

    # Write .json data
    records = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    ]
    with open(os.path.join(data_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    # dataset_config.json WITHOUT tokenizer_path
    config = {
        "version": 1,
        "input": {
            "sections": [{"field": "messages", "action": "$role", "template": True}]
        },
        "mask": {"user": "mask", "assistant": "train"},
        "mask_default": "mask",
        "preprocessing": {"max_seq_len": 128, "min_chars": 0},
        "output": {"position_ids_mode": "continuous"},
    }
    with open(
        os.path.join(data_dir, "dataset_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    store = StoreFactory.create("jsonl")
    store.load(data_dir)
    assert len(store) > 0
    assert "sequence" in store.keys
    assert "loss_mask" in store.keys


def test_jsonl_store_seq(base_test_env):
    test_dir = base_test_env["test_dir"]
    tokenizer_path = _save_test_tokenizer(test_dir, base_test_env["tokenizer"])
    data_dir = _write_jsonl_dataset(
        test_dir,
        tokenizer_path,
        [{"text": "hello world"}, {"text": "foo bar baz qux"}],
        config_overrides={"preprocessing": {"max_seq_len": 128, "min_chars": 0}},
    )

    store = StoreFactory.create("jsonl")
    store.load(data_dir)
    assert len(store) > 0
    assert "sequence" in store.keys

    dataset = DatasetFactory.load("seq", data_dir, window_size=8)
    assert len(dataset) > 0
    item = dataset[0]
    assert "input_ids" in item
    assert "target_ids" in item
    assert item["input_ids"].dtype == torch.long


def test_jsonl_store_sft(base_test_env):
    test_dir = base_test_env["test_dir"]
    tokenizer = base_test_env["tokenizer"]
    tokenizer.set_chat_template(
        "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}\n{% endfor %}"
    )
    tokenizer_path = _save_test_tokenizer(test_dir, tokenizer)
    data_dir = _write_jsonl_dataset(
        test_dir,
        tokenizer_path,
        [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ],
        config_overrides={
            "input": {
                "sections": [{"field": "messages", "action": "$role", "template": True}]
            },
            "mask": {"system": "mask", "user": "mask", "assistant": "train"},
            "mask_default": "mask",
        },
    )

    store = StoreFactory.create("jsonl")
    store.load(data_dir)
    assert "sequence" in store.keys
    assert "loss_mask" in store.keys
    assert "position_ids" in store.keys

    dataset = DatasetFactory.load("sft", data_dir, window_size=8)
    item = dataset[0]
    assert "input_ids" in item
    assert "target_ids" in item
    assert "loss_mask" in item
    assert "position_ids" in item
    assert item["loss_mask"].dtype == torch.bool


def test_jsonl_store_pipeline_config_roundtrip(base_test_env):
    test_dir = base_test_env["test_dir"]
    config_path = os.path.join(test_dir, "dataset_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_path": os.path.join(test_dir, "tokenizer"),
                "version": 1,
                "input": {"sections": [{"field": "text", "action": "train"}]},
                "mask": {"assistant": "train"},
                "preprocessing": {"max_seq_len": 64},
                "output": {"position_ids_mode": "doc_reset"},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw.pop("tokenizer_path")
    config = PipelineConfig.from_dict(raw)
    assert config.output.position_ids_mode == "doc_reset"
    assert config.preprocessing.max_seq_len == 64
