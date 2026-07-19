import json
import os
import tempfile

import numpy as np
import pytest
import torch

from astrai.config.preprocess_config import PipelineConfig
from astrai.dataset.dataset import DatasetFactory, dpo_tokenize
from astrai.dataset.storage import (
    H5Store,
    JsonlStore,
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


def _fake_fetch_record(self, idx, keys):
    """FakeStore.fetch_record matching real Store semantics."""
    if isinstance(keys, str):
        return self._data[keys][idx]
    return {k: self._data[k][idx] for k in keys}


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
    assert dpo_dataset.store is not None
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
    assert sft_dataset.store is not None
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


def test_dataset_token_count_property(base_test_env):
    """dataset.token_count exposes the raw stream token length."""
    test_dir = base_test_env["test_dir"]
    dataset = _make_seq_dataset(test_dir, "count_test_data")
    assert dataset.token_count == 200
    assert dataset.token_count > len(dataset)
    assert len(dataset) == (200 - 1 - 64) // 64 + 1


def test_dataset_too_short_for_window(base_test_env):
    test_dir = base_test_env["test_dir"]
    dataset = _make_seq_dataset(test_dir, "short", seq_length=30)
    assert len(dataset) == 0
    assert dataset.token_count == 30


def test_unloaded_sample_window_raises():
    """Store.sample_window before load raises RuntimeError."""
    from astrai.dataset.storage import H5Store

    store = H5Store(window_size=64, stride=64)
    with pytest.raises(IndexError, match="Data too short"):
        store.sample_window(0)


def test_unloaded_dataset_len():
    """__len__ on a store with no data returns 0."""
    from astrai.dataset.storage import H5Store

    store = H5Store(window_size=64, stride=64)
    assert len(store) == 0


def test_store_unloaded_len():
    """Unloaded Store has __len__ == 0"""
    store = H5Store()
    assert len(store) == 0
    assert store.keys == []


def test_store_fetch_begin_equals_end(base_test_env):
    test_dir = base_test_env["test_dir"]
    dataset = _make_seq_dataset(test_dir, "empty_fetch", seq_length=100, window_size=32)
    result = dataset.store.fetch(10, 10, "sequence")
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


def test_normalize_empty_key():
    """_normalize with empty tensor list does not crash."""
    store = H5Store()
    store._normalize({"sequence": []})
    assert len(store) == 0
    assert store.num_records == 0  # empty key forces num_records=0
    assert store.keys == ["sequence"]


def test_normalize_mixed_empty_key():
    """_normalize with empty + non-empty keys returns min=0 records."""
    store = H5Store()
    store._normalize({"sequence": [torch.tensor([1, 2, 3])], "loss_mask": []})
    assert len(store) == 0
    assert store.num_records == 0
    assert store.token_count == 0  # min() over keys
    assert set(store.keys) == {"sequence", "loss_mask"}


def test_grpo_dataset_dtype(base_test_env):
    """GRPO dataset returns correct dtypes for per-record structured data."""
    from astrai.dataset.dataset import GRPODataset

    G = 4
    store = type(
        "FakeStore",
        (),
        {
            "keys": ["prompts", "responses", "masks", "rewards"],
            "num_records": 1,
            "token_count": 0,
            "_data": {
                "prompts": [torch.randint(0, 100, (10,), dtype=torch.int32)],
                "responses": [
                    [torch.randint(0, 100, (5,), dtype=torch.int32) for _ in range(G)]
                ],
                "masks": [[torch.ones(5, dtype=torch.int32) for _ in range(G)]],
                "rewards": [torch.rand(G, dtype=torch.float32)],
            },
            "fetch_record": _fake_fetch_record,
            "__len__": lambda self: self.num_records,
        },
    )()
    dataset = GRPODataset(store=store)
    item = dataset[0]

    assert item["prompts"].dtype == torch.long
    assert all(r.dtype == torch.long for r in item["responses"])
    assert all(m.dtype == torch.bool for m in item["masks"])
    assert item["rewards"].dtype == torch.float32


def test_grpo_dataset_load(base_test_env):
    """GRPO dataset loads record-structured data with per-response boundaries."""
    from astrai.dataset.dataset import GRPODataset

    G = 3
    prompt_len = 8
    resp_lens = [5, 7, 4]
    store = type(
        "FakeStore",
        (),
        {
            "keys": ["prompts", "responses", "masks", "rewards"],
            "num_records": 1,
            "token_count": 0,
            "_data": {
                "prompts": [torch.randint(0, 100, (prompt_len,))],
                "responses": [[torch.randint(0, 100, (rl,)) for rl in resp_lens]],
                "masks": [[torch.ones(rl, dtype=torch.int64) for rl in resp_lens]],
                "rewards": [torch.tensor([0.9, 0.3, 0.7], dtype=torch.float32)],
            },
            "fetch_record": _fake_fetch_record,
            "__len__": lambda self: self.num_records,
        },
    )()
    dataset = GRPODataset(store=store)

    assert len(dataset) == 1
    item = dataset[0]
    assert "prompts" in item
    assert "responses" in item
    assert "masks" in item
    assert "rewards" in item

    # Prompts is 1-D
    assert item["prompts"].shape == (prompt_len,)

    # Responses is a list of G tensors with correct lengths
    assert len(item["responses"]) == G
    for i, r in enumerate(item["responses"]):
        assert r.shape == (resp_lens[i],)

    # Masks align with responses
    assert len(item["masks"]) == G
    for i, m in enumerate(item["masks"]):
        assert m.shape == (resp_lens[i],)

    # Rewards has G elements
    assert item["rewards"].shape == (G,)


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
    assert dataset.token_count == 200


def _write_json_dataset(test_dir, tokenizer_path, records, config_overrides=None):
    """Write JSONL dataset — one JSON object per line."""
    data_dir = os.path.join(test_dir, "json_data")
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "data.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

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

    # Write .jsonl data
    records = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    ]
    with open(os.path.join(data_dir, "data.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

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


# ---------------------------------------------------------------------------
# GRPO end-to-end: builder → JsonlStore → GRPODataset → collate_fn
# ---------------------------------------------------------------------------


def _write_grpo_jsonl(test_dir, tokenizer_path, records):
    """Write a GRPO JSONL dataset directory with config."""
    data_dir = os.path.join(test_dir, "grpo_jsonl")
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "data.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    config = {
        "tokenizer_path": tokenizer_path,
        "version": 1,
        "input": {
            "sources": {
                "prompts": {
                    "sections": [
                        {
                            "field": "prompt",
                            "action": "mask",
                            "add_special_tokens": True,
                        }
                    ]
                },
                "responses": {
                    "sections": [{"field": "responses", "action": "train"}],
                    "list_field": True,
                    "mask_key": "masks",
                },
                "rewards": {
                    "sections": [{"field": "rewards", "action": "value"}],
                },
            }
        },
        "mask": {"user": "mask", "assistant": "train"},
        "mask_default": "mask",
        "preprocessing": {"max_seq_len": 128},
        "output": {"position_ids_mode": "none"},
    }

    with open(
        os.path.join(data_dir, "dataset_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return data_dir


def test_grpo_builder_preserves_response_boundaries(base_test_env):
    """MultiOutputMaskBuilder with list_field returns List[List[int]] for responses."""
    from astrai.preprocessing.builder import SectionedMaskBuilder
    from tests.data.conftest import make_grpo_no_template_config

    tokenizer = base_test_env["tokenizer"]
    tokenizer_path = _save_test_tokenizer(base_test_env["test_dir"], tokenizer)

    builder = SectionedMaskBuilder()
    config = make_grpo_no_template_config()
    config.preprocessing.max_seq_len = 128

    item = {
        "prompt": "What is 2+2?",
        "responses": ["4", "four", "2+2=4"],
        "rewards": [0.9, 0.1, 0.5],
    }

    result = builder.build(item, config, tokenizer)
    assert result is not None

    # prompts should be flat list of ints
    assert isinstance(result["prompts"], list)
    assert isinstance(result["prompts"][0], int)

    # responses should be list of lists (one per response)
    assert isinstance(result["responses"], list)
    assert isinstance(result["responses"][0], list)
    assert isinstance(result["responses"][0][0], int)
    assert len(result["responses"]) == 3

    # masks should match responses structure
    assert isinstance(result["masks"], list)
    assert len(result["masks"]) == 3
    for i in range(3):
        assert len(result["masks"][i]) == len(result["responses"][i])

    # rewards should be flat list of floats
    assert isinstance(result["rewards"], list)
    assert all(isinstance(r, float) for r in result["rewards"])
    assert len(result["rewards"]) == 3


def test_grpo_end_to_end_jsonl(base_test_env):
    """Full GRPO pipeline: JSONL → JsonlStore → GRPODataset → collate_fn."""
    from astrai.dataset.dataset import grpo_collate_fn

    test_dir = base_test_env["test_dir"]
    tokenizer = base_test_env["tokenizer"]
    tokenizer_path = _save_test_tokenizer(test_dir, tokenizer)

    records = [
        {
            "prompt": "What is 2+2?",
            "responses": ["4", "four", "The answer is 4"],
            "rewards": [0.9, 0.1, 0.5],
        },
        {
            "prompt": "Write a haiku",
            "responses": ["Leaves fall", "Cherry blossoms bloom in spring"],
            "rewards": [0.3, 0.8],
        },
    ]

    data_dir = _write_grpo_jsonl(test_dir, tokenizer_path, records)

    dataset = DatasetFactory.load("grpo", data_dir, window_size=0)
    assert len(dataset) == 2

    # Item 0: 3 responses
    item0 = dataset[0]
    assert item0["prompts"].ndim == 1
    assert len(item0["responses"]) == 3
    assert len(item0["masks"]) == 3
    assert item0["rewards"].shape == (3,)
    for r, m in zip(item0["responses"], item0["masks"]):
        assert r.shape == m.shape

    # Item 1: 2 responses (different group size)
    item1 = dataset[1]
    assert len(item1["responses"]) == 2
    assert item1["rewards"].shape == (2,)

    # Collate: batch records with same G (item0 has G=3)
    batch = grpo_collate_fn([item0, item0])
    assert batch["prompts"].shape[0] == 2
    assert batch["responses"].ndim == 3
    assert batch["responses"].shape[0] == 2
    assert batch["responses"].shape[1] == 3  # G=3
    assert batch["masks"].shape == batch["responses"].shape
    assert batch["rewards"].shape == (2, 3)


def test_grpo_collate_variable_lengths():
    """collate_fn pads variable-length responses to [B, G, R_max]."""
    from astrai.dataset.dataset import grpo_collate_fn

    batch = [
        {
            "prompts": torch.tensor([1, 2, 3]),
            "responses": [torch.tensor([4, 5]), torch.tensor([6, 7, 8, 9])],
            "masks": [torch.tensor([1, 1]), torch.tensor([1, 1, 1, 1])],
            "rewards": torch.tensor([0.9, 0.1]),
        },
        {
            "prompts": torch.tensor([10, 11]),
            "responses": [torch.tensor([12]), torch.tensor([13, 14, 15])],
            "masks": [torch.tensor([1]), torch.tensor([1, 1, 1])],
            "rewards": torch.tensor([0.5, 0.5]),
        },
    ]

    result = grpo_collate_fn(batch)

    assert result["prompts"].shape == (2, 3)  # B=2, P_max=3
    assert result["responses"].shape == (2, 2, 4)  # B=2, G=2, R_max=4
    assert result["masks"].shape == (2, 2, 4)
    assert result["rewards"].shape == (2, 2)

    # Check padding: item 1 prompt is length 2, padded to 3
    assert result["prompts"][1, 2] == 0

    # Check response content: item 0, response 0 is [4,5] padded to 4
    assert result["responses"][0, 0, 0] == 4
    assert result["responses"][0, 0, 1] == 5
    assert result["responses"][0, 0, 2] == 0  # padded
    assert not result["masks"][0, 0, 2]  # padded

    # Check response content: item 0, response 1 is [6,7,8,9] no padding
    assert result["responses"][0, 1, 3] == 9
    assert result["masks"][0, 1, 3]


def test_grpo_multiple_records(base_test_env):
    """GRPODataset loads multiple records with correct structure."""
    from astrai.dataset.dataset import GRPODataset

    G = 4
    n_records = 5

    dummy_responses = [
        [torch.randint(0, 100, (np.random.randint(3, 8),)) for _ in range(G)]
        for _ in range(n_records)
    ]
    store = type(
        "FakeStore",
        (),
        {
            "keys": ["prompts", "responses", "masks", "rewards"],
            "num_records": n_records,
            "token_count": 0,
            "_data": {
                "prompts": [torch.randint(0, 100, (10,)) for _ in range(n_records)],
                "responses": dummy_responses,
                "masks": [
                    [torch.ones(r.shape[0], dtype=torch.int64) for r in resps]
                    for resps in dummy_responses
                ],
                "rewards": [
                    torch.rand(G, dtype=torch.float32) for _ in range(n_records)
                ],
            },
            "fetch_record": _fake_fetch_record,
            "__len__": lambda self: self.num_records,
        },
    )()
    dataset = GRPODataset(store=store)

    assert len(dataset) == n_records

    for i in range(n_records):
        item = dataset[i]
        assert len(item["responses"]) == G
        assert len(item["masks"]) == G
        assert item["rewards"].shape == (G,)
        for g in range(G):
            assert item["responses"][g].shape == item["masks"][g].shape


def _write_dpo_jsonl(test_dir, records):
    """Write a raw DPO JSONL file (no dataset_config.json)."""
    path = os.path.join(test_dir, "dpo.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def test_dpo_tokenize_pure_function():
    """dpo_tokenize returns flat lists with correct mask alignment."""

    class FakeTokenizer:
        def apply_chat_template(
            self, messages, tokenize=True, add_generation_prompt=True
        ):
            ids = []
            for m in messages:
                ids.append(len(m["content"]))
                ids.append(-1)
            if add_generation_prompt:
                ids.append(99)
            return ids

    record = {"prompt": "ab", "chosen": "xyz", "rejected": "w"}
    result = dpo_tokenize(record, FakeTokenizer(), max_len=64)

    assert set(result.keys()) == {"chosen", "rejected", "chosen_mask", "rejected_mask"}
    assert len(result["chosen"]) == len(result["chosen_mask"])
    assert len(result["rejected"]) == len(result["rejected_mask"])

    assert result["chosen_mask"][0] == 0
    assert any(m == 1 for m in result["chosen_mask"])
    assert result["rejected_mask"][0] == 0


def test_dpo_tokenize_malformed_record():
    """dpo_tokenize returns None for missing fields."""

    class FakeTokenizer:
        def apply_chat_template(
            self, messages, tokenize=True, add_generation_prompt=True
        ):
            return [1]

    assert dpo_tokenize({}, FakeTokenizer()) is None
    assert dpo_tokenize({"prompt": "a"}, FakeTokenizer()) is None
    assert dpo_tokenize({"prompt": "a", "chosen": "b"}, FakeTokenizer()) is None


def test_dpo_jsonl_lazy_load(base_test_env):
    """DPODataset loads raw JSONL with tokenizer_path → lazy processor."""
    test_dir = base_test_env["test_dir"]
    tokenizer_path = _save_test_tokenizer(test_dir, base_test_env["tokenizer"])

    records = [
        {"input": "Hello", "chosen": "world", "rejected": "earth"},
        {"input": "Foo", "chosen": "bar", "rejected": "baz"},
    ]
    path = _write_dpo_jsonl(test_dir, records)

    ds = DatasetFactory.load(
        train_type="dpo",
        load_path=path,
        window_size=0,
        tokenizer_path=tokenizer_path,
    )

    assert len(ds) == 2
    assert ds.store.num_records == 2
    assert ds.store._processor is not None

    item = ds[0]
    assert set(item.keys()) == {"chosen", "rejected", "chosen_mask", "rejected_mask"}
    assert item["chosen"].dtype == torch.long
    assert item["chosen_mask"].dtype == torch.bool
    assert item["chosen"].shape == item["chosen_mask"].shape
    assert item["chosen"].shape == item["rejected"].shape


def test_dpo_jsonl_lazy_no_tokenizer():
    """DPODataset on jsonl without tokenizer_path falls back to eager
    (which requires dataset_config.json, so it should raise)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dpo.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"input": "a", "chosen": "b", "rejected": "c"}) + "\n")

        with pytest.raises(FileNotFoundError, match="dataset_config.json"):
            DatasetFactory.load(
                train_type="dpo",
                load_path=path,
                window_size=0,
            )


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
    data_dir = _write_jsonl_dataset(
        test_dir,
        tokenizer_path,
        [{"text": "hello world"}, {"text": "foo bar"}],
        config_overrides={
            "preprocessing": {"max_seq_len": 128, "min_chars": 0},
            "output": {"position_ids_mode": "none"},
        },
    )

    store = JsonlStore()
    store.load(data_dir)

    assert store.num_records == 2
    assert len(store.keys) > 0


def test_h5_store_dual_mode(base_test_env):
    """H5Store supports both fetch (stream) and fetch_record (record).

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
    save_h5(test_dir, "dpo_data", dummy_data)

    store = H5Store()
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
    stream_view = H5Store(window_size=seq_length, stride=seq_length)
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
