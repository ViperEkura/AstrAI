import json
import os
import tempfile

import pytest
import torch

from astrai.dataset.dataset import (
    DatasetFactory,
    _build_jsonl_transform,
    grpo_collate_fn,
)
from astrai.dataset.storage import (
    StoreFactory,
)
from astrai.preprocessing.builder import SectionedMaskBuilder
from tests.support.data import make_grpo_config

SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}:{{ message['content'] }}\n{% endfor %}"
)


from tests.support.dataset import (
    _dump_jsonl,
    _save_test_tokenizer,
    _write_dpo_jsonl,
    _write_grpo_jsonl,
    _write_text_dataset,
)


@pytest.mark.parametrize("dirname", ["json_data", "jsonl_data"])
def test_json_store_seq(base_test_env, dirname):
    """JsonlStore loads a text JSONL dataset and feeds seq training."""
    test_dir = base_test_env["test_dir"]
    tokenizer_path = _save_test_tokenizer(test_dir, base_test_env["tokenizer"])
    data_dir = _write_text_dataset(
        test_dir,
        dirname,
        tokenizer_path,
        [{"text": "hello world"}, {"text": "foo bar baz qux"}],
    )

    store = StoreFactory.create("jsonl")
    store.load(data_dir, transform=_build_jsonl_transform(data_dir))
    assert len(store) > 0
    assert "sequence" in store.keys

    dataset = DatasetFactory.load("seq", data_dir, window_size=8)
    assert len(dataset) > 0
    item = dataset[0]
    assert "input_ids" in item
    assert "target_ids" in item
    assert item["input_ids"].dtype == torch.long


def test_json_store_no_tokenizer_path(base_test_env):
    """JsonlStore uses dataset dir as tokenizer_path when omitted."""
    test_dir = base_test_env["test_dir"]
    tokenizer = base_test_env["tokenizer"]
    tokenizer.set_chat_template(SIMPLE_CHAT_TEMPLATE)

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
    _dump_jsonl(os.path.join(data_dir, "data.jsonl"), records)

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
    store.load(data_dir, transform=_build_jsonl_transform(data_dir))
    assert len(store) > 0
    assert "sequence" in store.keys
    assert "loss_mask" in store.keys


def test_jsonl_store_sft(base_test_env):
    test_dir = base_test_env["test_dir"]
    tokenizer = base_test_env["tokenizer"]
    tokenizer.set_chat_template(SIMPLE_CHAT_TEMPLATE)
    tokenizer_path = _save_test_tokenizer(test_dir, tokenizer)
    data_dir = _write_text_dataset(
        test_dir,
        "sft_jsonl",
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
    store.load(data_dir, transform=_build_jsonl_transform(data_dir))
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


def test_sft_jsonl_default_messages_config(base_test_env):
    """SFT loads a chat-style JSONL dir with no dataset_config.json.

    Falls back to the built-in messages config: every role except
    ``assistant`` is masked, loss on assistant only.
    """
    test_dir = base_test_env["test_dir"]
    tokenizer = base_test_env["tokenizer"]
    tokenizer.set_chat_template(SIMPLE_CHAT_TEMPLATE)
    tokenizer_path = _save_test_tokenizer(test_dir, tokenizer)

    data_dir = os.path.join(test_dir, "jsonl_data")
    os.makedirs(data_dir, exist_ok=True)
    records = [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "bye"},
                {"role": "assistant", "content": "see you"},
            ]
        },
    ]
    _dump_jsonl(os.path.join(data_dir, "data.jsonl"), records)

    dataset = DatasetFactory.load(
        "sft", data_dir, window_size=8, tokenizer_path=tokenizer_path
    )
    assert "sequence" in dataset.keys
    assert "loss_mask" in dataset.keys
    assert "position_ids" in dataset.keys
    assert len(dataset) > 0
    item = dataset[0]
    assert "input_ids" in item
    assert "target_ids" in item
    assert "loss_mask" in item
    assert "position_ids" in item
    assert item["loss_mask"].dtype == torch.bool


def test_sft_jsonl_explicit_config_takes_priority(base_test_env):
    """When dataset_config.json exists, it overrides the default messages config."""
    test_dir = base_test_env["test_dir"]
    tokenizer = base_test_env["tokenizer"]
    tokenizer.set_chat_template(SIMPLE_CHAT_TEMPLATE)
    tokenizer_path = _save_test_tokenizer(test_dir, tokenizer)

    data_dir = _write_text_dataset(
        test_dir,
        "sft_explicit",
        tokenizer_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        ],
        config_overrides={
            "input": {
                "sections": [{"field": "messages", "action": "$role", "template": True}]
            },
            "mask": {"user": "mask", "assistant": "train"},
            "mask_default": "mask",
            "preprocessing": {"max_seq_len": 128},
            "output": {"position_ids_mode": "doc_reset"},
        },
    )
    dataset = DatasetFactory.load(
        "sft", data_dir, window_size=8, tokenizer_path=tokenizer_path
    )
    assert "sequence" in dataset.keys
    assert "loss_mask" in dataset.keys


def test_grpo_builder_preserves_response_boundaries(base_test_env):
    """MultiOutputMaskBuilder with list_field returns List[List[int]] for responses."""
    tokenizer = base_test_env["tokenizer"]
    _save_test_tokenizer(base_test_env["test_dir"], tokenizer)

    builder = SectionedMaskBuilder()
    config = make_grpo_config(template=False)
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
