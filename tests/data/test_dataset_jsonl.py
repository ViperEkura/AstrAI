import json
import os
import tempfile

import pytest
import torch

from astrai.dataset.dataset import (
    DatasetFactory,
    grpo_collate_fn,
)
from tests.support.data import make_grpo_config


def test_grpo_end_to_end_jsonl(tmp_path, tokenizer_dir):

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

    data_dir = tmp_path / "grpo"
    data_dir.mkdir()
    config = make_grpo_config(template=False).to_dict()
    config["tokenizer_path"] = tokenizer_dir
    (data_dir / "dataset_config.json").write_text(json.dumps(config))
    (data_dir / "data.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    dataset = DatasetFactory.load("grpo", str(data_dir), window_size=0)
    assert len(dataset) == 2

    # Item 0: 3 responses
    item0 = dataset[0]
    assert item0["prompts"].dtype == torch.long
    assert all(response.dtype == torch.long for response in item0["responses"])
    assert all(mask.dtype == torch.bool for mask in item0["masks"])
    assert item0["rewards"].dtype == torch.float32
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


def test_dpo_jsonl_lazy_load(tmp_path, chat_tokenizer_dir):
    tokenizer_path = chat_tokenizer_dir

    records = [
        {"input": "Hello", "chosen": "world", "rejected": "earth"},
        {"input": "Foo", "chosen": "bar", "rejected": "baz"},
    ]
    path = tmp_path / "dpo.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    ds = DatasetFactory.load(
        train_type="dpo",
        load_path=str(path),
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
    assert item["rejected"].shape == item["rejected_mask"].shape


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
                load_path=str(path),
                window_size=0,
            )


@pytest.mark.parametrize("configuration", ["default", "explicit", "colocated"])
def test_sft_jsonl_config_priority_and_tokenizer_fallback(
    tmp_path, chat_tokenizer, chat_tokenizer_dir, configuration
):
    record = {
        "messages": [
            {"role": "user", "content": "Hello world."},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps(record) + "\n")
    tokenizer_path = chat_tokenizer_dir
    if configuration != "default":
        config = {
            "input": {
                "sections": [{"field": "messages", "action": "$role", "template": True}]
            },
            "mask_default": "train",
            "preprocessing": {"max_seq_len": 128, "min_chars": 0},
        }
        if configuration == "colocated":
            chat_tokenizer.save_pretrained(str(tmp_path))
            tokenizer_path = None
        else:
            config["tokenizer_path"] = tokenizer_path
            # The explicit dataset config must win over this unused argument.
            tokenizer_path = str(tmp_path / "unused-tokenizer")
        (tmp_path / "dataset_config.json").write_text(json.dumps(config))
    dataset = DatasetFactory.load(
        "sft", str(tmp_path), window_size=8, tokenizer_path=tokenizer_path
    )
    item = dataset[0]
    assert item["input_ids"].dtype == torch.long
    assert item["loss_mask"].dtype == torch.bool
    mask = dataset.store.fetch(0, dataset.token_count, "loss_mask")
    assert mask.any()
    assert not mask[0]  # BOS is never a supervised token.
    assert mask[1:].bool().all().item() == (configuration != "default")
