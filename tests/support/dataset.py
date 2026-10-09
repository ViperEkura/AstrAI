import json
import os

import torch

from astrai.dataset.dataset import (
    DatasetFactory,
)
from astrai.serialization import (
    save_bin,
)

SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}:{{ message['content'] }}\n{% endfor %}"
)


def _rand_seq(length, vocab=1000):
    return torch.randint(0, vocab, (length,), dtype=torch.int64)


def _dump_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_test_tokenizer(test_dir, tokenizer):
    tokenizer_path = os.path.join(test_dir, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    return tokenizer_path


def _write_text_dataset(
    test_dir, dirname, tokenizer_path, records, config_overrides=None
):
    """Write a JSONL dataset directory with a text-section default config."""
    data_dir = os.path.join(test_dir, dirname)
    os.makedirs(data_dir, exist_ok=True)
    _dump_jsonl(os.path.join(data_dir, "data.jsonl"), records)

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


def _fake_fetch_record(self, idx, keys):
    """FakeStore.fetch_record matching real Store semantics."""
    if isinstance(keys, str):
        return self._data[keys][idx]
    return {k: self._data[k][idx] for k in keys}


def _grpo_fake_store(prompts, responses, masks, rewards):
    """Fake GRPO record store matching real Store semantics."""
    return type(
        "FakeStore",
        (),
        {
            "keys": ["prompts", "responses", "masks", "rewards"],
            "num_records": len(prompts),
            "token_count": 0,
            "_data": {
                "prompts": prompts,
                "responses": responses,
                "masks": masks,
                "rewards": rewards,
            },
            "fetch_record": _fake_fetch_record,
            "__len__": lambda self: self.num_records,
        },
    )()


def _make_seq_dataset(
    test_dir, name="data", seq_length=200, train_type="seq", data=None, **load_kwargs
):
    if data is None:
        data = {"sequence": [_rand_seq(seq_length)]}
    save_bin(test_dir, data)
    return DatasetFactory.load(
        train_type,
        test_dir,
        window_size=load_kwargs.pop("window_size", 64),
        **load_kwargs,
    )


def _write_grpo_jsonl(test_dir, tokenizer_path, records):
    """Write a GRPO JSONL dataset directory with config."""
    data_dir = os.path.join(test_dir, "grpo_jsonl")
    os.makedirs(data_dir, exist_ok=True)
    _dump_jsonl(os.path.join(data_dir, "data.jsonl"), records)

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


def _write_dpo_jsonl(test_dir, records):
    """Write a raw DPO JSONL file (no dataset_config.json)."""
    path = os.path.join(test_dir, "dpo.jsonl")
    _dump_jsonl(path, records)
    return path
