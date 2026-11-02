import torch

from astrai.dataset.dataset import (
    dpo_tokenize,
)
from astrai.dataset.storage import (
    MmapStore,
)

SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}:{{ message['content'] }}\n{% endfor %}"
)


def test_normalize_empty_key():
    """_normalize with empty tensor list does not crash."""
    store = MmapStore()
    store._normalize({"sequence": []})
    assert len(store) == 0
    assert store.num_records == 0  # empty key forces num_records=0
    assert store.keys == ["sequence"]


def test_normalize_mixed_empty_key():
    """_normalize with empty + non-empty keys returns min=0 records."""
    store = MmapStore()
    store._normalize({"sequence": [torch.tensor([1, 2, 3])], "loss_mask": []})
    assert len(store) == 0
    assert store.num_records == 0
    assert store.token_count == 0  # min() over keys
    assert set(store.keys) == {"sequence", "loss_mask"}


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
