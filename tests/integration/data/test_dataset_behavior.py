import numpy as np
import torch

from astrai.dataset.dataset import (
    DatasetFactory,
    GRPODataset,
    grpo_collate_fn,
)
from astrai.serialization import (
    save_bin,
)

SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}:{{ message['content'] }}\n{% endfor %}"
)


from tests.support.dataset import (
    _grpo_fake_store,
    _make_seq_dataset,
    _rand_seq,
)


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
    save_bin(
        test_dir,
        dummy_data,
        record_keys=["chosen", "rejected", "chosen_mask", "rejected_mask"],
    )
    dpo_dataset = DatasetFactory.load(
        train_type="dpo",
        load_path=test_dir,
        window_size=0,
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


def test_grpo_dataset_dtype(base_test_env):
    """GRPO dataset returns correct dtypes for per-record structured data."""
    G = 4
    store = _grpo_fake_store(
        prompts=[torch.randint(0, 100, (10,), dtype=torch.int32)],
        responses=[[torch.randint(0, 100, (5,), dtype=torch.int32) for _ in range(G)]],
        masks=[[torch.ones(5, dtype=torch.int32) for _ in range(G)]],
        rewards=[torch.rand(G, dtype=torch.float32)],
    )
    dataset = GRPODataset(store=store)
    item = dataset[0]

    assert item["prompts"].dtype == torch.long
    assert all(r.dtype == torch.long for r in item["responses"])
    assert all(m.dtype == torch.bool for m in item["masks"])
    assert item["rewards"].dtype == torch.float32


def test_grpo_dataset_load(base_test_env):
    """GRPO dataset loads record-structured data with per-response boundaries."""
    G = 3
    prompt_len = 8
    resp_lens = [5, 7, 4]
    store = _grpo_fake_store(
        prompts=[torch.randint(0, 100, (prompt_len,))],
        responses=[[torch.randint(0, 100, (rl,)) for rl in resp_lens]],
        masks=[[torch.ones(rl, dtype=torch.int64) for rl in resp_lens]],
        rewards=[torch.tensor([0.9, 0.3, 0.7], dtype=torch.float32)],
    )
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


def test_grpo_collate_variable_lengths():
    """collate_fn pads variable-length responses to [B, G, R_max]."""
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

    # Prompts are left-padded so each response follows its real prompt tokens.
    assert torch.equal(result["prompts"][1], torch.tensor([0, 10, 11]))
    assert torch.equal(result["prompt_mask"][1], torch.tensor([False, True, True]))

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
    G = 4
    n_records = 5

    dummy_responses = [
        [torch.randint(0, 100, (np.random.randint(3, 8),)) for _ in range(G)]
        for _ in range(n_records)
    ]
    store = _grpo_fake_store(
        prompts=[torch.randint(0, 100, (10,)) for _ in range(n_records)],
        responses=dummy_responses,
        masks=[
            [torch.ones(r.shape[0], dtype=torch.int64) for r in resps]
            for resps in dummy_responses
        ],
        rewards=[torch.rand(G, dtype=torch.float32) for _ in range(n_records)],
    )
    dataset = GRPODataset(store=store)

    assert len(dataset) == n_records

    for i in range(n_records):
        item = dataset[i]
        assert len(item["responses"]) == G
        assert len(item["masks"]) == G
        assert item["rewards"].shape == (G,)
        for g in range(G):
            assert item["responses"][g].shape == item["masks"][g].shape
