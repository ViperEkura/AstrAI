import json
import os
import shutil
import tempfile

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.utils.data import Dataset

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.transformer import AutoRegressiveLM
from astrai.tokenize import AutoTokenizer


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "unit: fast unit tests")


def create_test_tokenizer(vocab_size: int = 1000) -> AutoTokenizer:
    """Create a simple tokenizer for testing purposes."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=1, special_tokens=["<unk>", "<pad>"]
    )
    tokenizer.train_from_iterator([chr(i) for i in range(256)], trainer)
    auto_tokenizer = AutoTokenizer()
    auto_tokenizer._tokenizer = tokenizer
    auto_tokenizer._special_token_map = {"unk_token": "<unk>", "pad_token": "<pad>"}
    return auto_tokenizer


class RandomDataset(Dataset):
    """Random dataset for testing purposes."""

    def __init__(self, length=None, max_length=64, vocab_size=1000):
        self.length = length or int(torch.randint(100, 200, (1,)).item())
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
            "target_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
        }


class MultiTurnDataset(Dataset):
    """Multi-turn dataset with loss mask for SFT training tests."""

    def __init__(self, length=None, max_length=64, vocab_size=1000):
        self.length = length or int(torch.randint(100, 200, (1,)).item())
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.max_length,))
        target_ids = torch.randint(0, self.vocab_size, (self.max_length,))
        loss_mask = torch.randint(0, 1, (self.max_length,))

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
        }


class EarlyStoppingDataset(Dataset):
    """Dataset that triggers early stopping after consuming a specified number of samples."""

    def __init__(self, length=10, stop_after=5):
        self.length = length
        self.stop_after = stop_after
        self.count = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.count += 1
        if self.count == self.stop_after:
            raise RuntimeError("Simulated early stopping")

        return {
            "input_ids": torch.randint(0, 1000, (64,)),
            "target_ids": torch.randint(0, 1000, (64,)),
        }


@pytest.fixture(scope="session")
def test_tokenizer():
    """Session-scoped tokenizer, created once for the entire test run."""
    return create_test_tokenizer()


@pytest.fixture(scope="session")
def test_model():
    """Session-scoped small AutoRegressiveLM model, created once."""
    config = AutoRegressiveLMConfig(
        vocab_size=1000,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        max_position_embeddings=64,
        num_hidden_layers=2,
        rms_norm_eps=1e-5,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoRegressiveLM(config).to(device=device)

    return {
        "model": model,
        "device": device,
        "config": config,
    }


@pytest.fixture
def base_test_env(test_model, test_tokenizer):
    """Function-scoped test environment with isolated temp directory.

    Composes session-scoped model and tokenizer with a per-test temp dir.
    """
    test_dir = tempfile.mkdtemp()
    config_path = os.path.join(test_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "vocab_size": 1000,
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 16,
                "max_position_embeddings": 64,
                "num_hidden_layers": 2,
                "rms_norm_eps": 1e-5,
            },
            f,
        )

    yield {
        "device": test_model["device"],
        "test_dir": str(test_dir),
        "config_path": config_path,
        "transformer_config": test_model["config"],
        "model": test_model["model"],
        "tokenizer": test_tokenizer,
    }

    shutil.rmtree(test_dir)


@pytest.fixture
def random_dataset():
    dataset = RandomDataset()
    yield dataset


@pytest.fixture
def multi_turn_dataset():
    dataset = MultiTurnDataset()
    yield dataset


@pytest.fixture
def early_stopping_dataset():
    dataset = EarlyStoppingDataset()
    yield dataset
