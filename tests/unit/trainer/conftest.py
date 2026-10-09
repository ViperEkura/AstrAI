import pytest

from tests.support.rollout import RandomTokenDataset
from tests.support.trainer import create_train_config


@pytest.fixture
def train_config_factory():
    """Fixture providing the ``create_train_config`` factory function."""
    return create_train_config


@pytest.fixture
def trainer_dataset():
    """Fixture providing a dataset for trainer tests."""
    return RandomTokenDataset()
