"""Dataset implementations with factory pattern for training."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from astrai.dataset.storage import (
    Store,
    StoreFactory,
    detect_format,
)
from astrai.factory import BaseFactory


class BaseDataset(Dataset, ABC):
    """Abstract base class for all dataset types.

    Implements common functionality for window-based data fetching.
    Uses a storage abstraction for format-agnostic data loading.
    """

    def __init__(self, window_size: int, stride: int):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.storage: Optional[Store] = None

    @property
    def required_keys(self) -> List[str]:
        """Return required storage keys for this dataset type.

        Subclasses should override to specify expected keys.
        """
        return []

    def _validate_keys(self):
        if not self.required_keys:
            return
        actual_keys = set(self.storage.keys)
        missing = [k for k in self.required_keys if k not in actual_keys]
        if missing:
            raise KeyError(
                f"Dataset {type(self).__name__} requires keys {self.required_keys}, "
                f"but storage at {self._load_path} only has {sorted(actual_keys)}. "
                f"Missing: {missing}"
            )

    def load(self, load_path: str, storage_type: Optional[str] = None, **kwargs):
        """Load dataset from the given path.

        Auto-detects the storage format if not specified.

        Args:
            load_path: Path to the data directory or file
            storage_type: Force a specific storage type ("h5", "bin", "jsonl"),
                          or None for auto-detection
            **kwargs: Extra arguments forwarded to the store constructor and
                      to ``store.load()``.

        Raises:
            KeyError: If the loaded storage is missing required keys.
        """
        if storage_type is None:
            storage_type = detect_format(load_path)
        self.storage = StoreFactory.create(storage_type, **kwargs)
        self._load_path = load_path
        self.storage.load(load_path, **kwargs)
        self._validate_keys()

    @property
    def count(self) -> int:
        """Return the total number of raw elements (tokens) in the dataset."""
        if self.storage is None:
            return 0
        return len(self.storage)

    @property
    def keys(self) -> List[str]:
        """Return the available data keys."""
        if self.storage is None:
            return []
        return self.storage.keys

    def get_index(self, index: int) -> tuple:
        """Calculate begin and end indices for a sample.

        Args:
            index: Sample index

        Returns:
            Tuple of (begin_idx, end_idx)
        """
        if self.storage is None:
            raise RuntimeError("Dataset not loaded, call load() first")
        total = len(self.storage)
        if total <= self.window_size:
            raise ValueError(
                f"Data too short: {total} tokens <= window_size {self.window_size}"
            )

        begin_idx = min(index * self.stride, total - 1 - self.window_size)
        end_idx = min(begin_idx + self.window_size, total - 1)

        return begin_idx, end_idx

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Get a single sample by index.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        if self.storage is None:
            return 0
        total = len(self.storage)
        if total <= self.window_size:
            return 0
        return (total - 1 - self.window_size) // self.stride + 1


class DatasetFactory(BaseFactory["BaseDataset"]):
    """Factory class for creating dataset instances.

    Supports decorator-based registration for extensible dataset types.
    All default dataset types (seq, sft, dpo, grpo) are registered automatically
    when their classes are defined with the decorator.

    Example usage:
        @DatasetFactory.register("custom")
        class CustomDataset(BaseDataset):
            ...

        dataset = DatasetFactory.create("custom", window_size, stride)
    """

    @classmethod
    def load(
        cls,
        train_type: str,
        load_path: str,
        window_size: int,
        stride: Optional[int] = None,
        storage_type: Optional[str] = None,
        **kwargs,
    ) -> "BaseDataset":
        """Create and load a dataset in one step.

        Args:
            train_type: Type of training dataset
            load_path: Path to the data file
            window_size: Window size for data sampling
            stride: Stride between consecutive samples (default: same as window_size)
            storage_type: Storage type ("h5", "bin", "jsonl") or None for auto-detection
            **kwargs: Extra arguments forwarded to ``dataset.load()``.

        Returns:
            Loaded dataset instance
        """
        if stride is None:
            stride = window_size

        dataset = cls.create(train_type, window_size, stride)
        dataset.load(load_path, storage_type=storage_type, **kwargs)

        return dataset


@DatasetFactory.register("seq")
class SEQDataset(BaseDataset):
    """Dataset for sequential next-token prediction training."""

    @property
    def required_keys(self) -> List[str]:
        return ["sequence"]

    def _fetch_data(self, begin_idx: int, end_idx: int) -> Tensor:
        return self.storage.fetch(begin_idx, end_idx, "sequence")

    def __getitem__(self, index):
        begin_idx, end_idx = self.get_index(index)

        x = self._fetch_data(begin_idx, end_idx).to(dtype=torch.long)
        y = self._fetch_data(begin_idx + 1, end_idx + 1).to(dtype=torch.long)

        return {"input_ids": x, "target_ids": y}


@DatasetFactory.register("sft")
class SFTDataset(BaseDataset):
    """Dataset for supervised fine-tuning with loss masking."""

    @property
    def required_keys(self) -> List[str]:
        return ["sequence", "loss_mask", "position_ids"]

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.storage.fetch(begin_idx, end_idx, key)

    def __getitem__(self, index):
        begin_idx, end_idx = self.get_index(index)

        x = self._fetch_data(begin_idx, end_idx, "sequence")
        y = self._fetch_data(begin_idx + 1, end_idx + 1, "sequence")
        position_ids = self._fetch_data(begin_idx, end_idx, "position_ids")
        loss_mask = self._fetch_data(begin_idx + 1, end_idx + 1, "loss_mask")

        return {
            "input_ids": x.to(dtype=torch.long),
            "target_ids": y.to(dtype=torch.long),
            "position_ids": position_ids.to(dtype=torch.long),
            "loss_mask": loss_mask.to(dtype=torch.bool),
        }


@DatasetFactory.register("dpo")
class DPODataset(BaseDataset):
    """Dataset for Direct Preference Optimization training."""

    @property
    def required_keys(self) -> List[str]:
        return ["chosen", "rejected", "chosen_mask", "rejected_mask"]

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.storage.fetch(begin_idx, end_idx, key)

    def __getitem__(self, index: int):
        begin_idx, end_idx = self.get_index(index)

        chosen = self._fetch_data(begin_idx, end_idx, "chosen").to(dtype=torch.long)
        rejected = self._fetch_data(begin_idx, end_idx, "rejected").to(dtype=torch.long)
        chosen_mask = self._fetch_data(begin_idx, end_idx, "chosen_mask").to(
            dtype=torch.bool
        )
        rejected_mask = self._fetch_data(begin_idx, end_idx, "rejected_mask").to(
            dtype=torch.bool
        )

        return {
            "chosen": chosen,
            "rejected": rejected,
            "chosen_mask": chosen_mask,
            "rejected_mask": rejected_mask,
        }


@DatasetFactory.register("grpo")
class GRPODataset(BaseDataset):
    """Dataset for Group Relative Policy Optimization training."""

    @property
    def required_keys(self) -> List[str]:
        return ["prompts", "responses", "masks", "rewards"]

    def _fetch_data(self, begin_idx: int, end_idx: int, key: str) -> Tensor:
        return self.storage.fetch(begin_idx, end_idx, key)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        begin_idx, end_idx = self.get_index(index)

        prompts = self._fetch_data(begin_idx, end_idx, "prompts").to(dtype=torch.long)
        responses = self._fetch_data(begin_idx, end_idx, "responses").to(
            dtype=torch.long
        )
        masks = self._fetch_data(begin_idx, end_idx, "masks").to(dtype=torch.bool)
        rewards = self._fetch_data(begin_idx, end_idx, "rewards")

        return {
            "prompts": prompts,
            "responses": responses,
            "masks": masks,
            "rewards": rewards,
        }
