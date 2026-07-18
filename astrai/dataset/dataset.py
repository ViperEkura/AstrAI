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


def dpo_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate variable-length DPO samples into padded 2-D tensors.

    Input: list of dicts, each with:
      - chosen:        [C_i]
      - rejected:      [R_i]
      - chosen_mask:   [C_i]
      - rejected_mask: [R_i]

    Output (padded to the max length across chosen/rejected within the batch):
      - chosen:        [B, S_max]
      - rejected:      [B, S_max]
      - chosen_mask:   [B, S_max]
      - rejected_mask: [B, S_max]
    """
    B = len(batch)
    S_max = max(b["chosen"].size(0) for b in batch)
    S_max = max(S_max, max(b["rejected"].size(0) for b in batch))

    chosen = torch.zeros(B, S_max, dtype=torch.long)
    rejected = torch.zeros(B, S_max, dtype=torch.long)
    chosen_mask = torch.zeros(B, S_max, dtype=torch.bool)
    rejected_mask = torch.zeros(B, S_max, dtype=torch.bool)

    for i, b in enumerate(batch):
        c_len = b["chosen"].size(0)
        r_len = b["rejected"].size(0)
        chosen[i, :c_len] = b["chosen"]
        rejected[i, :r_len] = b["rejected"]
        chosen_mask[i, :c_len] = b["chosen_mask"]
        rejected_mask[i, :r_len] = b["rejected_mask"]

    return {
        "chosen": chosen,
        "rejected": rejected,
        "chosen_mask": chosen_mask,
        "rejected_mask": rejected_mask,
    }


def grpo_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate variable-length GRPO samples into padded 3-D tensors.

    Input: list of dicts, each with:
      - prompts:  [P_i]
      - responses: list of G tensors, each [R_ij]
      - masks:     list of G tensors, each [R_ij]
      - rewards:  [G]

    Output:
      - prompts:   [B, P_max]
      - responses: [B, G, R_max]
      - masks:     [B, G, R_max]
      - rewards:   [B, G]
    """
    B = len(batch)
    G = len(batch[0]["responses"])
    P_max = max(b["prompts"].size(0) for b in batch)
    R_max = max(r.size(0) for b in batch for r in b["responses"])

    prompts = torch.zeros(B, P_max, dtype=torch.long)
    responses = torch.zeros(B, G, R_max, dtype=torch.long)
    masks = torch.zeros(B, G, R_max, dtype=torch.bool)
    rewards = torch.zeros(B, G, dtype=torch.float32)

    for i, b in enumerate(batch):
        p_len = b["prompts"].size(0)
        prompts[i, :p_len] = b["prompts"]
        rewards[i, : b["rewards"].size(0)] = b["rewards"]
        for g in range(min(G, len(b["responses"]))):
            r_len = b["responses"][g].size(0)
            responses[i, g, :r_len] = b["responses"][g]
            if g < len(b["masks"]):
                masks[i, g, :r_len] = b["masks"][g]

    return {
        "prompts": prompts,
        "responses": responses,
        "masks": masks,
        "rewards": rewards,
    }


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
    """Record-structured dataset for Direct Preference Optimization.

    Each sample is one preference pair (chosen + rejected) and is an
    independent training unit — no windowing, stride, or cross-record
    concatenation.  This keeps each sequence self-contained so attention
    never leaks across preference pairs.

    Delegates record access to ``Store.fetch_record``, which works with
    any storage backend (H5 per-record datasets, bin+offsets memmap, or
    JSONL on-the-fly tokenization).
    """

    def __init__(self, window_size: int = 0, stride: int = 0, **kwargs):
        super().__init__(window_size=window_size, stride=stride or window_size)

    @property
    def required_keys(self) -> List[str]:
        return ["chosen", "rejected", "chosen_mask", "rejected_mask"]

    def load(self, load_path: str, storage_type: Optional[str] = None, **kwargs):
        if storage_type is None:
            storage_type = detect_format(load_path)
        self.storage = StoreFactory.create(storage_type, **kwargs)
        self._load_path = load_path
        self.storage.load(load_path, **kwargs)
        self._validate_keys()

    def _validate_keys(self):
        actual_keys = set(self.storage.keys)
        missing = [k for k in self.required_keys if k not in actual_keys]
        if missing:
            raise KeyError(
                f"DPODataset requires keys {self.required_keys}, "
                f"but storage only has {sorted(actual_keys)}. Missing: {missing}"
            )

    @property
    def count(self) -> int:
        return self.storage.num_records

    def __len__(self) -> int:
        return self.storage.num_records

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "chosen": self.storage.fetch_record(index, "chosen").to(dtype=torch.long),
            "rejected": self.storage.fetch_record(index, "rejected").to(
                dtype=torch.long
            ),
            "chosen_mask": self.storage.fetch_record(index, "chosen_mask").to(
                dtype=torch.bool
            ),
            "rejected_mask": self.storage.fetch_record(index, "rejected_mask").to(
                dtype=torch.bool
            ),
        }


@DatasetFactory.register("grpo")
class GRPODataset(BaseDataset):
    """Dataset for offline Group Relative Policy Optimization.

    Unlike the window-based datasets (SEQ/SFT/DPO), GRPO data is
    record-structured: each sample is one prompt with its group of
    responses and scalar rewards.  There is no windowing or stride —
    every record is an independent training unit.

    Expected storage layout (produced by JsonlStore or pre-tokenized):

    - ``prompts``:   List[Tensor]  — one 1-D token tensor per record
    - ``responses``: List[List[Tensor]] — G response tensors per record
    - ``masks``:     List[List[Tensor]] — G mask tensors per record
    - ``rewards``:   List[Tensor]  — one 1-D float tensor (len G) per record
    """

    def __init__(self, window_size: int = 0, stride: int = 0, **kwargs):
        super().__init__(window_size=window_size, stride=stride or window_size)

    @property
    def required_keys(self) -> List[str]:
        return ["prompts", "responses", "masks", "rewards"]

    def load(self, load_path: str, storage_type: Optional[str] = None, **kwargs):
        if storage_type is None:
            storage_type = detect_format(load_path)
        self.storage = StoreFactory.create(storage_type, **kwargs)
        self._load_path = load_path
        self.storage.load(load_path, **kwargs)
        self._validate_keys()

    def _validate_keys(self):
        actual_keys = set(self.storage.keys)
        missing = [k for k in self.required_keys if k not in actual_keys]
        if missing:
            raise KeyError(
                f"GRPODataset requires keys {self.required_keys}, "
                f"but storage only has {sorted(actual_keys)}. Missing: {missing}"
            )

    @property
    def count(self) -> int:
        return self.storage.num_records

    def __len__(self) -> int:
        return self.storage.num_records

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        prompts = self.storage.fetch_record(index, "prompts")
        responses = self.storage.fetch_record(index, "responses")
        masks = self.storage.fetch_record(index, "masks")
        rewards = self.storage.fetch_record(index, "rewards")
        return {
            "prompts": prompts.to(dtype=torch.long),
            "responses": [r.to(dtype=torch.long) for r in responses],
            "masks": [m.to(dtype=torch.bool) for m in masks],
            "rewards": rewards.to(dtype=torch.float32),
        }
