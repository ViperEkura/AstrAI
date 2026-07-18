"""Dataset implementations with factory pattern for training.

Class hierarchy:

    BaseDataset (ABC)           — load/validate, owns a Store
    ├── SEQDataset              — stream, next-token prediction (PT)
    ├── SFTDataset              — stream, loss-mask + position_ids
    └── RecordDataset           — record access, optional processor
        ├── DPODataset          — chosen/rejected pairs
        └── GRPODataset         — prompt + response group

``RecordDataset`` holds an optional *processor* (pure
``record -> Dict[str, Tensor]`` function).  When the backing Store is
a lazy JsonlStore, the processor tokenises on the fly; otherwise it
is ignored and ``fetch_record`` reads pre-tokenised tensors.

``__len__`` returns the sample count (stream: windows, record:
records) so DataLoader and progress bars work uniformly.
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset

from astrai.dataset.storage import (
    Store,
    StoreFactory,
    detect_format,
)
from astrai.factory import BaseFactory
from astrai.tokenize import AutoTokenizer


def dpo_tokenize(
    record: dict,
    tokenizer,
    max_len: int = 2048,
    pad_id: int = 2,
) -> Optional[dict]:
    """Tokenize one DPO record into chosen/rejected + masks.

    Pure processor function (HF ``datasets.map`` style):
    ``record -> dict_of_lists``.  Each value is a flat list of ints/bools.

    No packing, no ``position_ids`` — DPO sequences are independent and
    the model defaults to ``arange(0, seq_len)``.
    """
    inp = record.get("input")
    chosen_text = record.get("chosen")
    rejected_text = record.get("rejected")
    if inp is None or chosen_text is None or rejected_text is None:
        return None

    in_ids = tokenizer.encode(inp, add_special_tokens=True)
    ch_ids = tokenizer.encode(chosen_text, add_special_tokens=False)
    re_ids = tokenizer.encode(rejected_text, add_special_tokens=False)

    full_ch = (in_ids + ch_ids)[:max_len]
    full_re = (in_ids + re_ids)[:max_len]

    max_record_len = max(len(full_ch), len(full_re))
    ch_pad = full_ch + [pad_id] * (max_record_len - len(full_ch))
    re_pad = full_re + [pad_id] * (max_record_len - len(full_re))

    ch_mask = [0] * len(in_ids) + [1] * len(ch_ids)
    ch_mask = ch_mask[:max_len]
    ch_mask += [0] * (max_record_len - len(ch_mask))
    re_mask = [0] * len(in_ids) + [1] * len(re_ids)
    re_mask = re_mask[:max_len]
    re_mask += [0] * (max_record_len - len(re_mask))

    return {
        "chosen": ch_pad,
        "rejected": re_pad,
        "chosen_mask": ch_mask,
        "rejected_mask": re_mask,
    }


def dpo_processor(
    record: dict,
    tokenizer,
    max_len: int = 2048,
) -> Dict[str, Tensor]:
    """DPO processor: wraps :func:`dpo_tokenize` and returns tensors."""
    result = dpo_tokenize(record, tokenizer, max_len=max_len)
    if result is None:
        raise ValueError(f"Malformed DPO record: {list(record.keys())}")
    return {
        "chosen": torch.tensor(result["chosen"], dtype=torch.int32),
        "rejected": torch.tensor(result["rejected"], dtype=torch.int32),
        "chosen_mask": torch.tensor(result["chosen_mask"], dtype=torch.bool),
        "rejected_mask": torch.tensor(result["rejected_mask"], dtype=torch.bool),
    }


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


class RecordDataset(BaseDataset):
    """Base class for record-structured datasets (DPO/GRPO).

    Each sample is an independent record — no windowing, stride, or
    cross-record concatenation.  ``__len__`` returns the record count
    so progress bars advance per-record.

    A *processor* (pure ``record -> Dict[str, Tensor]`` function) may be
    supplied for lazy on-the-fly tokenisation of raw JSONL.  The
    processor is forwarded to ``JsonlStore`` and applied per access;
    pre-tokenised backends (H5/bin) ignore it.
    """

    def __init__(
        self,
        window_size: int = 0,
        stride: int = 0,
        processor: Optional[Callable[[dict], Dict[str, Tensor]]] = None,
        **kwargs,
    ):
        super().__init__(window_size=window_size, stride=stride or window_size)
        self.processor = processor

    def load(self, load_path: str, storage_type: Optional[str] = None, **kwargs):
        """Load data from *load_path*.

        Args:
            load_path: Path to data file or directory.
            storage_type: Force backend ("h5"/"bin"/"jsonl") or None for
                auto-detection.
            **kwargs: Forwarded to ``store.load()``.  When the backend is
                JSONL and a processor was set, it is passed as
                ``processor=`` for lazy tokenisation.
        """
        if storage_type is None:
            storage_type = detect_format(load_path)
        self.storage = StoreFactory.create(storage_type, **kwargs)
        self._load_path = load_path

        if self.processor is not None:
            self.storage.load(load_path, processor=self.processor, **kwargs)
        else:
            self.storage.load(load_path, **kwargs)
            self._validate_keys()

    def __len__(self) -> int:
        if self.storage is None:
            return 0
        return self.storage.num_records

    @property
    def count(self) -> int:
        if self.storage is None:
            return 0
        return self.storage.num_records


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
        tokenizer_path: Optional[str] = None,
        max_len: int = 2048,
        **kwargs,
    ) -> "BaseDataset":
        """Create and load a dataset in one step.

        Args:
            train_type: Type of training dataset
            load_path: Path to the data file
            window_size: Window size for data sampling
            stride: Stride between consecutive samples (default: same as window_size)
            storage_type: Storage type ("h5", "bin", "jsonl") or None for auto-detection
            tokenizer_path: Path to tokenizer. Used to build an on-the-fly
                processor when loading raw JSONL with a record dataset
                (DPO/GRPO).  Ignored for pre-tokenised backends (H5/bin)
                and for stream datasets (SEQ/SFT).
            max_len: Max sequence length for the processor.
            **kwargs: Extra arguments forwarded to ``dataset.load()``.

        Returns:
            Loaded dataset instance
        """
        if stride is None:
            stride = window_size

        if storage_type is None:
            storage_type = detect_format(load_path)

        processor = cls._maybe_build_processor(
            train_type, storage_type, tokenizer_path, max_len
        )

        dataset = cls.create(train_type, window_size, stride, processor=processor)
        dataset.load(load_path, storage_type=storage_type, **kwargs)

        return dataset

    @classmethod
    def from_store(
        cls,
        train_type: str,
        store: Store,
        window_size: int = 0,
        stride: Optional[int] = None,
    ) -> "BaseDataset":
        """Create a dataset bound to an already-loaded store.

        The caller is responsible for constructing and loading the store
        (including any processor).  The dataset simply wraps it.
        """
        if stride is None:
            stride = window_size
        dataset = cls.create(train_type, window_size, stride)
        dataset.storage = store
        return dataset

    @staticmethod
    def _maybe_build_processor(
        train_type: str,
        storage_type: str,
        tokenizer_path: Optional[str],
        max_len: int,
    ) -> Optional[Callable[[dict], Dict[str, Tensor]]]:
        """Build an on-the-fly tokenisation processor if applicable.

        Only raw JSONL + record datasets (DPO/GRPO) need a processor;
        pre-tokenised backends (H5/bin) and stream datasets (SEQ/SFT)
        return ``None`` so no tokenizer is loaded.
        """
        if tokenizer_path is None or storage_type != "jsonl":
            return None
        if train_type == "dpo":
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            return partial(dpo_processor, tokenizer=tokenizer, max_len=max_len)
        return None


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
class DPODataset(RecordDataset):
    """Record-structured dataset for Direct Preference Optimization.

    Each sample is one preference pair (chosen + rejected) and is an
    independent training unit — no windowing, stride, or cross-record
    concatenation.  This keeps each sequence self-contained so attention
    never leaks across preference pairs.

    Two loading paths (handled by :class:`RecordDataset`):

    - **Pre-tokenized** (H5/bin): ``load(path)`` reads per-record tensors,
      ``__getitem__`` returns them directly.
    - **Raw JSONL** (``tokenizer_path=...``): builds a lazy processor via
      :func:`dpo_processor` that tokenises on the fly — no packing, no
      ``position_ids``.
    """

    @property
    def required_keys(self) -> List[str]:
        return ["chosen", "rejected", "chosen_mask", "rejected_mask"]

    def make_processor(self, tokenizer, max_len: int):
        return partial(dpo_processor, tokenizer=tokenizer, max_len=max_len)

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
class GRPODataset(RecordDataset):
    """Dataset for offline Group Relative Policy Optimization.

    Each sample is one prompt with its group of responses and scalar
    rewards — an independent training unit with no windowing or stride.

    Expected storage layout (produced by JsonlStore or pre-tokenized):

    - ``prompts``:   List[Tensor]  — one 1-D token tensor per record
    - ``responses``: List[List[Tensor]] — G response tensors per record
    - ``masks``:     List[List[Tensor]] — G mask tensors per record
    - ``rewards``:   List[Tensor]  — one 1-D float tensor (len G) per record
    """

    @property
    def required_keys(self) -> List[str]:
        return ["prompts", "responses", "masks", "rewards"]

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
