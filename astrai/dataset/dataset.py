"""Dataset implementations for training.

Composition over inheritance — every dataset is a thin wrapper that
binds a :class:`Store` to a particular train-type's key mapping.  All
sample-id → token/record indexing lives on the Store; datasets never
know about window/stride math or segment layouts.

Class hierarchy:

    BaseDataset (ABC)         — holds a Store, exposes __len__/keys,
                                 overrides __getitem__
    ├── SEQDataset            — next-token prediction (stream)
    ├── SFTDataset            — loss-mask + position_ids (stream)
    ├── DPODataset            — chosen/rejected pairs (record)
    └── GRPODataset           — prompt + response group (record)

``DatasetFactory.load(train_type, load_path, window_size, stride, …)``
builds the Store (auto-detecting format) before constructing the
matching dataset.  Passing ``store=`` skips Store construction.

When a record dataset (DPO) reads from raw JSONL, a *processor*
function (pure ``record -> Dict[str, Tensor]``) is forwarded to
:class:`JsonlStore` so tokenisation happens on the fly.
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
) -> Optional[dict]:
    """Tokenize one DPO record into chosen/rejected + masks.

    Applies the tokenizer's chat template so token sequences match the
    SFT checkpoint's format.  Prompt is rendered with
    ``add_generation_prompt=True``; chosen/rejected are appended as a
    single assistant turn.

    Accepts:

    - Flat:   ``{"prompt": str, "chosen": str, "rejected": str}``
    - Conv:   ``{"prompt": [{role, content}, ...], "chosen": [...], ...}``
    - Legacy: ``{"input": str, "chosen": str, "rejected": str}``

    No packing, no ``position_ids`` — DPO sequences are independent.
    """
    prompt = record.get("prompt") or record.get("input")
    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if prompt is None or chosen is None or rejected is None:
        return None

    prompt_messages = _to_messages(prompt)
    chosen_text = _extract_text(chosen)
    rejected_text = _extract_text(rejected)
    if chosen_text is None or rejected_text is None:
        return None
    chosen_messages = prompt_messages + [{"role": "assistant", "content": chosen_text}]
    rejected_messages = prompt_messages + [
        {"role": "assistant", "content": rejected_text}
    ]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )
    ch_ids = tokenizer.apply_chat_template(
        chosen_messages, tokenize=True, add_generation_prompt=False
    )
    re_ids = tokenizer.apply_chat_template(
        rejected_messages, tokenize=True, add_generation_prompt=False
    )

    full_ch = ch_ids[:max_len]
    full_re = re_ids[:max_len]

    prompt_len = min(len(prompt_ids), max_len)
    ch_mask = [0] * prompt_len + [1] * max(0, len(full_ch) - prompt_len)
    ch_mask = ch_mask[:max_len]
    re_mask = [0] * prompt_len + [1] * max(0, len(full_re) - prompt_len)
    re_mask = re_mask[:max_len]

    return {
        "chosen": full_ch,
        "rejected": full_re,
        "chosen_mask": ch_mask,
        "rejected_mask": re_mask,
    }


def _to_messages(value) -> list:
    """Accept str or conversation list; return message list."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list):
        return value
    return [{"role": "user", "content": str(value)}]


def _extract_text(value) -> Optional[str]:
    """Accept str or conversation list; return plain text."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(m.get("content", "") for m in value if isinstance(m, dict))
    return None


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


def validate_keys(store: Store, required: List[str]) -> None:
    """Raise ``KeyError`` if *store* is missing any *required* key."""
    if not required:
        return
    actual = set(store.keys)
    missing = [k for k in required if k not in actual]
    if missing:
        raise KeyError(
            f"Store at {getattr(store, '_load_path', '?')} is missing required "
            f"keys {missing}; available keys are {sorted(actual)}."
        )


class BaseDataset(Dataset, ABC):
    """Abstract base class for dataset types.

    Holds a :class:`Store`.  All sample-id indexing is delegated to the
    store — this class exposes ``__len__`` as ``len(store)`` and the
    ``keys`` property as ``store.keys``.  Subclasses implement
    ``__getitem__`` with the train-type-specific key mapping and any
    training-only index arithmetic (e.g. the next-token ``+1`` shift).
    """

    required_keys: List[str] = []

    def __init__(self, store: Store):
        super().__init__()
        self.store: Store = store
        validate_keys(store, self.required_keys)

    def __len__(self) -> int:
        return len(self.store)

    @property
    def keys(self) -> List[str]:
        return self.store.keys

    @property
    def token_count(self) -> int:
        return self.store.token_count

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        raise NotImplementedError


class DatasetFactory(BaseFactory["BaseDataset"]):
    """Factory for creating dataset instances by train-type.

    Use :meth:`DatasetFactory.register("custom")` to register new
    dataset classes; they must inherit from :class:`BaseDataset`.
    """

    @classmethod
    def load(
        cls,
        train_type: str,
        load_path: Optional[str] = None,
        window_size: int = 0,
        stride: Optional[int] = None,
        storage_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        max_len: int = 2048,
        store: Optional[Store] = None,
        **kwargs,
    ) -> "BaseDataset":
        """Create and load a dataset in one step.

        Two entry points:

        - **store given**: bind it directly — the caller fully controls
          Store construction and processor setup.  *load_path*,
          *storage_type*, *tokenizer_path*, *window_size*, *stride* are
          ignored.
        - **store is None**: build a Store from *load_path*, auto-detecting
          format and constructing a processor when *tokenizer_path* is
          given for a record dataset on JSONL.

        Args:
            train_type: Registered dataset name ("seq", "sft", "dpo",
                "grpo", …).
            load_path: Path to the data file or directory (ignored if
                *store* is given).
            window_size: Stream window length — only meaningful for
                stream datasets (SEQ/SFT).  Record datasets ignore it.
            stride: Stride between consecutive stream samples
                (default: same as *window_size*).
            storage_type: Storage backend ("h5", "bin", "jsonl") or
                None for auto-detection.
            tokenizer_path: Path to tokenizer for lazy JSONL
                tokenisation (record datasets only).
            max_len: Max sequence length forwarded to processors.
            store: Pre-built, already-loaded Store instance.
            **kwargs: Extra arguments forwarded to ``store.load()``.

        Returns:
            Loaded dataset instance.
        """
        if store is not None:
            return cls.create(train_type, store=store)

        if load_path is None:
            raise ValueError("Either load_path or store must be provided")

        if storage_type is None:
            storage_type = detect_format(load_path)

        if stride is None:
            stride = window_size

        processor = cls._maybe_build_processor(
            train_type, storage_type, tokenizer_path, max_len
        )

        store_window = cls._store_window_for(train_type, window_size)
        store = StoreFactory.create(
            storage_type,
            window_size=store_window,
            stride=stride if stride else store_window,
        )
        if processor is not None:
            store.load(load_path, processor=processor, **kwargs)
        else:
            load_kwargs = dict(kwargs)
            if (
                tokenizer_path is not None
                and storage_type == "jsonl"
                and train_type in ("seq", "sft")
                and "tokenizer_path" not in load_kwargs
            ):
                load_kwargs["tokenizer_path"] = tokenizer_path
            store.load(load_path, **load_kwargs)

        return cls.create(train_type, store=store)

    @staticmethod
    def _store_window_for(train_type: str, window_size: int) -> int:
        """Stream datasets consume ``window_size``; record datasets ignore it.

        Record datasets (dpo/grpo) treat each record as an independent
        training unit and never window, so the store is built with
        ``window_size=0`` and ``len(store)`` returns the record count.
        """
        if train_type in ("seq", "sft"):
            return window_size
        return 0

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
    """Dataset for sequential next-token prediction training.

    Stream mode: ``store.fetch(begin, end, "sequence")`` returns the
    input window; the +1 shifted call returns the next-token target.
    """

    required_keys = ["sequence"]

    def __getitem__(self, index: int):
        begin, end = self.store.sample_window(index)
        x = self.store.fetch(begin, end, "sequence")
        y = self.store.fetch(begin + 1, end + 1, "sequence")
        return {
            "input_ids": x.to(dtype=torch.long),
            "target_ids": y.to(dtype=torch.long),
        }


@DatasetFactory.register("sft")
class SFTDataset(BaseDataset):
    """Dataset for supervised fine-tuning with loss masking.

    Stream mode: ``sequence``/``loss_mask``/``position_ids`` are sliced
    to the window.  ``loss_mask`` and ``target_ids`` use the +1 shifted
    slice so they align with the predicted positions.
    """

    required_keys = ["sequence", "loss_mask", "position_ids"]

    def __getitem__(self, index: int):
        begin, end = self.store.sample_window(index)
        x = self.store.fetch(begin, end, "sequence")
        y = self.store.fetch(begin + 1, end + 1, "sequence")
        position_ids = self.store.fetch(begin, end, "position_ids")
        loss_mask = self.store.fetch(begin + 1, end + 1, "loss_mask")
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

    Two loading paths (handled by :class:`DatasetFactory`):

    - **Pre-tokenized** (H5/bin): ``store.load(path)`` reads per-record
      tensors; ``__getitem__`` returns them directly.
    - **Raw JSONL** (``tokenizer_path=...``): builds a lazy processor
      via :func:`dpo_processor` that tokenises on the fly — no packing,
      no ``position_ids``.
    """

    required_keys = ["chosen", "rejected", "chosen_mask", "rejected_mask"]

    def make_processor(self, tokenizer, max_len: int):
        return partial(dpo_processor, tokenizer=tokenizer, max_len=max_len)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "chosen": self.store.fetch_record(index, "chosen").to(dtype=torch.long),
            "rejected": self.store.fetch_record(index, "rejected").to(dtype=torch.long),
            "chosen_mask": self.store.fetch_record(index, "chosen_mask").to(
                dtype=torch.bool
            ),
            "rejected_mask": self.store.fetch_record(index, "rejected_mask").to(
                dtype=torch.bool
            ),
        }


@DatasetFactory.register("grpo")
class GRPODataset(BaseDataset):
    """Dataset for offline Group Relative Policy Optimization.

    Each sample is one prompt with its group of responses and scalar
    rewards — an independent training unit with no windowing or stride.

    Expected storage layout (produced by JsonlStore or pre-tokenized):

    - ``prompts``:   List[Tensor]  — one 1-D token tensor per record
    - ``responses``: List[List[Tensor]] — G response tensors per record
    - ``masks``:     List[List[Tensor]] — G mask tensors per record
    - ``rewards``:   List[Tensor]  — one 1-D float tensor (len G) per record
    """

    required_keys = ["prompts", "responses", "masks", "rewards"]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        prompts = self.store.fetch_record(index, "prompts")
        responses = self.store.fetch_record(index, "responses")
        masks = self.store.fetch_record(index, "masks")
        rewards = self.store.fetch_record(index, "rewards")
        return {
            "prompts": prompts.to(dtype=torch.long),
            "responses": [r.to(dtype=torch.long) for r in responses],
            "masks": [m.to(dtype=torch.bool) for m in masks],
            "rewards": rewards.to(dtype=torch.float32),
        }
