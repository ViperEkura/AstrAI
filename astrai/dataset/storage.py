"""Storage backends for different data formats.

Layers:
  - I/O layer:       save_* / load_* functions, read/write raw files (HDF5/bin)
                      return Dict[str, List[Tensor]] — format-specific, no state
  - Store (ABC):     central abstraction, normalizes multi-segment into
                      Dict[str, List[Tensor]] per key via _normalize(),
                      fetch() uses bisect across segments — no forced concat
  - Dataset layer:   BaseDataset owns a Store, only calls store.fetch(begin, end, key)

Key properties:
  - Multi-segment:   segments kept as-is, no forced concatenation — safe for
                      datasets larger than RAM
  - Explicit length: _length = min(total elements across keys), set at load,
                      __len__ returns O(1)
  - Zero-copy mmap:  MmapStore wraps np.memmap(mode="r"), all DataLoader
                       workers share OS page-cache pages
"""

import bisect
import glob
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch import Tensor

from astrai.config.preprocess_config import PipelineConfig
from astrai.factory import BaseFactory
from astrai.preprocessing.builder import MaskBuilderFactory
from astrai.preprocessing.position_id import PositionIdStrategyFactory
from astrai.serialization import (
    load_bin,
    load_h5,
)
from astrai.tokenize import AutoTokenizer

logger = logging.getLogger(__name__)


def detect_format(load_path: str) -> str:
    """Auto-detect storage format from files in the directory.

    Args:
        load_path: Directory or file path

    Returns:
        Format string ("h5", "bin", or "jsonl")

    Raises:
        FileNotFoundError: If no supported data files are found
    """
    root = Path(load_path)
    if root.is_file():
        suffix = root.suffix.lower()
        if suffix in (".h5", ".hdf5"):
            return "h5"
        if suffix == ".jsonl":
            return "jsonl"
        raise ValueError(f"Unsupported file format: {suffix}")

    h5_files = [
        Path(p)
        for pattern in ("*.h5", "*.hdf5")
        for p in glob.glob(str(root / "**" / pattern), recursive=True)
    ]
    if h5_files:
        return "h5"
    bin_files = [Path(p) for p in glob.glob(str(root / "**" / "*.bin"), recursive=True)]
    if bin_files:
        has_meta = (root / "meta.json").exists() or len(
            [Path(p) for p in glob.glob(str(root / "**" / "meta.json"), recursive=True)]
        ) > 0
        if has_meta:
            return "bin"
    jsonl_files = [
        Path(p) for p in glob.glob(str(root / "**" / "*.jsonl"), recursive=True)
    ]
    if jsonl_files:
        return "jsonl"
    raise FileNotFoundError(f"No supported data files found at {load_path}")


class Store(ABC):
    """String keys -> segmented tensors with ``fetch(begin, end, keys)``.

    Each key maps to one or more tensor segments (no forced concatenation).
    ``len(store)`` returns ``self._length`` (explicit, O(1)), the minimum
    total element count across all keys.

    Subclasses fill ``self._data`` and ``self._cum`` during ``load()``
    via ``_normalize()``.
    """

    def __init__(self):
        self._data: Dict[str, List[Tensor]] = {}
        self._cum: Dict[str, List[int]] = {}
        self._length: int = 0

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @property
    def keys(self) -> List[str]:
        return list(self._data.keys())

    def __len__(self) -> int:
        return self._length

    def fetch(
        self,
        begin: int,
        end: int,
        keys: Union[str, List[str]],
    ):
        if not self._data:
            raise RuntimeError("Store not loaded")
        if not (0 <= begin < self._length and 0 <= end <= self._length):
            raise ValueError(
                f"Index out of bounds: begin={begin}, end={end}, length={self._length}"
            )
        if isinstance(keys, str):
            return self._fetch_key(keys, begin, end)
        return {k: self._fetch_key(k, begin, end) for k in keys}

    def _fetch_key(self, key: str, begin: int, end: int) -> Tensor:
        """Fetch slice [begin, end) across potentially multiple segments."""
        segments = self._data[key]
        cum = self._cum[key]
        seg_start = bisect.bisect_right(cum, begin)
        seg_end = bisect.bisect_left(cum, end)

        results = []
        for i in range(seg_start, seg_end + 1):
            prev = cum[i - 1] if i > 0 else 0
            s = max(begin - prev, 0)
            e = min(end - prev, segments[i].shape[0])
            results.append(segments[i][s:e])

        return results[0] if len(results) == 1 else torch.cat(results, dim=0)

    def _normalize(self, raw: Dict[str, List[Tensor]]):
        """Register segments and pre-compute cumulative lengths.

        Does NOT concatenate — segments are kept as-is to avoid OOM on
        large datasets.  Sets ``self._length`` to the minimum total
        element count across all keys.
        """
        for key, tensors in raw.items():
            self._data[key] = tensors
            cum = []
            total = 0
            for t in tensors:
                total += t.shape[0]
                cum.append(total)
            self._cum[key] = cum
        self._length = (
            min((cum[-1] if cum else 0) for cum in self._cum.values())
            if self._cum
            else 0
        )


class StoreFactory(BaseFactory["Store"]):
    """Factory for creating Store instances by type name.

    Example::

        @StoreFactory.register("custom")
        class CustomStore(Store):
            ...
    """


@StoreFactory.register("h5")
class H5Store(Store):
    """HDF5-based storage backend (pre-tokenized data)."""

    def load(self, path: str):
        self._normalize(load_h5(path))


@StoreFactory.register("bin")
class MmapStore(Store):
    """Memory-mapped binary storage backend.

    Each key is a single .bin file backed by ``np.memmap(mode="r")``.
    No per-process memory duplication — all DataLoader workers share the
    same OS page-cache pages.

    Format on disk::

        data_root/
          meta.json          # {key: {shape, dtype}, ...}
          <key>.bin          # raw numpy array, one per key
    """

    def load(self, path: str):
        self._mmap_refs = []
        root = Path(path)
        all_raw: Dict[str, List[Tensor]] = {}
        meta_paths = [
            Path(p) for p in glob.glob(str(root / "**" / "meta.json"), recursive=True)
        ]
        for meta_path in meta_paths:
            raw = load_bin(str(meta_path.parent))
            for key, tensors in raw.items():
                if key not in all_raw:
                    all_raw[key] = []
                all_raw[key].extend(tensors)
        if not meta_paths:
            raise FileNotFoundError(f"No meta.json found under {path}")
        self._normalize(all_raw)
        for tensors in self._data.values():
            self._mmap_refs.extend(tensors)


@StoreFactory.register("jsonl")
class JsonlStore(Store):
    """On-the-fly tokenization store for raw JSONL files.

    A JSONL dataset directory contains ``*.jsonl`` files plus a
    ``dataset_config.json`` file that follows the same schema as
    :class:`PipelineConfig` with an additional ``tokenizer_path`` field.
    Records are tokenized when the store is loaded and concatenated into
    segmented tensors matching the key layout expected by the dataset
    classes (``sequence``, ``loss_mask``, ``position_ids``, ...).
    """

    CONFIG_NAME = "dataset_config.json"

    def load(self, path: str):
        root = Path(path)
        config_path = root / self.CONFIG_NAME
        if not config_path.exists():
            raise FileNotFoundError(
                f"JSONL dataset config not found: {config_path}. "
                f"Expected {self.CONFIG_NAME} alongside *.jsonl files."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)

        tokenizer_path = raw_config.pop("tokenizer_path", None)
        if tokenizer_path is None:
            raise ValueError(
                f"JSONL dataset config must specify 'tokenizer_path': {config_path}"
            )

        self.config = PipelineConfig.from_dict(raw_config)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        mask_builder = MaskBuilderFactory.create("sectioned")
        position_strategy = PositionIdStrategyFactory.create(
            self.config.output.position_ids_mode
        )

        raw: Dict[str, List[Tensor]] = {}
        doc_sequences: List[List[int]] = []

        for jsonl_path in sorted(root.glob("*.jsonl")):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse JSON line in %s, skipping", jsonl_path
                        )
                        continue

                    result = mask_builder.build(item, self.config, tokenizer)
                    if result is None:
                        continue

                    result.pop("domain", None)
                    primary_ids = self._primary_ids(result)
                    if not primary_ids:
                        continue

                    doc_sequences.append(primary_ids)
                    for key, ids in result.items():
                        if key not in raw:
                            raw[key] = []
                        raw[key].append(torch.tensor(ids, dtype=self._infer_dtype(ids)))

        pos_ids = position_strategy.generate(doc_sequences)
        if pos_ids:
            raw["position_ids"] = [torch.tensor(pos_ids, dtype=torch.int32)]

        self._normalize(raw)

    @staticmethod
    def _primary_ids(result: dict) -> List[int]:
        """Return the first integer list in *result* as the primary id sequence."""
        for val in result.values():
            if isinstance(val, list) and val and isinstance(val[0], int):
                return val
        return []

    @staticmethod
    def _infer_dtype(ids: List) -> torch.dtype:
        """Infer tensor dtype from the first element of a token/value list."""
        if ids and isinstance(ids[0], float):
            return torch.float32
        return torch.int32
