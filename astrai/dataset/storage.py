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
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from astrai.config.preprocess_config import PipelineConfig
from astrai.factory import BaseFactory
from astrai.preprocessing.builder import MaskBuilderFactory
from astrai.preprocessing.position_id import PositionIdStrategyFactory
from astrai.serialization import (
    load_bin,
    load_bin_offsets,
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
    json_files = [
        Path(p) for p in glob.glob(str(root / "**" / "*.json"), recursive=True)
    ]
    if json_files:
        return "jsonl"
    raise FileNotFoundError(f"No supported data files found at {load_path}")


class Store(ABC):
    """String keys -> segmented tensors with two access modes.

    Stream mode (SEQ/SFT):
        ``fetch(begin, end, keys)`` slices across concatenated segments,
        transparently ``torch.cat``-ing across segment boundaries.
        ``len(store)`` returns total token count.

    Record mode (DPO/GRPO):
        ``fetch_record(index, keys)`` returns the i-th record without
        cross-record concatenation.  ``num_records`` returns the record
        count.  Backed by either per-record segment lists (H5/JSONL) or
        a single concatenated segment plus per-record offsets (bin).

    Subclasses fill ``self._data`` and ``self._cum`` during ``load()``
    via ``_normalize()``.
    """

    def __init__(self):
        self._data: Dict[str, List[Tensor]] = {}
        self._cum: Dict[str, List[int]] = {}
        self._offsets: Dict[str, List[int]] = {}
        self._length: int = 0
        self._num_records: int = 0

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

    @property
    def num_records(self) -> int:
        return self._num_records

    def fetch_record(
        self,
        index: int,
        keys: Union[str, List[str]],
    ):
        """Fetch the *index*-th record without cross-record concatenation.

        Returns a tensor (flat key) or ``List[Tensor]`` (nested key such as
        GRPO ``responses``).
        """
        if not self._data:
            raise RuntimeError("Store not loaded")
        if not 0 <= index < self._num_records:
            raise ValueError(
                f"Record index out of bounds: {index}, num_records={self._num_records}"
            )
        if isinstance(keys, str):
            return self._fetch_record_key(keys, index)
        return {k: self._fetch_record_key(k, index) for k in keys}

    def _fetch_record_key(self, key: str, index: int):
        """Return the *index*-th record for *key*.

        Two storage layouts are supported:

        - **bin + offsets**: ``_data[key]`` is ``[single_long_segment]``;
          ``_offsets[key]`` holds cumulative per-record offsets.  The record
          is sliced as ``segment[offsets[i]:offsets[i+1]]``.
        - **h5 / jsonl**: ``_data[key]`` is ``[t0, t1, ...]`` with one tensor
          (or nested list of tensors for GRPO) per record.  Direct indexing.
        """
        offsets = self._offsets.get(key)
        if offsets:
            start = offsets[index]
            end = (
                offsets[index + 1]
                if index + 1 < len(offsets)
                else self._data[key][0].shape[0]
            )
            return self._data[key][0][start:end]
        return self._data[key][index]

    def _normalize(
        self,
        raw: Dict[str, list],
        offsets: Optional[Dict[str, List[int]]] = None,
        per_record: bool = False,
    ):
        """Register segments and pre-compute indices for both access modes.

        Stream mode: ``_cum[key]`` accumulates per-segment lengths so
        ``_fetch_key`` can bisect across segments without concatenation.

        Record mode: if *offsets* is provided (bin layout), ``_offsets[key]``
        stores cumulative per-record offsets into the single concatenated
        segment.  Otherwise (h5/jsonl layout), ``_data[key]`` is already a
        per-record list and ``fetch_record`` indexes it directly.

        Nested keys (GRPO ``responses``/``masks`` as ``List[List[Tensor]]``)
        are stored as-is and excluded from both cumulative bookkeepings —
        they are only accessed record-by-record.
        """
        flat_lengths = []
        for key, tensors in raw.items():
            self._data[key] = tensors
            if not tensors:
                self._cum[key] = []
                flat_lengths.append(0)
                continue
            # Skip nested lists (GRPO responses/masks) — record-level access
            if isinstance(tensors[0], list):
                self._cum[key] = []
                continue
            cum = []
            total = 0
            for t in tensors:
                total += t.shape[0]
                cum.append(total)
            self._cum[key] = cum
            flat_lengths.append(cum[-1] if cum else 0)
        self._length = min(flat_lengths) if flat_lengths else 0

        # Record-mode offsets (bin layout).  Only valid when each key is a
        # single concatenated segment — multi-shard bin + offsets is not
        # supported (merge shards or use H5/JSONL instead).
        valid_offsets: Dict[str, List[int]] = {}
        if offsets:
            for key, off in offsets.items():
                segs = self._data.get(key, [])
                if len(segs) == 1 and len(off) > 1:
                    valid_offsets[key] = off
                elif len(segs) > 1:
                    logger.warning(
                        "Key '%s' has %d segments with offsets — record mode "
                        "disabled for this key (multi-shard bin+offsets not "
                        "supported). Merge shards or use H5/JSONL.",
                        key,
                        len(segs),
                    )
        self._offsets = valid_offsets
        if valid_offsets:
            record_counts = [len(v) - 1 for v in valid_offsets.values()]
            self._num_records = min(record_counts) if record_counts else 0
        elif per_record:
            # H5/JSONL layout: _data[key] is a per-record list where each
            # segment is one record.  Even a single segment counts as one
            # record.
            per_record_counts = []
            for key, tensors in self._data.items():
                if not tensors or isinstance(tensors[0], list):
                    continue
                per_record_counts.append(len(tensors))
            self._num_records = min(per_record_counts) if per_record_counts else 0
        else:
            # bin layout without offsets: _data[key] is [concatenated_stream].
            # Cannot determine record boundaries — stream mode only.
            self._num_records = 0


class StoreFactory(BaseFactory["Store"]):
    """Factory for creating Store instances by type name.

    Example::

        @StoreFactory.register("custom")
        class CustomStore(Store):
            ...
    """


@StoreFactory.register("h5")
class H5Store(Store):
    """HDF5-based storage backend (pre-tokenized data).

    Each key is stored as a group of per-record datasets (``data_0``,
    ``data_1``, …), so record mode indexes ``_data[key]`` directly.
    Stream mode concatenates across records via ``_cum``.
    """

    def load(self, path: str):
        self._normalize(load_h5(path), per_record=True)


@StoreFactory.register("bin")
class MmapStore(Store):
    """Memory-mapped binary storage backend.

    Each key is a single .bin file backed by ``np.memmap(mode="r")``.
    No per-process memory duplication — all DataLoader workers share the
    same OS page-cache pages.

    When ``meta.json`` contains per-record ``offsets`` for a key (written
    via ``save_bin(..., record_keys=...)``), record-mode access slices
    individual records from the concatenated memmap.  Legacy bin files
    without offsets only support stream mode.

    Format on disk::

        data_root/
          meta.json          # {key: {shape, dtype, offsets?}, ...}
          <key>.bin          # raw numpy array, one per key
    """

    def load(self, path: str):
        self._mmap_refs = []
        root = Path(path)
        all_raw: Dict[str, List[Tensor]] = {}
        all_offsets: Dict[str, List[int]] = {}
        meta_paths = [
            Path(p) for p in glob.glob(str(root / "**" / "meta.json"), recursive=True)
        ]
        for meta_path in meta_paths:
            raw = load_bin(str(meta_path.parent))
            off = load_bin_offsets(str(meta_path.parent))
            for key, tensors in raw.items():
                if key not in all_raw:
                    all_raw[key] = []
                all_raw[key].extend(tensors)
            for key, o in off.items():
                if key not in all_offsets:
                    all_offsets[key] = []
                all_offsets[key].extend(o)
        if not meta_paths:
            raise FileNotFoundError(f"No meta.json found under {path}")
        self._normalize(all_raw, offsets=all_offsets or None)
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

        tokenizer_path = raw_config.pop("tokenizer_path", None) or str(root)
        self.config = PipelineConfig.from_dict(raw_config)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        mask_builder = MaskBuilderFactory.create("sectioned")
        position_strategy = PositionIdStrategyFactory.create(
            self.config.output.position_ids_mode
        )

        raw: Dict[str, List[Tensor]] = {}
        doc_sequences: List[List[int]] = []

        def _process_item(item: dict) -> None:
            nonlocal raw, doc_sequences
            result = mask_builder.build(item, self.config, tokenizer)
            if result is None:
                return
            result.pop("domain", None)
            primary_ids = self._primary_ids(result)
            if not primary_ids:
                return
            doc_sequences.append(primary_ids)
            for key, ids in result.items():
                if key not in raw:
                    raw[key] = []
                if ids and isinstance(ids[0], list):
                    # GRPO multi-response: List[List[int]] → List[Tensor]
                    raw[key].append(
                        [torch.tensor(sub, dtype=self._infer_dtype(sub)) for sub in ids]
                    )
                else:
                    raw[key].append(torch.tensor(ids, dtype=self._infer_dtype(ids)))

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
                    _process_item(item)

        for json_path in sorted(root.glob("*.json")):
            if json_path.name == self.CONFIG_NAME:
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON file %s, skipping", json_path)
                    continue
            if isinstance(data, list):
                for item in data:
                    _process_item(item)
            elif isinstance(data, dict):
                _process_item(data)

        pos_ids = position_strategy.generate(doc_sequences)
        if pos_ids:
            raw["position_ids"] = [torch.tensor(pos_ids, dtype=torch.int32)]

        self._normalize(raw, per_record=True)

    @staticmethod
    def _primary_ids(result: dict) -> List[int]:
        """Return the first flat integer list in *result* as the primary id sequence."""
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
