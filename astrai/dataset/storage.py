"""Storage backends for different data formats.

Architecture (mixin composition, no diamond inheritance):

    Store (ABC)              — shared _data/_cum/_offsets bookkeeping
                               + _normalize() for registering segments
    Streamable (mixin)       — fetch(begin, end, key) for stream access
    Recordable (mixin)       — fetch_record(i, key) for record access

    H5Store(Store, Streamable, Recordable)
    MmapStore(Store, Streamable, Recordable)
    JsonlStore(Store, Streamable, Recordable)

Each mixin is a stateless trait that relies on ``self._data`` etc.
provided by :class:`Store`.  Concrete stores mix in whichever access
modes they support — ``Store`` is the sole base class, so there is no
diamond inheritance or MRO ambiguity.

Access-mode semantics:

- **Stream** (SEQ/SFT): ``fetch(begin, end, key)`` slices across
  concatenated segments.  ``len(store)`` returns the total token count.
- **Record** (DPO/GRPO): ``fetch_record(i, key)`` returns the *i*-th
  record without cross-record concatenation.  ``num_records`` returns
  the record count.

``segments_are_records`` (class attribute on each Store subclass)
tells ``_normalize`` whether segments are inherently per-record (H5/
JSONL) or opaque shards (bin).  Record access for bin relies on
``_offsets`` instead.

:class:`JsonlStore` supports a lazy mode (``processor=fn``) that keeps
raw records and defers tokenisation to ``fetch_record`` — used by DPO
to train directly from a ``.jsonl`` file without a pre-tokenised copy.
"""

import bisect
import glob
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
from torch import Tensor

from astrai.factory import BaseFactory
from astrai.preprocessing.transform import TokenizeTransform
from astrai.serialization import (
    load_bin,
    load_bin_offsets,
    load_h5,
)

logger = logging.getLogger(__name__)


def detect_format(load_path: str) -> str:
    """Auto-detect storage format from files in the directory.

    Args:
        load_path: Directory or file path

    Returns:
        Format string ("h5", "bin", "jsonl", or "processed")

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
    """Common base for all storage backends.

    Owns the shared ``_data`` / ``_cum`` / ``_offsets`` bookkeeping and
    the ``_normalize`` entry point used by tensor-backed subclasses.
    Does **not** expose an access API — that is the job of
    :class:`StreamStore` and :class:`RecordStore`.
    """

    segments_are_records: bool = False

    def __init__(self):
        self._data: Dict[str, List[Tensor]] = {}
        self._cum: Dict[str, List[int]] = {}
        self._offsets: Dict[str, List[int]] = {}
        self._length: int = 0
        self._num_records: int = 0

    @abstractmethod
    def load(self, path: str, **kwargs) -> None:
        raise NotImplementedError

    @property
    def keys(self) -> List[str]:
        return list(self._data.keys())

    def __len__(self) -> int:
        """Default: token count (stream semantics).

        Subclasses that are record-only (e.g. lazy JsonlStore) override
        to return ``self._num_records``.
        """
        return self._length

    def _normalize(
        self,
        raw: Dict[str, list],
        offsets: Optional[Dict[str, List[int]]] = None,
    ):
        """Register segments and pre-compute indices for both access modes.

        Stream mode: ``_cum[key]`` accumulates per-segment lengths so
        ``StreamStore._fetch_key`` can bisect across segments without
        concatenation.

        Record mode: if *offsets* is provided (bin layout),
        ``_offsets[key]`` stores cumulative per-record offsets into the
        single concatenated segment.  Otherwise, when
        ``segments_are_records`` is True (H5/JSONL), ``_data[key]`` is a
        per-record list and ``fetch_record`` indexes it directly.

        Nested keys (GRPO ``responses``/``masks`` as
        ``List[List[Tensor]]``) are stored as-is and excluded from both
        cumulative bookkeepings — they are only accessed record-by-record.
        """
        flat_lengths = []
        for key, tensors in raw.items():
            self._data[key] = tensors
            if not tensors:
                self._cum[key] = []
                flat_lengths.append(0)
                continue
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
        elif self.segments_are_records:
            per_record_counts = []
            for key, tensors in self._data.items():
                if not tensors or isinstance(tensors[0], list):
                    continue
                per_record_counts.append(len(tensors))
            self._num_records = min(per_record_counts) if per_record_counts else 0
        else:
            self._num_records = 0


class Streamable:
    """Mixin: stream access ``fetch(begin, end, key)``.

    No base class — relies on ``self._data``, ``self._cum``,
    ``self._length`` provided by :class:`Store`.  Used by SEQ/SFT
    where data is a long token stream.
    """

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
            return self._fetch_stream_key(keys, begin, end)
        return {k: self._fetch_stream_key(k, begin, end) for k in keys}

    def _fetch_stream_key(self, key: str, begin: int, end: int) -> Tensor:
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


class Recordable:
    """Mixin: record access ``fetch_record(i, key)``.

    No base class — relies on ``self._data``, ``self._offsets``,
    ``self._num_records`` provided by :class:`Store`.  Used by
    DPO/GRPO where each record is an independent training unit.
    """

    segments_are_records = True

    @property
    def num_records(self) -> int:
        return self._num_records

    def fetch_record(
        self,
        index: int,
        keys: Union[str, List[str]],
    ):
        if not self._data and self._num_records == 0:
            raise RuntimeError("Store not loaded")
        if not 0 <= index < self._num_records:
            raise ValueError(
                f"Record index out of bounds: {index}, num_records={self._num_records}"
            )
        if isinstance(keys, str):
            return self._fetch_record_key(keys, index)
        return {k: self._fetch_record_key(k, index) for k in keys}

    def _fetch_record_key(self, key: str, index: int):
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


class StoreFactory(BaseFactory["Store"]):
    """Factory for creating Store instances by type name."""


@StoreFactory.register("h5")
class H5Store(Store, Streamable, Recordable):
    """HDF5-based storage backend (pre-tokenized data).

    Each key is stored as a group of per-record datasets (``data_0``,
    ``data_1``, …).  Supports both access modes:

    - **Stream** (``fetch(begin, end, key)``): concatenates across
      records via ``_cum`` — used by SEQ/SFT where data is a token stream.
    - **Record** (``fetch_record(i, key)``): indexes ``_data[key]``
      directly — used by DPO/GRPO where each record is independent.

    ``len(store)`` returns the **token count** (stream semantics) so
    SEQ/SFT windowing works.  Record-only code uses
    ``store.num_records`` instead.
    """

    segments_are_records = True

    def load(self, path: str, **kwargs):
        self._normalize(load_h5(path))


@StoreFactory.register("bin")
class MmapStore(Store, Streamable, Recordable):
    """Memory-mapped binary storage backend.

    Each key is a single .bin file backed by ``np.memmap(mode="r")``.
    No per-process memory duplication — all DataLoader workers share the
    same OS page-cache pages.

    Supports both access modes:

    - **Stream** (``fetch(begin, end, key)``): always available.
    - **Record** (``fetch_record(i, key)``): only when ``meta.json``
      contains per-record ``offsets`` (written via
      ``save_bin(..., record_keys=...)``).  Legacy bin files without
      offsets have ``num_records == 0``.

    ``segments_are_records`` is ``False`` here (bin segments are
    contiguous streams, not per-record) — record access is driven
    purely by ``_offsets``.
    """

    segments_are_records = False

    def load(self, path: str, **kwargs):
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


class JsonlSource:
    """Read raw JSON records from a ``.jsonl`` file or directory.

    A thin reader used by :class:`JsonlStore` in processor mode — holds
    no tokenizer, performs no tokenisation, just yields dicts.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._records: Optional[List[dict]] = None

    def load(self) -> List[dict]:
        if self._records is None:
            self._records = self._read(self.path)
        return self._records

    @staticmethod
    def _read(root: Path) -> List[dict]:
        if root.is_file():
            return JsonlSource._read_file(root)
        return JsonlSource._read_dir(root)

    @staticmethod
    def _read_file(path: Path) -> List[dict]:
        records: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON line in %s, skipping", path)
        return records

    @staticmethod
    def _read_dir(root: Path) -> List[dict]:
        records: List[dict] = []
        for jsonl_path in sorted(root.glob("*.jsonl")):
            records.extend(JsonlSource._read_file(jsonl_path))
        return records


@StoreFactory.register("jsonl")
class JsonlStore(Store, Streamable, Recordable):
    """JSONL reader with two tokenisation modes.

    A JSONL dataset is a ``.jsonl`` file or a directory of ``*.jsonl``
    files plus (optionally) a ``dataset_config.json`` describing the
    tokenization pipeline.

    Two modes, selected at :meth:`load` time:

    - **Eager** (default): applies a :class:`TokenizeTransform` to every
      record at load time and registers per-key tensors via
      ``_normalize``.  Both stream (``fetch``) and record
      (``fetch_record``) access work — stream concatenates across
      records via ``_cum``, record indexes directly.
    - **Lazy** (``processor=fn`` given): keeps raw records and defers
      tokenisation to ``fetch_record``.  Only record access works.
      Used by DPO/GRPO where each record is independent.

    ``len(store)`` returns the **token count** (stream semantics) in
    eager mode so SEQ/SFT windowing works; returns the **record count**
    in lazy mode where stream access is unavailable.
    """

    CONFIG_NAME = "dataset_config.json"
    segments_are_records = True

    def __init__(self):
        super().__init__()
        self._source: Optional[JsonlSource] = None
        self._processor: Optional[Callable[[dict], Dict[str, Tensor]]] = None
        self._keys_cache: Optional[List[str]] = None

    def __len__(self) -> int:
        if self._processor is not None:
            return self._num_records
        return self._length

    def load(self, path: str, transform=None, processor=None, **kwargs):
        self._source = JsonlSource(path)
        records = self._source.load()

        if processor is not None:
            self._processor = processor
            self._num_records = len(records)
            return

        if transform is None:
            root = Path(path)
            config_path = root / self.CONFIG_NAME if root.is_dir() else None
            if config_path is None or not config_path.exists():
                raise FileNotFoundError(
                    f"JSONL dataset config not found. Expected "
                    f"{self.CONFIG_NAME} alongside *.jsonl files, pass an "
                    f"explicit transform, or pass processor= for lazy "
                    f"on-the-fly tokenisation."
                )
            transform = TokenizeTransform.from_config_file(str(config_path))

        transformed = transform.apply(records)
        self._normalize(transformed)

    @property
    def keys(self) -> List[str]:
        if self._processor is not None:
            if self._keys_cache is None and self._num_records > 0:
                sample = self._processor(self._source.load()[0])
                self._keys_cache = list(sample.keys())
            return self._keys_cache or []
        return list(self._data.keys())

    def fetch_record(self, index: int, keys: Union[str, List[str]]):
        if self._processor is not None:
            if not 0 <= index < self._num_records:
                raise ValueError(
                    f"Record index out of bounds: {index}, "
                    f"num_records={self._num_records}"
                )
            record = self._source.load()[index]
            data = self._processor(record)
            if isinstance(keys, str):
                return data[keys]
            return {k: data[k] for k in keys}
        return super().fetch_record(index, keys)
