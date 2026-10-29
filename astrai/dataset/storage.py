"""Storage backends for different data formats.

Architecture (composition over inheritance):

    Store (ABC)               — owns _data/_cum/_offsets bookkeeping
                                  + window_size/stride for sample-id
                                  indexing.  __getitem__/__len__ produce
                                  the smallest iterable unit so Dataset
                                  classes are pure delegators.
    Streamable (mixin)        — raw token slice  fetch(begin, end, keys)
    Recordable (mixin)        — raw record slice fetch_record(idx, keys)

    MmapStore(Store, Streamable, Recordable)
    JsonlStore(Store, Streamable, Recordable)

Each mixin is a stateless trait that relies on ``self._data`` etc.
provided by :class:`Store`.  Concrete stores mix in whichever access
primitives they support — ``Store`` is the sole base class, so there is
no diamond inheritance or MRO ambiguity.

Sample-id indexing lives on :class:`Store`, not on the dataset:

- **Stream mode** (``window_size > 0``):  ``len(store)`` returns the number
  of ``(window_size, stride)`` windows that fit in the token river;
  ``store[i]`` returns the *i*-th window as a dict of per-key tensors;
  ``store.sample_window(i)`` exposes the underlying ``(begin, end)``
  token slice for callers (e.g. next-token trainers) that need a +1
  shifted companion window.
- **Record mode** (``num_records > 0``):  ``len(store)`` returns the
  record count; ``store[i]`` returns the *i*-th record dict.

Raw token/record access via :meth:`fetch` / :meth:`fetch_record`
remains available for low-level callers that want explicit index
control.  ``store.token_count`` is the total stream token count (what
``len(store)`` used to mean in the legacy stream-only API).

``segments_are_records`` (class attribute on each Store subclass)
tells ``_normalize`` whether segments are inherently per-record (JSONL)
or opaque shards (bin).  Record access for bin relies on ``_offsets``
instead.

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
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from astrai.factory import BaseFactory
from astrai.serialization import (
    load_bin,
    load_bin_offsets,
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
        if suffix == ".jsonl":
            return "jsonl"
        raise ValueError(f"Unsupported file format: {suffix}")

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


def _stream_fetch(self, begin: int, end: int, keys: Union[str, List[str]]):
    if not getattr(self, "_data", None):
        raise RuntimeError("Store not loaded")
    if not (0 <= begin < self._length and 0 <= end <= self._length):
        raise ValueError(
            f"Index out of bounds: begin={begin}, end={end}, length={self._length}"
        )
    if isinstance(keys, str):
        return _fetch_stream_key(self, keys, begin, end)
    return {k: _fetch_stream_key(self, k, begin, end) for k in keys}


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


def _record_fetch(self, index: int, keys: Union[str, List[str]]):
    num_records = self.num_records
    if not getattr(self, "_data", None) and num_records == 0:
        raise RuntimeError("Store not loaded")
    if not 0 <= index < num_records:
        raise ValueError(
            f"Record index out of bounds: {index}, num_records={num_records}"
        )
    if isinstance(keys, str):
        return _fetch_record_key(self, keys, index)
    return {k: _fetch_record_key(self, k, index) for k in keys}


def _fetch_record_key(self, key: str, index: int):
    offsets = self._offsets.get(key)
    if offsets:
        shard_index = bisect.bisect_right(self._record_cum, index)
        local_index = index - (self._record_cum[shard_index - 1] if shard_index else 0)
        start, end = offsets[shard_index][local_index : local_index + 2]
        return self._data[key][shard_index][start:end]
    if self._offsets:
        raise ValueError(f"Record field '{key}' requires offsets")
    return self._data[key][index]


def _read_jsonl(root: Path) -> List[Dict]:
    """Read raw records in file order, skipping empty or malformed lines."""
    records: List[Dict] = []
    paths = [root] if root.is_file() else sorted(root.glob("*.jsonl"))
    for path in paths:
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


class Store(ABC):
    """Common base for all storage backends.

    A Store owns both its data layout AND its sample-id → token/record
    index translation.  Datasets are thin wrappers that bind a Store
    to a particular train-type's key mapping; they never know about
    window/stride math.

    Two iteration modes:

    - **Stream** (``window_size > 0``): data is treated as one long
      token river.  ``len(store)`` returns the number of windows;
      ``store[i]`` slices every stream-compatible key to window ``i``;
      ``store.sample_window(i)`` returns the ``(begin, end)`` token
      slice for callers needing a +1 shifted companion window.
    - **Record** (``num_records > 0``): data is per-record.
      ``len(store)`` returns ``num_records``; ``store[i]`` returns
      the *i*-th record as a dict.

    Raw token slicing is still available via :meth:`fetch` (mixed in
    by :class:`Streamable`) when a store has stream support configured.
    Raw record slicing via :meth:`fetch_record` (mixed in by
    :class:`Recordable`) when a store has record support.

    ``token_count`` exposes the raw total stream length — this is what
    ``len(store)`` returned in the legacy stream-only API and what
    stream-bound ``fetch`` uses for its bounds check.
    """

    segments_are_records: bool = False

    def __init__(
        self,
        window_size: int = 0,
        stride: Optional[int] = None,
    ):
        self._data: Dict[str, List[Tensor]] = {}
        self._cum: Dict[str, List[int]] = {}
        self._offsets: Dict[str, List[List[int]]] = {}
        self._record_cum: List[int] = []
        self._length: int = 0
        self._num_records: int = 0
        self._window_size: int = int(window_size)
        self._stride: int = int(stride) if stride is not None else int(window_size)

    @abstractmethod
    def load(self, path: str, **kwargs) -> None:
        raise NotImplementedError

    @property
    def keys(self) -> List[str]:
        return list(self._data.keys())

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def token_count(self) -> int:
        """Total tokens across all stream segments.

        Useful for the bounds-checked raw :meth:`fetch` and as the
        legacy ``len(store)`` value.
        """
        return self._length

    @property
    def num_records(self) -> int:
        """Number of records available via :meth:`fetch_record`.

        Non-zero only when the backing layout provides per-record
        indexing (JSONL segments or bin ``_offsets``).
        """
        return self._num_records

    @property
    def num_samples(self) -> int:
        """Number of items produced by ``__getitem__``.

        Stream-mode wins when ``window_size > 0`` and there are tokens
        to slice; otherwise falls back to ``num_records``.
        """
        if self._window_size > 0 and self._length > 0:
            total = self._length
            w = self._window_size
            if total <= w:
                return 0
            return (total - 1 - w) // self._stride + 1
        return self.num_records

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        if index < 0:
            index += self.num_samples
        if not 0 <= index < self.num_samples:
            raise IndexError(
                f"Store index out of range: {index}, num_samples={self.num_samples}"
            )
        if self._window_size > 0 and self._length > 0:
            begin, end = self.sample_window(index)
            keys = self._stream_keys()
            return {k: self.fetch(begin, end, k) for k in keys}
        return self.fetch_record(index, self._record_keys())

    def sample_window(self, index: int) -> Tuple[int, int]:
        """Return ``(begin, end)`` token positions for stream sample *index*.

        The clipped tail keeps the last reachable window inside the
        token river instead of overshooting.  Caller is responsible
        for staying within :attr:`num_samples`: an out-of-range index
        raises ``IndexError``.
        """
        if self._window_size <= 0:
            raise RuntimeError("sample_window() requires window_size > 0 (stream mode)")
        if self._length <= self._window_size:
            raise IndexError(
                f"Data too short for window: token_count={self._length}, "
                f"window_size={self._window_size}"
            )
        if not 0 <= index < self.num_samples:
            raise IndexError(
                f"Sample index out of range: {index}, num_samples={self.num_samples}"
            )
        total = self._length
        begin = min(index * self._stride, total - 1 - self._window_size)
        end = min(begin + self._window_size, total - 1)
        return begin, end

    def _stream_keys(self) -> List[str]:
        out: List[str] = []
        for k, tensors in self._data.items():
            if tensors and isinstance(tensors[0], list):
                continue
            out.append(k)
        return out

    def _record_keys(self) -> List[str]:
        return list(self._data.keys())

    def _normalize(
        self,
        raw: Dict[str, List],
    ):
        """Register segments and pre-compute indices for both access modes.

        Stream mode: ``_cum[key]`` accumulates per-segment lengths so
        ``Streamable._fetch_stream_key`` can bisect across segments
        without concatenation.

        Record mode: when ``segments_are_records`` is True (JSONL),
        ``_data[key]`` is a per-record list and ``fetch_record`` indexes
        it directly. Binary offsets are indexed lazily by MmapStore.

        Nested keys (GRPO ``responses``/``masks`` as
        ``List[List[Tensor]]``) are stored as-is and excluded from both
        cumulative bookkeepings — they are only accessed record-by-record.
        """
        self._data = {}
        self._cum = {}
        self._offsets = {}
        self._record_cum = []
        self._num_records = 0
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
            if key != "seq_lengths":
                flat_lengths.append(cum[-1] if cum else 0)
        self._length = min(flat_lengths) if flat_lengths else 0

        if self.segments_are_records:
            per_record_counts = []
            for key, tensors in self._data.items():
                if tensors and isinstance(tensors[0], list):
                    continue
                per_record_counts.append(len(tensors))
            self._num_records = min(per_record_counts) if per_record_counts else 0
        else:
            self._num_records = 0

    def _build_record_index(self, shards: List[Dict[str, List[int]]]) -> None:
        keys = {key for shard in shards for key in shard}
        indexed: Dict[str, List[List[int]]] = {key: [] for key in sorted(keys)}
        cumulative = []
        total = 0
        for shard_index, shard in enumerate(shards):
            if set(shard) != keys:
                raise ValueError(
                    f"Shard {shard_index} has inconsistent record offsets fields"
                )
            counts = set()
            for key in indexed:
                segments = self._data.get(key, [])
                if len(segments) != len(shards):
                    raise ValueError(f"Record field '{key}' is missing from a shard")
                offsets = shard[key]
                if (
                    not isinstance(offsets, list)
                    or len(offsets) < 2
                    or any(type(offset) is not int for offset in offsets)
                    or offsets[0] != 0
                    or offsets[-1] != segments[shard_index].shape[0]
                    or any(begin > end for begin, end in zip(offsets, offsets[1:]))
                ):
                    raise ValueError(
                        f"Invalid offsets for '{key}' in shard {shard_index}"
                    )
                indexed[key].append(offsets)
                counts.add(len(offsets) - 1)
            if len(counts) != 1:
                raise ValueError(f"Inconsistent record count in shard {shard_index}")
            total += counts.pop()
            cumulative.append(total)
        self._offsets = indexed
        self._record_cum = cumulative
        self._num_records = total


class Streamable:
    """Mixin granting raw token-stream access via :meth:`fetch`.

    Stateless trait relying on ``self._data``, ``self._cum``,
    ``self._length`` maintained by :class:`Store`.  Stream mode is
    active when the owning store has ``window_size > 0``; for stores
    that can also serve record access (JSONL/bin+offsets), the
    ``fetch_record`` API from :class:`Recordable` is used instead.
    """

    def fetch(
        self,
        begin: int,
        end: int,
        keys: Union[str, List[str]],
    ):
        return _stream_fetch(self, begin, end, keys)


class Recordable:
    """Mixin granting raw record access via :meth:`fetch_record`.

    Stateless trait relying on ``self._data``, ``self._offsets``,
    ``self._num_records`` maintained by :class:`Store`.
    """

    def fetch_record(
        self,
        index: int,
        keys: Union[str, List[str]],
    ):
        return _record_fetch(self, index, keys)


class StoreFactory(BaseFactory["Store"]):
    """Factory for creating Store instances by type name."""


@StoreFactory.register("bin")
class MmapStore(Store, Streamable, Recordable):
    """Memory-mapped binary storage backend.

    Each key is a single .bin file backed by ``np.memmap(mode="r")``.
    No per-process memory duplication — all DataLoader workers share the
    same OS page-cache pages.

    Supports both access modes:

    - **Stream**: always available via :meth:`fetch`.
    - **Record** (``fetch_record(i, key)``): only when ``meta.json``
      contains per-record ``offsets`` (written via
      ``save_bin(..., record_keys=...)``).  Legacy bin files without
      offsets have ``num_records == 0`` and ``len(store)`` reflects the
      windowed sample count when ``window_size > 0``.

    Record offsets are validated lazily on record access or count queries.
    Token-stream access does not depend on the validity of record metadata.

    ``segments_are_records`` is ``False`` here (bin segments are
    contiguous streams, not per-record) — record access is driven
    purely by ``_offsets``.
    """

    segments_are_records = False

    def __init__(
        self,
        window_size: int = 0,
        stride: Optional[int] = None,
    ):
        super().__init__(window_size=window_size, stride=stride)
        self._mmap_refs: List[Tensor] = []
        self._record_shards: List[Dict[str, List[int]]] = []
        self._record_index_ready = False

    @property
    def num_records(self) -> int:
        self._ensure_record_index()
        return self._num_records

    def _ensure_record_index(self) -> None:
        if self._record_index_ready:
            return
        if any(self._record_shards):
            self._build_record_index(self._record_shards)
        self._record_index_ready = True

    def load(self, path: str, **kwargs):
        self._mmap_refs = []
        root = Path(path)
        all_raw: Dict[str, List[Tensor]] = {}
        all_offsets: List[Dict[str, List[int]]] = []
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
            all_offsets.append(off)
        if not meta_paths:
            raise FileNotFoundError(f"No meta.json found under {path}")
        self._normalize(all_raw)
        self._record_shards = all_offsets
        self._record_index_ready = False
        for tensors in self._data.values():
            self._mmap_refs.extend(tensors)

    def validate_record_alignment(self, keys: List[str]) -> None:
        self._ensure_record_index()
        for key in keys:
            if key not in self._offsets:
                raise ValueError(
                    f"Binary record field '{key}' requires offsets; preprocess the data again"
                )
        for key in keys[1:]:
            if self._offsets[key] != self._offsets[keys[0]]:
                raise ValueError(f"Record lengths for '{key}' must match '{keys[0]}'")


@StoreFactory.register("jsonl")
class JsonlStore(Store, Streamable, Recordable):
    """JSONL reader with eager/lazy tokenisation modes.

    A JSONL dataset is a ``.jsonl`` file or a directory of ``*.jsonl``
    files plus (optionally) a ``dataset_config.json`` describing the
    tokenization pipeline.

    Three ways to supply an eager transform (first match wins):

    - **Explicit** (``transform=``): caller-built
      :class:`TokenizeTransform` applied eagerly.
    - **Config file**: ``dataset_config.json`` alongside the ``*.jsonl``
      files — loaded via :meth:`TokenizeTransform.from_config_file`.
    - **Default messages** (``tokenizer_path=`` given, no config file):
      a built-in chatml config that tokenises the ``messages`` field,
      masking every role except ``assistant`` (loss on assistant only).
      Lets SFT/SEQ train straight from a chat-style JSONL directory
      without a hand-written config.

    Two tokenisation modes, selected at :meth:`load` time:

    - **Eager** (default): applies the transform to every record at load
      time and registers per-key tensors via ``_normalize``.  Both
      ``fetch`` (stream) and ``fetch_record`` (record) work.
    - **Lazy** (``processor=fn`` passed): keeps raw records and defers
      tokenisation to ``fetch_record``.  Only record access works —
      ``len(store)`` returns ``num_records``; stream primitives raise.
    """

    segments_are_records = True

    def __init__(
        self,
        window_size: int = 0,
        stride: Optional[int] = None,
    ):
        super().__init__(window_size=window_size, stride=stride)
        self._records: List[Dict] = []
        self._processor: Optional[Callable[[Dict], Dict[str, Tensor]]] = None
        self._keys_cache: Optional[List[str]] = None

    def load(self, path: str, transform=None, processor=None, **kwargs):
        self._records = _read_jsonl(Path(path))

        if processor is not None:
            self._processor = processor
            self._num_records = len(self._records)
            return

        if transform is None:
            raise ValueError(
                "JsonlStore eager mode requires transform=. "
                "Use DatasetFactory.load() which auto-constructs it."
            )

        transformed = transform.apply(self._records)
        self._normalize(transformed)

    @property
    def keys(self) -> List[str]:
        if self._processor is not None:
            if self._keys_cache is None and self._num_records > 0:
                sample = self._processor(self._records[0])
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
            record = self._records[index]
            data = self._processor(record)
            if isinstance(keys, str):
                return data[keys]
            return {k: data[k] for k in keys}
        return _record_fetch(self, index, keys)

    def fetch(self, begin: int, end: int, keys: Union[str, List[str]]):
        if self._processor is not None:
            raise RuntimeError(
                "JsonlStore in lazy (processor) mode does not support "
                "stream fetch(); use fetch_record() instead."
            )
        return _stream_fetch(self, begin, end, keys)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        if self._processor is not None:
            return self.fetch_record(index, self._record_keys())
        return super().__getitem__(index)
