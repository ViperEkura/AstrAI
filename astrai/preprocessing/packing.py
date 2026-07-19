"""Sequence packing strategies for shard-level reordering and truncation.

Each strategy receives the accumulated ``{key: [list of token lists]}``
dict for a shard and returns a reordered / truncated version.  The
pipeline later flattens the result into contiguous tensors.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from astrai.factory import BaseFactory


def _truncate(seq: List[int], max_len: int, mode: str) -> List[int]:
    if len(seq) <= max_len:
        return seq
    if mode == "keep_end":
        return seq[-max_len:]
    return seq[:max_len]


def plan_bfd(
    sequences: List[List[int]], max_packed_len: int, truncation_mode: str = "keep_start"
) -> List[List[int]]:
    """Best-Fit Decreasing bin packing of *sequences* into bins.

    Returns a list of bins, each bin a list of original indices into
    *sequences*.  Bin capacities are respected on the *truncated*
    length of each sequence (so a sequence longer than
    *max_packed_len* counts at *max_packed_len*).

    Pure index-based so callers can apply the same plan to any
    aligned key (``loss_mask``, ``position_ids``…).
    """
    n = len(sequences)
    order = sorted(range(n), key=lambda i: len(sequences[i]), reverse=True)
    bins: List[List[int]] = []
    bin_lengths: List[int] = []

    for orig_idx in order:
        seq_len = len(_truncate(sequences[orig_idx], max_packed_len, truncation_mode))
        best_bin = None
        best_remain = max_packed_len + 1
        for i, bl in enumerate(bin_lengths):
            remain = max_packed_len - bl
            if seq_len <= remain < best_remain:
                best_remain = remain
                best_bin = i
        if best_bin is not None:
            bins[best_bin].append(orig_idx)
            bin_lengths[best_bin] += seq_len
        else:
            bins.append([orig_idx])
            bin_lengths.append(seq_len)

    return bins


class PackingStrategy(ABC):
    """Reorder and truncate sequences within a shard."""

    @abstractmethod
    def apply(
        self,
        keys: Dict[str, List[List[int]]],
        max_packed_len: int,
        truncation_mode: str,
    ) -> Dict[str, List[List[int]]]:
        raise NotImplementedError


class PackingStrategyFactory(BaseFactory["PackingStrategy"]):
    pass


@PackingStrategyFactory.register("simple")
class SimplePacking(PackingStrategy):
    def apply(
        self,
        keys: Dict[str, List[List[int]]],
        max_packed_len: int,
        truncation_mode: str,
    ) -> Dict[str, List[List[int]]]:
        return {
            k: [_truncate(v, max_packed_len, truncation_mode) for v in vals]
            for k, vals in keys.items()
        }


@PackingStrategyFactory.register("bfd")
class BFDPacking(PackingStrategy):
    """Best-Fit Decreasing bin packing.

    Assigns sequences to bins using a best-fit heuristic (sorted by
    decreasing length) and concatenates sequences within each bin into
    a single packed sequence.  Packed sequences are truncated to
    *max_packed_len* so that each packed bin fits within one context
    window during training.
    """

    def apply(
        self,
        keys: Dict[str, List[List[int]]],
        max_packed_len: int,
        truncation_mode: str,
    ) -> Dict[str, List[List[int]]]:
        sequences = keys.get("sequence", [])
        if not sequences:
            return keys
        bins = plan_bfd(sequences, max_packed_len, truncation_mode)

        packed: Dict[str, List[List[int]]] = {}
        for k, vals in keys.items():
            packed[k] = [
                _truncate(
                    self._concat_bin(vals, bin_indices),
                    max_packed_len,
                    truncation_mode,
                )
                for bin_indices in bins
            ]
        return packed

    @staticmethod
    def _concat_bin(vals: List[List[int]], indices: List[int]) -> List[int]:
        result: List[int] = []
        for i in indices:
            result.extend(vals[i])
        return result


@PackingStrategyFactory.register("bfd_split")
class BFDSplitPacking(BFDPacking):
    """BFD packing with over-length sequences split into chunks.

    Sequences longer than *max_packed_len* are split into consecutive
    chunks of at most *max_packed_len* tokens instead of being
    truncated.  Each chunk becomes an independent sequence that enters
    BFD planning.  All keys (``loss_mask``, ``position_ids``, …) are
    split in lockstep so per-token alignment is preserved.

    Note: because each chunk is treated as a separate document, the
    second chunk of a split sequence loses the preceding context.
    """

    def apply(
        self,
        keys: Dict[str, List[List[int]]],
        max_packed_len: int,
        truncation_mode: str,
    ) -> Dict[str, List[List[int]]]:
        sequences = keys.get("sequence", [])
        if not sequences:
            return keys
        if max_packed_len <= 0:
            return super().apply(keys, max_packed_len, truncation_mode)

        split_keys = self._split_all(keys, max_packed_len)
        return super().apply(split_keys, max_packed_len, truncation_mode)

    @staticmethod
    def _split_all(
        keys: Dict[str, List[List[int]]], max_packed_len: int
    ) -> Dict[str, List[List[int]]]:
        """Split every sequence exceeding *max_packed_len* into chunks,
        applying the same chunk boundaries to all keys."""
        sequences = keys["sequence"]
        chunk_bounds = [list(range(0, len(s), max_packed_len)) for s in sequences]
        result: Dict[str, List[List[int]]] = {}
        for key, vals in keys.items():
            split_vals: List[List[int]] = []
            for val, starts in zip(vals, chunk_bounds):
                for start in starts:
                    split_vals.append(val[start : start + max_packed_len])
            result[key] = split_vals
        return result
