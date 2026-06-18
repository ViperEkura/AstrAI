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
        bins = self._plan(sequences, max_packed_len, truncation_mode)

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

    @staticmethod
    def _plan(
        sequences: List[List[int]], max_packed_len: int, truncation_mode: str
    ) -> List[List[int]]:
        n = len(sequences)
        order = sorted(range(n), key=lambda i: len(sequences[i]), reverse=True)
        bins: List[List[int]] = []
        bin_lengths: List[int] = []

        for orig_idx in order:
            seq_len = len(
                _truncate(sequences[orig_idx], max_packed_len, truncation_mode)
            )
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
