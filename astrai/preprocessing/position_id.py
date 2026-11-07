"""Pure position-id generation for document sequences."""

from typing import List


def generate_position_ids(sequences: List[List[int]], mode: str) -> List[int]:
    """Return no positions, per-document resets, or one continuous range.

    The result is flat. Offline document-reset positions must be attached
    per record before packing so they follow the packer's document order.
    """
    if mode == "none":
        return []
    if mode == "doc_reset":
        return [position for sequence in sequences for position in range(len(sequence))]
    if mode == "continuous":
        return list(range(sum(len(sequence) for sequence in sequences)))
    raise ValueError(f"Unknown position_ids_mode: {mode!r}")
