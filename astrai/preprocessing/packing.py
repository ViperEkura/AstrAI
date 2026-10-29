"""Independent overflow handling and length-only BFD planning.

All token-aligned fields follow the same slices and bin plan. No padding,
label shifting, or model-specific attention logic belongs in this module.
"""


def plan_bfd(lengths: list[int], capacity: int) -> list[list[int]]:
    """Best-fit decreasing: return bins of input indices, without truncation.

    A bitset finds the smallest available capacity that fits each item.
    Bins are returned in first-source-index order to avoid a length curriculum.
    """
    if capacity <= 0 or any(n <= 0 or n > capacity for n in lengths):
        raise ValueError("BFD lengths must be positive and fit the positive capacity")
    bins = []
    by_remaining = [[] for _ in range(capacity + 1)]
    available = 0
    for index in sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True):
        length = lengths[index]
        candidates = available >> length
        if candidates:
            remaining = length + (candidates & -candidates).bit_length() - 1
            bin_id = by_remaining[remaining].pop()
            if not by_remaining[remaining]:
                available &= ~(1 << remaining)
        else:
            bin_id = len(bins)
            bins.append([])
            remaining = capacity
        bins[bin_id].append(index)
        remaining -= length
        if remaining:
            by_remaining[remaining].append(bin_id)
            available |= 1 << remaining
    return sorted(bins, key=min)


def pack_sequences(
    keys: dict[str, list[list[int]]],
    max_seq_len: int,
    max_packed_len: int,
    *,
    overflow: str = "truncate",
    packing: str = "bfd",
    truncation_mode: str = "keep_start",
) -> dict[str, list[list[int]]]:
    """Normalize lengths once, then assemble bins and their segment lengths.

    Split chunks are independent segments and do not inherit earlier context.
    Multi-output records (without a sequence key) are not packed or split.
    """
    if overflow not in {"truncate", "split"} or packing not in {"bfd", "none"}:
        raise ValueError("Expected overflow=truncate|split and packing=bfd|none")
    if truncation_mode not in {"keep_start", "keep_end"}:
        raise ValueError("Expected truncation_mode=keep_start|keep_end")
    limit = min(max_seq_len, max_packed_len)
    if limit <= 0:
        raise ValueError("Sequence and packed lengths must be positive")
    if "sequence" not in keys:
        return keys
    if "seq_lengths" in keys:
        raise ValueError("seq_lengths is generated after packing")
    sequences = keys["sequence"]
    for name, rows in keys.items():
        if len(rows) != len(sequences) or any(
            len(row) != len(seq) for row, seq in zip(rows, sequences)
        ):
            raise ValueError(f"Packing field {name!r} is not token-aligned")
    chunks = {key: [] for key in keys}
    for index, seq in enumerate(sequences):
        if not seq:
            continue
        if overflow == "split":
            starts = range(0, len(seq), limit)
        else:
            starts = [max(0, len(seq) - limit) if truncation_mode == "keep_end" else 0]
        for start in starts:
            for name, rows in keys.items():
                chunk = rows[index][start : start + limit]
                chunks[name].append(
                    list(range(len(chunk))) if name == "position_ids" else chunk
                )
    lengths = [len(seq) for seq in chunks["sequence"]]
    bins = (
        plan_bfd(lengths, max_packed_len)
        if packing == "bfd"
        else [[index] for index in range(len(lengths))]
    )
    packed = {
        key: [[token for index in indices for token in rows[index]] for indices in bins]
        for key, rows in chunks.items()
    }
    packed["seq_lengths"] = [[lengths[index] for index in indices] for indices in bins]
    return packed
