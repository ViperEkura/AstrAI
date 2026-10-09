"""Stable response seeds independent of worker assignment and process RNG."""

import hashlib
from typing import List


def response_seeds(
    base_seed: int, sample_cursor: int, prompts: int, group_size: int
) -> List[int]:
    return [
        int.from_bytes(
            hashlib.blake2b(
                f"{base_seed}:{sample_cursor + i}:{g}".encode(), digest_size=8
            ).digest(),
            "little",
        )
        & ((1 << 63) - 1)
        for i in range(prompts)
        for g in range(group_size)
    ]
