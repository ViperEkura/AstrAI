"""Attention layouts shared by the data pipeline and decoder backends."""

from dataclasses import dataclass
from functools import cached_property

import torch
from torch import Tensor


@dataclass(frozen=True)
class VarlenAttentionLayout:
    """Document offsets and valid row indices for padded training batches."""

    cu_seqlens: Tensor
    indices: Tensor
    max_seqlen: int
    position_ids: Tensor

    @cached_property
    def dense(self) -> Tensor:
        positions = self.position_ids
        boundaries = positions[:, 1:] <= positions[:, :-1]
        doc_ids = torch.cat(
            [torch.zeros_like(positions[:, :1]), boundaries.long().cumsum(dim=1)],
            dim=1,
        )
        same_doc = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)
        causal = torch.ones(
            positions.size(1),
            positions.size(1),
            dtype=torch.bool,
            device=positions.device,
        ).tril()
        return (same_doc & causal).unsqueeze(1)
