"""Supervised sequence and SFT objectives."""

from typing import (
    Callable,
    Dict,
    List,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.tensor import DTensor

from astrai.extension.kernel.cross_entropy import is_available as ce_available
from astrai.extension.kernel.cross_entropy import linear_cross_entropy
from astrai.parallel.cp import LossReduction, TokenLoss
from astrai.trainer.strategy.base import BaseStrategy
from astrai.trainer.strategy.factory import StrategyFactory
from astrai.trainer.strategy.ops import (
    ForwardResult,
    _is_packed,
    make_doc_boundary_mask,
)


class _CEStrategy(BaseStrategy):
    """Shared explicit CE selection; torch remains the default."""

    def __init__(
        self,
        model: Union[nn.Module, Callable[..., Dict[str, Tensor]]],
        device: str,
        label_smoothing: float = 0.0,
        loss_backend: str = "torch",
        loss_chunk_size: int = 512,
        **kwargs,
    ):
        super().__init__(model, device, **kwargs)
        if loss_backend not in ("torch", "cuda_linear_ce"):
            raise ValueError("loss_backend must be torch or cuda_linear_ce")
        if (
            isinstance(loss_chunk_size, bool)
            or not isinstance(loss_chunk_size, int)
            or loss_chunk_size <= 0
        ):
            raise ValueError("loss_chunk_size must be a positive integer")
        self.label_smoothing = label_smoothing
        self.loss_backend = loss_backend
        self.loss_chunk_size = loss_chunk_size

    def _forward_ce(self, targets: Tensor, **model_kwargs) -> ForwardResult:
        if self.loss_backend != "cuda_linear_ce":
            outputs = self.model(**model_kwargs)
            return ForwardResult(
                outputs["logits"],
                outputs.get("aux_loss"),
                outputs.get("router_stats"),
            )

        outputs = self.model(
            **model_kwargs, skip_lm_head=True, return_lm_head_weight=True
        )
        hidden = outputs["hidden_states"]
        if targets.shape != hidden.shape[:-1]:
            raise ValueError("targets must match the training token shape")
        weight = outputs["lm_head_weight"]
        bias = outputs.get("lm_head_bias")
        if (
            hidden.is_cuda
            and not isinstance(weight, DTensor)
            and bias is None
            and ce_available()
        ):
            loss_sum = linear_cross_entropy(
                hidden.flatten(0, 1),
                weight,
                targets.flatten(),
                label_smoothing=self.label_smoothing,
                chunk_size=self.loss_chunk_size,
            )
        else:
            # FSDP's head owns its all-gather; use its forward for DTensors.
            head = self.model
            while hasattr(head, "_orig_mod") or hasattr(head, "module"):
                head = getattr(head, "_orig_mod", None) or head.module
            logits = (
                head.lm_head(hidden)
                if isinstance(weight, DTensor)
                else F.linear(hidden, weight, bias)
            )
            loss_sum = F.cross_entropy(
                logits.flatten(0, 1).float(),
                targets.flatten(),
                label_smoothing=self.label_smoothing,
                reduction="sum",
            )
        return ForwardResult(
            None,
            outputs.get("aux_loss"),
            outputs.get("router_stats"),
            loss_sum,
        )

    def _ce_sum(self, forward: ForwardResult, targets: Tensor) -> Tensor:
        if forward.loss_sum is not None:
            return forward.loss_sum
        logits = forward.logits.flatten(0, 1)
        targets = targets.flatten()
        return F.cross_entropy(
            logits.float(),
            targets,
            label_smoothing=self.label_smoothing,
            reduction="sum",
        )


@StrategyFactory.register("seq")
class SEQStrategy(_CEStrategy):
    """Standard next-token prediction training strategy.

    Computes cross-entropy loss for next token prediction.
    Optionally adds MoE load balancing auxiliary loss.
    """

    loss_reduction = LossReduction.TOKEN_MEAN

    def prepare_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch = super().prepare_batch(batch)
        if "position_ids" not in batch:
            # Positions are explicit so the CP shard can split them with the
            # inputs: the model's default arange is local-only and would
            # drop higher ranks' RoPE offsets.
            input_ids = batch["input_ids"]
            batch["position_ids"] = (
                torch.arange(input_ids.size(1), device=input_ids.device)
                .unsqueeze(0)
                .expand(input_ids.size(0), -1)
            )
        return batch

    def shard_spec(self, batch: Dict[str, Tensor]) -> Tuple[List[Tensor], List[int]]:
        return (
            [batch["input_ids"], batch["target_ids"], batch["position_ids"]],
            [1, 1, 1],
        )

    def forward_tokens(self, batch: Dict[str, Tensor]) -> ForwardResult:
        return self._forward_ce(
            batch["target_ids"],
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
        )

    def reduce_loss(
        self, forward: ForwardResult, batch: Dict[str, Tensor]
    ) -> TokenLoss:
        target_ids = batch["target_ids"]
        loss_sum = self._ce_sum(forward, target_ids)
        token_count = torch.tensor(
            target_ids.numel(), dtype=torch.float32, device=target_ids.device
        )
        return TokenLoss(loss_sum, token_count)


@StrategyFactory.register("sft")
class SFTStrategy(_CEStrategy):
    """Supervised Fine-tuning strategy with loss masking.

    Applies cross-entropy loss only to tokens where loss_mask is True.
    Optionally adds MoE load balancing auxiliary loss.
    """

    loss_reduction = LossReduction.TOKEN_MEAN

    def shard_spec(self, batch: Dict[str, Tensor]) -> Tuple[List[Tensor], List[int]]:
        if _is_packed(batch["position_ids"]):
            raise NotImplementedError(
                "packed multi-document SFT cannot shard across cp ranks: "
                "the doc-boundary attention mask cannot ride the ring "
                "attention kernels"
            )
        return (
            [
                batch["input_ids"],
                batch["target_ids"],
                batch["position_ids"],
                batch["loss_mask"],
            ],
            [1, 1, 1, 1],
        )

    def forward_tokens(self, batch: Dict[str, Tensor]) -> ForwardResult:
        position_ids = batch["position_ids"]
        # Unpacked positions make the doc-boundary mask plain causality, so
        # pass None and take the is_causal fast path — the only form ring
        # attention accepts.  shard_spec has rejected packed documents
        # before a sharded forward gets here.
        input_mask = (
            make_doc_boundary_mask(position_ids) if _is_packed(position_ids) else None
        )
        targets = batch["target_ids"]
        if self.loss_backend == "cuda_linear_ce":
            targets = targets.masked_fill(~batch["loss_mask"], -100)
        return self._forward_ce(
            targets,
            input_ids=batch["input_ids"],
            position_ids=position_ids,
            input_mask=input_mask,
        )

    def reduce_loss(
        self, forward: ForwardResult, batch: Dict[str, Tensor]
    ) -> TokenLoss:
        ignore_index = -100
        target_ids = batch["target_ids"].masked_fill(~batch["loss_mask"], ignore_index)
        loss_sum = self._ce_sum(forward, target_ids)
        token_count = (target_ids != ignore_index).sum().to(dtype=torch.float32)
        return TokenLoss(loss_sum, token_count)
