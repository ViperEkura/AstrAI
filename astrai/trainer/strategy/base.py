"""Common training strategy lifecycle and interfaces."""

from abc import ABC
from typing import (
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch.nn as nn
from torch import Tensor

from astrai.model.components.mlp import RouterStats
from astrai.parallel.cp import LossReduction, TokenLoss
from astrai.trainer.rollout import RolloutResult
from astrai.trainer.strategy.ops import (
    ForwardResult,
    LossOutput,
    _collect_moe_diagnostics,
    move_to_device,
)


class BaseStrategy(ABC):
    """Batch conversion, loss semantics, and algorithm-specific post-update work."""

    #: Declared loss reduction (see :class:`astrai.parallel.cp.LossReduction`).
    #: Token-mean strategies compose with
    #: :class:`astrai.parallel.cp.CPStrategy`; sequence-level strategies
    #: do not shard and inherit the SEQUENCE default.
    loss_reduction: ClassVar[LossReduction] = LossReduction.SEQUENCE

    def __init__(
        self,
        model: Union[nn.Module, Callable[..., Dict[str, Tensor]]],
        device: str,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.executor = kwargs.pop("executor", None)
        self.moe_aux_loss_coef = kwargs.pop("moe_aux_loss_coef", 0.01)
        self.gradient_chunked_logprobs = bool(
            kwargs.pop("gradient_chunked_logprobs", False)
        )
        self._moe_metrics: Dict[str, float] = {}
        self.strategy_kwargs = kwargs

    # ---------- token-mean two-phase protocol ----------
    # CP composes between the phases: astrai.parallel.cp.CPStrategy shards
    # the batch buffers after prepare_batch, runs forward_tokens and
    # reduce_loss on the local slice, and rescales the reduction.  A
    # strategy that declares LossReduction.TOKEN_MEAN implements these
    # instead of compute_loss_output, and its code runs identically on
    # full sequences and on cp shards.  Sequence-level strategies (dpo,
    # grpo, ...) keep overriding compute_loss_output wholesale.

    def prepare_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Place the batch on device and synthesize missing inputs."""
        return move_to_device(batch, self.device)

    def shard_spec(self, batch: Dict[str, Tensor]) -> Tuple[List[Tensor], List[int]]:
        """Buffers to shard along the sequence dimension, with their dims.

        Called only by :class:`~astrai.parallel.cp.CPStrategy`.  The base
        strategy declares no shard surface.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares no context-parallel shard surface"
        )

    def forward_tokens(self, batch: Dict[str, Tensor]) -> ForwardResult:
        """Run the model over the (possibly sharded) sequence."""
        raise NotImplementedError(f"{type(self).__name__} implements no token forward")

    def reduce_loss(
        self, forward: ForwardResult, batch: Dict[str, Tensor]
    ) -> TokenLoss:
        """Local token reduction: loss sum plus contributing token count."""
        raise NotImplementedError(
            f"{type(self).__name__} implements no token reduction"
        )

    def token_loss_output(
        self,
        loss: Tensor,
        reported_loss: Tensor,
        forward: ForwardResult,
        normalizer: Optional[float] = None,
    ) -> LossOutput:
        """Assemble the LossOutput around an already-reduced token loss."""
        return self._loss_output(
            loss,
            {"task_loss": reported_loss.detach()},
            forward.aux_loss,
            forward.router_stats,
            normalizer=normalizer,
        )

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Scalar view of :meth:`compute_loss_output`, for callers that need one."""
        return self.compute_loss_output(batch)["loss"]

    def compute_loss_output(self, batch: Dict[str, Tensor]) -> LossOutput:
        batch = self.prepare_batch(batch)
        forward = self.forward_tokens(batch)
        tokens = self.reduce_loss(forward, batch)
        return self.token_loss_output(
            tokens.mean(),
            tokens.mean(),
            forward,
            normalizer=float(tokens.token_count.item()),
        )

    def _loss_output(
        self,
        task_loss: Tensor,
        metrics: Dict[str, Tensor],
        aux_loss: Optional[Tensor] = None,
        router_stats: Optional[List[RouterStats]] = None,
        normalizer: Optional[float] = None,
    ) -> LossOutput:
        total_loss = task_loss
        if aux_loss is not None:
            weighted_aux_loss = self.moe_aux_loss_coef * aux_loss
            total_loss = total_loss + weighted_aux_loss
            metrics["moe_aux_loss"] = aux_loss
            metrics["moe_aux_loss_weighted"] = weighted_aux_loss
            self._refresh_moe_diagnostics(aux_loss, router_stats)
        metrics["loss"] = total_loss
        output = {
            "loss": total_loss,
            "metrics": {name: value.detach().item() for name, value in metrics.items()},
        }
        if normalizer is not None:
            output["normalizer"] = normalizer
        return output

    def supports_online(self) -> bool:
        """Whether this strategy can operate with a rollout runner.

        Base implementation returns ``False``; strategies that implement
        :meth:`prepare_from_rollout` should override to return ``True``.
        """
        return False

    def prepare_from_rollout(self, result: RolloutResult) -> Dict[str, Tensor]:
        """Map a :class:`RolloutResult` to the batch layout expected by
        :meth:`compute_loss`.

        Strategies that return ``True`` from :meth:`supports_online` must
        override this.  Default raises :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support online rollout"
        )

    def _refresh_moe_diagnostics(
        self,
        aux_loss: Tensor,
        router_stats: Optional[List[RouterStats]] = None,
    ) -> None:
        """Collect MoE routing diagnostics from the latest forward pass.

        Populates ``self._moe_metrics`` with router entropy, dead expert
        fraction, load imbalance, and aux_loss.  Called from
        :meth:`_loss_output` when an MoE aux loss is present.
        """
        self._moe_metrics = _collect_moe_diagnostics(router_stats or [])
        self._moe_metrics["aux_loss"] = float(aux_loss.detach().cpu().item())

    def __call__(self, batch: Dict[str, Tensor]) -> LossOutput:
        return self.compute_loss_output(batch)

    def loss_outputs(
        self, batch: Dict[str, Tensor], microbatch_prompts: Optional[int] = None
    ) -> Iterator[LossOutput]:
        """Yield losses whose gradients belong to one optimizer update."""
        yield self.compute_loss_output(batch)

    def after_update(self) -> None:
        """Update algorithm state after the actor's weight publication."""
