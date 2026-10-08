"""Common training strategy lifecycle and interfaces."""

from abc import ABC
from typing import (
    Any,
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
from torch.optim import Optimizer

from astrai.model.components.mlp import RouterStats
from astrai.parallel.cp import LossReduction, TokenLoss
from astrai.trainer.backend import WeightPublisher
from astrai.trainer.rollout import RolloutResult
from astrai.trainer.strategy.ops import (
    ForwardResult,
    LossOutput,
    _collect_moe_diagnostics,
    move_to_device,
)


class BaseStrategy(ABC):
    """Abstract base class for training strategies.

    When a :class:`~astrai.trainer.rollout.RolloutRunner` is injected via
    :meth:`set_rollout_runner`, the strategy transparently switches to
    online mode: each ``__call__`` produces a :class:`RolloutResult`,
    converts it to a training batch via :meth:`prepare_from_rollout`, and
    then computes the loss.  Without a runner the strategy runs in
    offline mode and consumes the batch directly.
    """

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
        self.rl_update_epochs = self._validate_update_epochs(
            kwargs.pop("rl_update_epochs", 1)
        )
        self.rl_minibatch_prompts = kwargs.pop("rl_minibatch_prompts", None)
        if self.rl_minibatch_prompts is not None and (
            isinstance(self.rl_minibatch_prompts, bool)
            or not isinstance(self.rl_minibatch_prompts, int)
            or self.rl_minibatch_prompts < 1
        ):
            raise ValueError("rl_minibatch_prompts must be a positive integer or None")
        self.gradient_chunked_logprobs = bool(
            kwargs.pop("gradient_chunked_logprobs", False)
        )
        self._moe_metrics: Dict[str, float] = {}
        self.strategy_kwargs = kwargs
        self._rollout_runner = None
        self._weight_publishers: Tuple[WeightPublisher, ...] = ()

    @staticmethod
    def _validate_update_epochs(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("rl_update_epochs must be a positive integer")
        return value

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
        self, loss: Tensor, reported_loss: Tensor, forward: ForwardResult
    ) -> LossOutput:
        """Assemble the LossOutput around an already-reduced token loss."""
        return self._loss_output(
            loss,
            {"task_loss": reported_loss.detach()},
            forward.aux_loss,
            forward.router_stats,
        )

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute loss for the given batch.

        Args:
            batch: Dictionary containing batch tensors

        Returns:
            Computed loss tensor
        """
        return self.compute_loss_output(batch)["loss"]

    def compute_loss_output(self, batch: Dict[str, Tensor]) -> LossOutput:
        if type(self).forward_tokens is BaseStrategy.forward_tokens:
            # Legacy contract: a strategy overriding only compute_loss
            # (returning a tensor) still gets a normalized LossOutput.
            return self._normalize_output(self.compute_loss(batch))
        batch = self.prepare_batch(batch)
        forward = self.forward_tokens(batch)
        tokens = self.reduce_loss(forward, batch)
        return self.token_loss_output(tokens.mean(), tokens.mean(), forward)

    def validate_online(self, batch: Dict[str, Any]) -> Optional[LossOutput]:
        """Validate one batch through a one-off rollout.

        Online strategies with an injected rollout runner evaluate a
        fresh, throw-away rollout so the training replay cache and its
        cadence stay untouched. Returns ``None`` when no runner is
        configured (offline mode); callers then fall back to
        ``strategy(batch)``.
        """
        if self._rollout_runner is None:
            return None
        result = self._rollout_runner.evaluate(batch)
        prepared = self.prepare_from_rollout(result)
        return self.compute_loss_output(prepared)

    def _loss_output(
        self,
        task_loss: Tensor,
        metrics: Dict[str, Tensor],
        aux_loss: Optional[Tensor] = None,
        router_stats: Optional[List[RouterStats]] = None,
    ) -> LossOutput:
        total_loss = task_loss
        if aux_loss is not None:
            weighted_aux_loss = self.moe_aux_loss_coef * aux_loss
            total_loss = total_loss + weighted_aux_loss
            metrics["moe_aux_loss"] = aux_loss
            metrics["moe_aux_loss_weighted"] = weighted_aux_loss
            self._refresh_moe_diagnostics(aux_loss, router_stats)
        metrics["loss"] = total_loss
        return {
            "loss": total_loss,
            "metrics": {name: value.detach().item() for name, value in metrics.items()},
        }

    @staticmethod
    def _normalize_output(output: Union[LossOutput, Tensor]) -> LossOutput:
        if isinstance(output, dict):
            return output
        return {"loss": output, "metrics": {"loss": output.detach().item()}}

    def supports_online(self) -> bool:
        """Whether this strategy can operate with a rollout runner.

        Base implementation returns ``False``; strategies that implement
        :meth:`prepare_from_rollout` should override to return ``True``.
        """
        return False

    def set_rollout_runner(self, runner):
        """Inject a :class:`RolloutRunner` to enable online rollout mode."""
        self._rollout_runner = runner

    @property
    def policy_version(self) -> Optional[int]:
        if self._rollout_runner is None:
            return None
        return self._rollout_runner.policy_version

    def prepare_from_rollout(self, result: RolloutResult) -> Dict[str, Tensor]:
        """Map a :class:`RolloutResult` to the batch layout expected by
        :meth:`compute_loss`.

        Strategies that return ``True`` from :meth:`supports_online` must
        override this.  Default raises :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support online rollout"
        )

    def _on_rollout_refresh(self):
        """Hook fired when a fresh rollout result is produced.

        Override to refresh stale state (e.g. syncing the behaviour
        policy).  Default is a no-op.
        """
        pass

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

    def on_optimizer_step(self):
        """Reject unsafe post-hoc publication for an online shared model."""
        if self._rollout_runner is not None:
            raise RuntimeError(
                "online training must call strategy.optimizer_step(optimizer) "
                "so weight mutation and policy-version publication are atomic"
            )

    def optimizer_step(self, optimizer: Optimizer):
        """Step the optimizer at an atomic online-rollout version boundary."""
        if self._rollout_runner is None:
            return optimizer.step()

        def commit(policy_version: int):
            result = optimizer.step()
            # Publishers run inside the version lock so a backend can
            # never observe new-version weights that are still stale.
            for publisher in self._weight_publishers:
                publisher.publish(policy_version, self.model)
            return result

        # None lets the scheduler derive live+1 under the policy lock,
        # avoiding a read-compute-write race on policy_version.
        result = self._rollout_runner.apply_weight_update(None, commit)
        self._rollout_runner.step()
        return result

    def set_weight_publishers(self, publishers) -> None:
        """Inject :class:`~astrai.trainer.rollout.WeightPublisher` instances.

        Each publisher is invoked inside the policy-version lock on every
        online optimizer step, after ``optimizer.step()`` and before the
        version commit — the seam where rollout-backend replicas receive
        fresh weights atomically with the trainer's own publication.
        """
        self._weight_publishers = tuple(publishers)

    def __call__(self, batch: Dict[str, Tensor]) -> LossOutput:
        """Run offline or online forward depending on runner injection."""
        if self._rollout_runner is None:
            return self.compute_loss_output(batch)

        result, is_fresh = self._rollout_runner(batch)
        if is_fresh:
            self._on_rollout_refresh()

        train_batch = self.prepare_from_rollout(result)
        return self.compute_loss_output(train_batch)

    def training_steps(self, batch: Dict[str, Tensor]) -> Iterator[LossOutput]:
        """Yield one :class:`LossOutput` per learner update for this batch.

        The trainer iterates this instead of calling the strategy once, so
        one incoming batch can expand into several backward/optimizer-step
        cycles.  In online mode that is the RL scheduling split — one
        rollout round is collected once under a fixed policy, then consumed
        for ``rl_update_epochs`` passes of ``rl_minibatch_prompts``-sized
        learner updates (classic PPO-style multiple epochs over one batch).
        Offline mode yields exactly once, preserving the historical
        one-call-per-batch contract.

        Each optimizer step still goes through
        :meth:`optimizer_step`, so weight publication and policy-version
        advancement happen once per learner update; note that online mode
        therefore assumes ``grad_accum_steps=1`` (accumulating >1 turns the
        minibatch updates into one merged step and inflates the runner's
        replay counter).
        """
        if self._rollout_runner is None:
            yield self.compute_loss_output(batch)
            return

        result, is_fresh = self._rollout_runner(batch)
        if is_fresh:
            self._on_rollout_refresh()
        prepared = self.prepare_from_rollout(result)
        slices = self._split_rollout_batch(prepared)
        for _ in range(self.rl_update_epochs):
            for chunk in slices:
                yield self.compute_loss_output(chunk)

    def training_updates(
        self, batch: Dict[str, Tensor]
    ) -> Iterator[Iterator[LossOutput]]:
        """Group backward outputs by their single optimizer/publication step."""
        for output in self.training_steps(batch):
            yield iter((output,))

    def _split_rollout_batch(self, prepared: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Slice one prepared rollout batch along the prompt dimension.

        Every prepared tensor's leading dim is the prompt count B, and the
        group dimension rides along inside each slice — GRPO advantages are
        group-normalized from per-group rewards, so slicing along B never
        re-mixes groups, and PPO's rollout-pinned advantages/returns are
        shared views rather than recomputed targets.
        """
        size = self.rl_minibatch_prompts
        if size is None:
            return [prepared]
        first = prepared[next(iter(prepared))]
        total = first.shape[0]
        if size >= total:
            return [prepared]
        return [
            {
                key: (
                    value[begin : begin + size]
                    if isinstance(value, Tensor) and value.shape[0] == total
                    else value
                )
                for key, value in prepared.items()
            }
            for begin in range(0, total, size)
        ]
