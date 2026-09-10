"""Training strategy implementations with factory pattern."""

from abc import ABC
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from astrai.factory import BaseFactory
from astrai.model.components.mlp import RouterStats
from astrai.parallel.cp import LossReduction, TokenLoss
from astrai.parallel.executor import broadcast_state_dict
from astrai.trainer.rollout import RolloutResult


class LossOutput(TypedDict):
    loss: Tensor
    metrics: Dict[str, float]


class LogprobsOutput(TypedDict):
    logprobs: Tensor
    aux_loss: Optional[Tensor]
    router_stats: Optional[List[RouterStats]]


@dataclass
class ForwardResult:
    """Model forward over the (possibly sequence-sharded) batch.

    ``logits`` keeps the model dtype; :meth:`BaseStrategy.reduce_loss`
    upcasts at the loss input.  The MoE extras ride along so the loss
    assembly can attach aux-loss and router diagnostics.
    """

    logits: Tensor
    aux_loss: Optional[Tensor] = None
    router_stats: Optional[List[RouterStats]] = None


def move_to_device(batch: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
    """Move batch tensors to specified device with non-blocking transfer."""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def get_logprobs(
    model: nn.Module,
    input_ids: Tensor,
    attn_mask: Tensor,
    loss_mask: Tensor,
    reduction: str,
) -> LogprobsOutput:
    """Compute token-wise log probabilities from model outputs.

    Args:
        model: The language model
        input_ids: Input token IDs of shape [batch_size, seq_len]
        attn_mask: Attention mask passed to the model (may include causal).
        loss_mask: Per-token mask for loss reduction.
        reduction: How to reduce over sequence dimension ("mean", "sum", "none")

    Returns:
        Log probabilities with reduction applied over sequence dimension
    """
    allowed_reductions = ["mean", "sum", "none"]
    if reduction not in allowed_reductions:
        raise ValueError(
            f"reduction must be one of {allowed_reductions}, got '{reduction}'"
        )

    shifted_input_ids = input_ids[:, 1:]
    shifted_loss_mask = loss_mask[:, 1:]

    outputs = model(
        input_ids[:, :-1],
        attn_mask[:, :, :-1, :-1] if attn_mask.dim() == 4 else attn_mask[:, :-1],
    )
    logits = outputs["logits"]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    token_logprobs = torch.gather(
        log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)

    if reduction == "mean":
        logprobs = (token_logprobs * shifted_loss_mask).sum(
            dim=-1
        ) / shifted_loss_mask.sum(dim=-1).clamp(min=1.0)
    elif reduction == "sum":
        logprobs = (token_logprobs * shifted_loss_mask).sum(dim=-1)
    else:
        logprobs = token_logprobs * shifted_loss_mask
    return {
        "logprobs": logprobs,
        "aux_loss": outputs.get("aux_loss"),
        "router_stats": outputs.get("router_stats"),
    }


def rollout_sequences(
    prompts: Tensor,
    prompt_mask: Tensor,
    responses: Tensor,
    response_masks: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Concatenate grouped prompts with responses for sequence scoring.

    Expands ``prompts`` [B, P] across the group dimension of ``responses``
    [B, G, R] and builds the combined key-padding + causal attention mask.

    Returns:
        ``(full_sequences, attn_mask)`` each shaped [B*G, P + R]; the
        attention mask is 4-D boolean.
    """
    group_size = responses.size(1)
    responses_flat = responses.view(-1, responses.size(-1))
    masks_flat = response_masks.view(-1, responses.size(-1)).bool()
    prompt_expanded = prompts.unsqueeze(1).repeat(1, group_size, 1).flatten(0, 1)
    prompt_mask_expanded = (
        prompt_mask.unsqueeze(1).expand(-1, group_size, -1).flatten(0, 1).bool()
    )

    full_sequences = torch.cat([prompt_expanded, responses_flat], dim=-1)
    # Build full attention mask: key-padding + causal
    key_pad = torch.cat([prompt_mask_expanded, masks_flat], dim=-1)[:, None, None, :]
    S = key_pad.shape[-1]
    causal = torch.tril(
        torch.ones(S, S, dtype=torch.bool, device=full_sequences.device)
    )[None, None, :, :]
    attn_mask = key_pad & causal
    return full_sequences, attn_mask


def rollout_token_logprobs(
    model: nn.Module,
    prompts: Tensor,
    prompt_mask: Tensor,
    responses: Tensor,
    response_masks: Tensor,
) -> LogprobsOutput:
    """Per-response-token log probabilities for a grouped rollout batch.

    Prompt tokens are masked out (0) so logprobs are computed only for
    response tokens.  ``get_logprobs`` shifts the mask by one position, so
    the first response token's logprob (predicted from the last prompt
    token) is correctly included.

    Returns:
        ``logprobs`` reshaped to [B, G, R]: position j is the log-probability
        of response token j under ``model``.
    """
    batch_size, group_size, response_len = responses.shape
    prompt_len = prompts.size(1)
    full_sequences, attn_mask = rollout_sequences(
        prompts, prompt_mask, responses, response_masks
    )
    masks_flat = response_masks.view(-1, response_len)
    full_masks = torch.cat(
        [
            torch.zeros(
                batch_size * group_size,
                prompt_len,
                dtype=torch.bool,
                device=full_sequences.device,
            ),
            masks_flat,
        ],
        dim=-1,
    )

    # get_logprobs returns [B*G, S-1] (S = prompt_len + response_len).
    # Response token logprobs occupy the last ``response_len`` positions.
    output = get_logprobs(model, full_sequences, attn_mask, full_masks, "none")
    output["logprobs"] = output["logprobs"][:, prompt_len - 1 :].view(
        batch_size, group_size, response_len
    )
    return output


def rollout_token_values(
    model: nn.Module,
    prompts: Tensor,
    prompt_mask: Tensor,
    responses: Tensor,
    response_masks: Tensor,
) -> Tensor:
    """Critic values [B, G, R] aligned with response token positions.

    Position j holds V(s_j) — the value of the state right before response
    token j is emitted — matching the logprob alignment of
    :func:`rollout_token_logprobs`.
    """
    prompt_len = prompts.size(1)
    full_sequences, attn_mask = rollout_sequences(
        prompts, prompt_mask, responses, response_masks
    )
    output = model(full_sequences, input_mask=attn_mask)
    values = output["values"].float()[
        :, prompt_len - 1 : prompt_len - 1 + responses.size(-1)
    ]
    return values.view(responses.shape)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Tensor, Tensor]:
    """Generalized advantage estimation over padded response tokens.

    Args:
        rewards: [B, G, R] per-token rewards; the terminal reward must sit
            at each response's last valid position, padded positions 0.
        values: [B, G, R] rollout-time critic values V(s_t) (see
            :func:`rollout_token_values`).
        mask: [B, G, R] valid-token mask; padded positions are excluded
            and cannot leak into valid advantages.
        gamma: Discount factor.
        gae_lambda: GAE bias/variance trade-off.

    Returns:
        ``(advantages, returns)`` shaped [B, G, R].  The episode ends at the
        last valid token (no bootstrap value beyond truncation).
    """
    response_len = rewards.size(-1)
    flat_rewards = rewards.reshape(-1, response_len)
    flat_mask = mask.reshape(-1, response_len).to(values.dtype)
    # Padded values must be zero or the backward scan would leak them.
    flat_values = values.reshape(-1, response_len) * flat_mask

    advantages = torch.zeros_like(flat_rewards)
    gae = torch.zeros_like(flat_values[:, 0])
    for t in range(response_len - 1, -1, -1):
        next_values = (
            flat_values[:, t + 1]
            if t + 1 < response_len
            else torch.zeros_like(flat_values[:, t])
        )
        delta = flat_rewards[:, t] + gamma * next_values - flat_values[:, t]
        gae = flat_mask[:, t] * (delta + gamma * gae_lambda * gae)
        advantages[:, t] = gae
    returns = advantages + flat_values
    return advantages.view_as(rewards), returns.view_as(rewards)


def _validate_behavior_logprobs(behavior_logprobs: Tensor, responses: Tensor) -> None:
    """Reject behaviour-policy logprobs that do not match the responses."""
    if behavior_logprobs.shape != responses.shape:
        raise ValueError(
            "logprobs_old shape must match responses: "
            f"got {tuple(behavior_logprobs.shape)}, "
            f"expected {tuple(responses.shape)}"
        )
    if not torch.isfinite(behavior_logprobs).all():
        raise ValueError("logprobs_old must contain only finite values")


def _is_packed(position_ids: Tensor) -> bool:
    """Whether rows pack multiple documents (positions reset mid-row)."""
    return bool((position_ids[:, 1:] <= position_ids[:, :-1]).any())


def make_doc_boundary_mask(position_ids: Tensor) -> Tensor:
    S = position_ids.size(1)
    device = position_ids.device
    boundaries = position_ids[:, 1:] <= position_ids[:, :-1]
    doc_ids = torch.cat(
        [
            torch.zeros(position_ids.size(0), 1, dtype=torch.long, device=device),
            boundaries.long().cumsum(dim=1),
        ],
        dim=1,
    )
    same_doc = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
    return (same_doc & causal).unsqueeze(1)


def _collect_moe_diagnostics(
    router_stats_list: List[RouterStats],
) -> Dict[str, float]:
    """Collect MoE routing diagnostic metrics from per-layer router stats.

    Args:
        router_stats_list: One :class:`RouterStats` dict per MoE layer with
            keys ``probs`` (N, E) and ``topk_indices`` (N, K), both detached.

    Returns:
        Dict with keys: router_entropy, dead_expert_fraction,
        load_imbalance_mean, load_imbalance_max.  Values are averaged
        across layers.
    """
    layer_entropies: List[Tensor] = []
    layer_dead_fractions: List[Tensor] = []
    layer_imbalance_means: List[Tensor] = []
    layer_imbalance_maxs: List[Tensor] = []

    for stats in router_stats_list:
        probs = stats["probs"].float()
        topk_indices = stats["topk_indices"]
        num_experts = probs.shape[-1]
        if num_experts == 0:
            continue
        probs = probs.reshape(-1, num_experts)
        if probs.numel() == 0:
            continue

        # Router entropy
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()

        # Load from the actual dispatch: one-hot sum of top-k assignments.
        expert_counts = F.one_hot(topk_indices, num_experts).sum(dim=(0, 1)).float()
        ideal_load = expert_counts.mean()  # N*K / E
        load_ratios = expert_counts / max(float(ideal_load), 1.0)
        imbalance_mean = (load_ratios - 1.0).abs().mean()
        imbalance_max = load_ratios.max()
        dead_fraction = (expert_counts == 0).float().mean()

        layer_entropies.append(entropy)
        layer_dead_fractions.append(dead_fraction)
        layer_imbalance_means.append(imbalance_mean)
        layer_imbalance_maxs.append(imbalance_max)

    if not layer_entropies:
        return {}

    return {
        "router_entropy": float(torch.stack(layer_entropies).mean().cpu().item()),
        "dead_expert_fraction": float(
            torch.stack(layer_dead_fractions).mean().cpu().item()
        ),
        "load_imbalance_mean": float(
            torch.stack(layer_imbalance_means).mean().cpu().item()
        ),
        "load_imbalance_max": float(
            torch.stack(layer_imbalance_maxs).mean().cpu().item()
        ),
    }


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
        self._moe_metrics: Dict[str, float] = {}
        self.strategy_kwargs = kwargs
        self._rollout_runner = None

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

        # None lets the scheduler derive live+1 under the policy lock,
        # avoiding a read-compute-write race on policy_version.
        result = self._rollout_runner.apply_weight_update(None, optimizer.step)
        self._rollout_runner.step()
        return result

    def __call__(self, batch: Dict[str, Tensor]) -> LossOutput:
        """Run offline or online forward depending on runner injection."""
        if self._rollout_runner is None:
            return self.compute_loss_output(batch)

        result, is_fresh = self._rollout_runner(batch)
        if is_fresh:
            self._on_rollout_refresh()

        train_batch = self.prepare_from_rollout(result)
        output = self.compute_loss_output(train_batch)
        if is_fresh:
            output["metrics"].update(
                {
                    f"dynamic_sampling/{name}": value
                    for name, value in getattr(
                        self._rollout_runner, "last_sampling_metrics", {}
                    ).items()
                }
            )
        return output


class StrategyFactory(BaseFactory["BaseStrategy"]):
    """Factory class for creating training strategy instances.

    Supports decorator-based registration for extensible strategy types.
    All default strategies (seq, sft, dpo, grpo) are automatically registered.

    Example usage:
        @StrategyFactory.register("custom")
        class CustomStrategy(BaseStrategy):
            ...

        strategy = StrategyFactory.create("custom", model, device)
    """


# ============== Strategy Classes ==============
# All strategies are registered at class definition time using the decorator


@StrategyFactory.register("seq")
class SEQStrategy(BaseStrategy):
    """Standard next-token prediction training strategy.

    Computes cross-entropy loss for next token prediction.
    Optionally adds MoE load balancing auxiliary loss.
    """

    loss_reduction = LossReduction.TOKEN_MEAN

    def __init__(
        self,
        model: Union[nn.Module, Callable[..., Dict[str, Tensor]]],
        device: str,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(model, device, **kwargs)
        self.label_smoothing = label_smoothing

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
        outputs = self.model(
            input_ids=batch["input_ids"], position_ids=batch["position_ids"]
        )
        return ForwardResult(
            outputs["logits"], outputs.get("aux_loss"), outputs.get("router_stats")
        )

    def reduce_loss(
        self, forward: ForwardResult, batch: Dict[str, Tensor]
    ) -> TokenLoss:
        target_ids = batch["target_ids"]
        loss_sum = F.cross_entropy(
            input=forward.logits.flatten(0, 1).float(),
            target=target_ids.flatten(),
            label_smoothing=self.label_smoothing,
            reduction="sum",
        )
        token_count = torch.tensor(
            target_ids.numel(), dtype=torch.float32, device=target_ids.device
        )
        return TokenLoss(loss_sum, token_count)


@StrategyFactory.register("sft")
class SFTStrategy(BaseStrategy):
    """Supervised Fine-tuning strategy with loss masking.

    Applies cross-entropy loss only to tokens where loss_mask is True.
    Optionally adds MoE load balancing auxiliary loss.
    """

    loss_reduction = LossReduction.TOKEN_MEAN

    def __init__(
        self,
        model: Union[nn.Module, Callable[..., Dict[str, Tensor]]],
        device: str,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(model, device, **kwargs)
        self.label_smoothing = label_smoothing

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
        outputs = self.model(
            input_ids=batch["input_ids"],
            position_ids=position_ids,
            input_mask=input_mask,
        )
        return ForwardResult(
            outputs["logits"], outputs.get("aux_loss"), outputs.get("router_stats")
        )

    def reduce_loss(
        self, forward: ForwardResult, batch: Dict[str, Tensor]
    ) -> TokenLoss:
        ignore_index = -100
        target_ids = batch["target_ids"].masked_fill(~batch["loss_mask"], ignore_index)
        loss_sum = F.cross_entropy(
            input=forward.logits.flatten(0, 1).float(),
            target=target_ids.flatten(),
            ignore_index=ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="sum",
        )
        token_count = (target_ids != ignore_index).sum().to(dtype=torch.float32)
        return TokenLoss(loss_sum, token_count)


@StrategyFactory.register("dpo")
class DPOStrategy(BaseStrategy):
    """Direct Preference Optimization strategy.

    Implements the DPO loss from the paper "Direct Preference Optimization".
    Uses a reference model to compute KL divergence penalty.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        ref_model: nn.Module,
        beta: float = 0.1,
        reduction: str = "sum",
        **kwargs,
    ):
        super().__init__(model, device, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.reduction = reduction

    def compute_loss_output(self, batch: Dict[str, Tensor]) -> LossOutput:
        batch = move_to_device(batch, self.device)
        chosen_ids, rejected_ids = batch["chosen"], batch["rejected"]
        chosen_loss_mask = batch["chosen_mask"]
        rejected_loss_mask = batch["rejected_mask"]
        chosen_attention_mask = batch.get("chosen_attention_mask")
        rejected_attention_mask = batch.get("rejected_attention_mask")
        if chosen_attention_mask is None:
            chosen_attention_mask = chosen_ids.ne(0)
        if rejected_attention_mask is None:
            rejected_attention_mask = rejected_ids.ne(0)

        concat_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        concat_loss_mask = torch.cat([chosen_loss_mask, rejected_loss_mask], dim=0)
        concat_attention_mask = torch.cat(
            [chosen_attention_mask, rejected_attention_mask], dim=0
        )

        # Build full attention mask: key-padding + causal
        key_pad = concat_attention_mask.bool()[:, None, None, :]
        S = key_pad.shape[-1]
        causal = torch.tril(
            torch.ones(S, S, dtype=torch.bool, device=concat_ids.device)
        )[None, None, :, :]  # [1, 1, S, S]
        full_mask = key_pad & causal  # [B*2, 1, S, S] — composed

        policy_output = get_logprobs(
            self.model,
            concat_ids,
            full_mask,
            concat_loss_mask,
            self.reduction,
        )
        log_pi = policy_output["logprobs"]
        aux_loss = policy_output["aux_loss"]

        with torch.no_grad():
            ref_output = get_logprobs(
                self.ref_model,
                concat_ids,
                full_mask,
                concat_loss_mask,
                self.reduction,
            )
            log_ref = ref_output["logprobs"]

        log_pi_chosen = log_pi[: chosen_ids.shape[0]]
        log_pi_rejected = log_pi[chosen_ids.shape[0] :]
        log_ref_chosen = log_ref[: chosen_ids.shape[0]]
        log_ref_rejected = log_ref[chosen_ids.shape[0] :]

        pi_log_ratio = log_pi_chosen - log_pi_rejected
        ref_log_ratio = log_ref_chosen - log_ref_rejected

        ratio_diff = pi_log_ratio - ref_log_ratio
        dpo_loss = -F.logsigmoid(self.beta * ratio_diff).mean()

        return self._loss_output(
            dpo_loss,
            {"dpo_loss": dpo_loss},
            aux_loss,
            policy_output.get("router_stats"),
        )

    def supports_online(self) -> bool:
        return True

    def prepare_from_rollout(self, result: RolloutResult) -> Dict[str, Tensor]:
        """Build prompt-conditioned chosen/rejected sequences from rollout.

        DPO scores each response conditioned on its original prompt.  The
        prompt remains visible to attention while the loss mask covers only
        valid response tokens.
        """
        rewards = result.rewards
        prompts = result.prompts
        prompt_mask = result.prompt_mask.bool()
        responses = result.responses
        response_masks = result.response_mask.bool()
        best = rewards.argmax(dim=-1)
        worst = rewards.argmin(dim=-1)
        B = responses.shape[0]
        idx = torch.arange(B, device=responses.device)
        chosen_response = responses[idx, best]
        chosen_response_mask = response_masks[idx, best]
        rejected_response = responses[idx, worst]
        rejected_response_mask = response_masks[idx, worst]

        chosen = torch.cat([prompts, chosen_response], dim=-1)
        rejected = torch.cat([prompts, rejected_response], dim=-1)
        prompt_loss_mask = torch.zeros_like(prompt_mask)
        chosen_mask = torch.cat([prompt_loss_mask, chosen_response_mask], dim=-1)
        rejected_mask = torch.cat([prompt_loss_mask, rejected_response_mask], dim=-1)
        chosen_attention_mask = torch.cat([prompt_mask, chosen_response_mask], dim=-1)
        rejected_attention_mask = torch.cat(
            [prompt_mask, rejected_response_mask], dim=-1
        )
        return {
            "chosen": chosen,
            "chosen_mask": chosen_mask,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected": rejected,
            "rejected_mask": rejected_mask,
            "rejected_attention_mask": rejected_attention_mask,
        }


@StrategyFactory.register("grpo")
class GRPOStrategy(BaseStrategy):
    """Group Relative Policy Optimization strategy.

    Implements GRPO following DeepSeek-R1 with token-level PPO clipping.
    Advantages are group-normalized from scalar per-response rewards and
    broadcast across all response tokens.  The loss is computed **only on
    response tokens** — prompt tokens are masked out.

    Three policy roles are distinguished:

    * **Policy** ``self.model`` — the model being trained.
    * **Behaviour policy** — represented by per-token ``logprobs_old`` captured
      during online rollout.  Offline batches may instead use ``self.old_model``
      as a compatibility fallback.
    * **Reference model** ``self.ref_model`` — a frozen copy of the initial
      policy (typically the SFT checkpoint) used **only** for the KL
      regularisation term.  It is never updated during training.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        old_model: Optional[nn.Module],
        ref_model: nn.Module,
        clip_eps: float = 0.2,
        kl_coef: float = 0.01,
        group_size: int = 4,
        **kwargs,
    ):
        super().__init__(model, device, **kwargs)
        self.old_model = old_model
        self.ref_model = ref_model
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.group_size = group_size

    def sync_old_model(self):
        """Copy current policy weights to old model."""
        if self.old_model is None:
            raise RuntimeError("Cannot sync an unconfigured old policy model")
        state_dict = self.executor.unwrap_model(self.model)
        if self.executor.use_distributed:
            state_dict = broadcast_state_dict(state_dict)
        if state_dict is not None:
            self.old_model.load_state_dict(state_dict)

    def optimizer_step(self, optimizer: Optimizer):
        """Step the optimizer, then refresh the offline behaviour policy.

        Without this sync the frozen ``old_model`` drifts away from the
        training policy, so the PPO ratio degenerates and clipping shuts
        learning down. Online GRPO passes ``logprobs_old`` instead and
        runs with ``old_model=None``, skipping the sync.
        """
        result = super().optimizer_step(optimizer)
        if self.old_model is not None:
            self.sync_old_model()
        return result

    def compute_loss_output(self, batch: Dict[str, Tensor]) -> LossOutput:
        batch = move_to_device(batch, self.device)
        prompts = batch["prompts"]
        responses = batch["responses"]
        masks = batch["masks"]
        rewards = batch["rewards"]

        behavior_logprobs = batch.get("logprobs_old")
        if behavior_logprobs is not None:
            _validate_behavior_logprobs(behavior_logprobs, responses)
            behavior_logprobs = behavior_logprobs.detach().float()
        elif self.old_model is None:
            raise ValueError(
                "GRPO batches must provide logprobs_old when no old_model is configured"
            )

        prompt_mask = batch.get("prompt_mask")
        if prompt_mask is None:
            prompt_mask = prompts.ne(0)

        policy_output = rollout_token_logprobs(
            self.model, prompts, prompt_mask, responses, masks
        )
        token_log_probs_policy = policy_output["logprobs"]
        aux_loss = policy_output["aux_loss"]
        with torch.no_grad():
            if behavior_logprobs is None:
                token_log_probs_old = rollout_token_logprobs(
                    self.old_model, prompts, prompt_mask, responses, masks
                )["logprobs"]
            else:
                token_log_probs_old = behavior_logprobs
            token_log_probs_ref = rollout_token_logprobs(
                self.ref_model, prompts, prompt_mask, responses, masks
            )["logprobs"]

        token_masks = masks.float()

        # Group-normalized advantages from scalar per-response rewards.
        eps = 1e-8
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True, unbiased=False)
        advantages = (rewards - mean) / (std + eps)
        # Broadcast scalar advantage to every response token: [B, G, 1]
        advantages = advantages.unsqueeze(-1)

        # Token-level ratio (π_θ / π_old) and PPO clipping.
        log_ratio = token_log_probs_policy - token_log_probs_old
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        per_token_policy_loss = -torch.min(surr1, surr2)
        token_count = token_masks.sum().clamp(min=1.0)
        policy_loss = (per_token_policy_loss * token_masks).sum() / token_count

        # KL penalty to frozen reference model with k1 estimator (non-negative):
        # k1 = π_ref / π_θ - log(π_ref / π_θ) - 1, where π_ref / π_θ = exp(log_ref - log_policy).
        log_ref_ratio = token_log_probs_ref - token_log_probs_policy
        r = torch.exp(log_ref_ratio)
        kl_per_token = r - torch.log(r + eps) - 1.0
        kl_penalty = self.kl_coef * (kl_per_token * token_masks).sum() / token_count

        task_loss = policy_loss + kl_penalty
        return self._loss_output(
            task_loss,
            {"policy_loss": policy_loss, "kl_loss": kl_penalty},
            aux_loss,
            policy_output.get("router_stats"),
        )

    def supports_online(self) -> bool:
        return True

    def prepare_from_rollout(self, result: RolloutResult) -> Dict[str, Tensor]:
        return {
            "prompts": result.prompts,
            "prompt_mask": result.prompt_mask,
            "responses": result.responses,
            "masks": result.response_mask,
            "rewards": result.rewards,
            "logprobs_old": result.logprobs_old,
        }


@StrategyFactory.register("online_ppo")
class PPOStrategy(BaseStrategy):
    """Proximal Policy Optimization with a learned critic (actor-critic).

    Uses the same token-level clipped surrogate as GRPO, but advantages
    come from GAE(λ) over a :class:`~astrai.model.value.ValueModel` critic
    instead of group-normalized rewards.

    Roles:

    * **Policy** ``self.model`` — the actor being trained.
    * **Behaviour policy** — per-token ``logprobs_old`` captured by the
      rollout sampler; always required (no offline old-model fallback).
    * **Critic** ``self.critic`` — trained jointly by masked MSE regression
      against the GAE returns.  It owns a separate optimizer, stepped by
      :meth:`optimizer_step` *outside* the policy-version lock: the critic
      never serves generation, so its weights are not part of a policy
      version publication.
    * **Reference model** ``self.ref_model`` — optional frozen copy of the
      initial policy.  When set (and ``kl_coef > 0``), a per-token KL
      penalty (k3 estimator, ``logπ_old − logπ_ref``) is folded into the
      rewards before GAE, following InstructGPT-style reward shaping.

    Advantages and returns are computed once per rollout — with rollout-time
    critic values — and pinned on the :class:`RolloutResult`, so every
    replayed gradient step optimizes the same fixed targets, mirroring
    classic PPO's multiple epochs over one batch.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        critic: nn.Module,
        critic_optimizer: Optional[Optimizer] = None,
        ref_model: Optional[nn.Module] = None,
        clip_eps: float = 0.2,
        kl_coef: float = 0.01,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        vf_coef: float = 0.5,
        max_grad_norm: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(model, device, **kwargs)
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.ref_model = ref_model
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def optimizer_step(self, optimizer: Optimizer):
        """Step the policy under the version lock, then the critic.

        The critic step runs after the policy's atomic version publication:
        a concurrent rollout may already observe the new policy version,
        but only the (unused-for-generation) critic lags by one step.
        """
        result = super().optimizer_step(optimizer)
        if self.critic_optimizer is not None:
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
        return result

    @torch.no_grad()
    def _compute_advantages(
        self,
        prompts: Tensor,
        prompt_mask: Tensor,
        responses: Tensor,
        response_masks: Tensor,
        rewards: Tensor,
        behavior_logprobs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """GAE advantages/returns pinned to rollout-time critic values."""
        device = self.device
        prompts = prompts.to(device)
        prompt_mask = prompt_mask.to(device)
        responses = responses.to(device)
        response_masks = response_masks.to(device)
        rewards = rewards.to(device)
        behavior_logprobs = behavior_logprobs.to(device)

        values = rollout_token_values(
            self.critic, prompts, prompt_mask, responses, response_masks
        )

        # Terminal reward lands on each response's last valid token.
        token_rewards = torch.zeros_like(values)
        lengths = response_masks.long().sum(dim=-1)
        terminal = (lengths - 1).clamp(min=0)
        token_rewards.view(-1, values.size(-1)).scatter_(
            1, terminal.reshape(-1, 1), rewards.reshape(-1, 1).to(values.dtype)
        )

        if self.ref_model is not None and self.kl_coef > 0:
            ref_logprobs = rollout_token_logprobs(
                self.ref_model, prompts, prompt_mask, responses, response_masks
            )["logprobs"]
            # k3 per-token KL estimator folded into the reward.
            kl_penalty = behavior_logprobs.float() - ref_logprobs
            token_rewards = (
                token_rewards
                - self.kl_coef * kl_penalty * response_masks.to(values.dtype)
            )

        return compute_gae(
            token_rewards, values, response_masks, self.gamma, self.gae_lambda
        )

    def prepare_from_rollout(self, result: RolloutResult) -> Dict[str, Tensor]:
        _validate_behavior_logprobs(result.logprobs_old, result.responses)
        if result.advantages is None or result.returns is None:
            result.advantages, result.returns = self._compute_advantages(
                result.prompts,
                result.prompt_mask,
                result.responses,
                result.response_mask,
                result.rewards,
                result.logprobs_old,
            )
        return {
            "prompts": result.prompts,
            "prompt_mask": result.prompt_mask,
            "responses": result.responses,
            "masks": result.response_mask,
            "rewards": result.rewards,
            "logprobs_old": result.logprobs_old,
            "advantages": result.advantages,
            "returns": result.returns,
        }

    def supports_online(self) -> bool:
        return True

    def compute_loss_output(self, batch: Dict[str, Tensor]) -> LossOutput:
        batch = move_to_device(batch, self.device)
        prompts = batch["prompts"]
        responses = batch["responses"]
        masks = batch["masks"]

        behavior_logprobs = batch.get("logprobs_old")
        if behavior_logprobs is None:
            raise ValueError(
                "PPO batches must provide logprobs_old captured at rollout"
            )
        _validate_behavior_logprobs(behavior_logprobs, responses)
        behavior_logprobs = behavior_logprobs.detach().float()

        prompt_mask = batch.get("prompt_mask")
        if prompt_mask is None:
            prompt_mask = prompts.ne(0)

        advantages = batch.get("advantages")
        returns = batch.get("returns")
        if advantages is None or returns is None:
            advantages, returns = self._compute_advantages(
                prompts,
                prompt_mask,
                responses,
                masks,
                batch["rewards"],
                behavior_logprobs,
            )

        policy_output = rollout_token_logprobs(
            self.model, prompts, prompt_mask, responses, masks
        )
        token_log_probs_policy = policy_output["logprobs"]

        # Critic forward with gradients: value regression against the
        # rollout-pinned GAE returns.
        values = rollout_token_values(
            self.critic, prompts, prompt_mask, responses, masks
        )
        token_masks = masks.float()
        token_count = token_masks.sum().clamp(min=1.0)

        # Token-level ratio (π_θ / π_old) and PPO clipping.
        log_ratio = token_log_probs_policy - behavior_logprobs
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        per_token_policy_loss = -torch.min(surr1, surr2)
        policy_loss = (per_token_policy_loss * token_masks).sum() / token_count

        value_loss = ((values - returns) ** 2 * token_masks).sum() / token_count
        task_loss = policy_loss + self.vf_coef * value_loss

        def masked_variance(x: Tensor) -> Tensor:
            mean = (x * token_masks).sum() / token_count
            return ((x - mean) ** 2 * token_masks).sum() / token_count

        with torch.no_grad():
            explained_variance = 1.0 - masked_variance(
                values - returns
            ) / masked_variance(returns).clamp(min=1e-8)

        return self._loss_output(
            task_loss,
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "explained_variance": explained_variance,
            },
            policy_output["aux_loss"],
            policy_output.get("router_stats"),
        )


# Factory aliases: online variants use the same strategy class; the
# ``RolloutRunner`` is injected by ``TrainContextBuilder`` to enable
# online mode, so no separate subclass is needed.  ``PPOStrategy`` is
# registered under its sole name ``online_ppo`` — it has no offline mode.
StrategyFactory.register("online_grpo")(GRPOStrategy)
StrategyFactory.register("online_dpo")(DPOStrategy)
