"""Training strategy implementations with factory pattern."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.factory import BaseFactory
from astrai.trainer.rollout import RolloutResult


def create_ref_model(
    model_fn: Callable[[], nn.Module], state_dict: Dict[str, Tensor]
) -> nn.Module:
    """Create a frozen reference model from model_fn + full state dict."""
    ref_model = model_fn()
    ref_model.load_state_dict(state_dict)
    ref_model.requires_grad_(False)
    ref_model.eval()
    return ref_model


def move_to_device(batch: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
    """Move batch tensors to specified device with non-blocking transfer."""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def get_logprobs(
    model: nn.Module,
    input_ids: Tensor,
    attn_mask: Tensor,
    loss_mask: Tensor,
    reduction: str,
) -> Tensor:
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

    logits = model(
        input_ids[:, :-1],
        attn_mask[:, :, :-1, :-1] if attn_mask.dim() == 4 else attn_mask[:, :-1],
    )["logits"]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    token_logprobs = torch.gather(
        log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)

    if reduction == "mean":
        return (token_logprobs * shifted_loss_mask).sum(dim=-1) / shifted_loss_mask.sum(
            dim=-1
        ).clamp(min=1.0)
    elif reduction == "sum":
        return (token_logprobs * shifted_loss_mask).sum(dim=-1)
    else:
        return token_logprobs * shifted_loss_mask


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


class BaseStrategy(ABC):
    """Abstract base class for training strategies.

    When a :class:`~astrai.trainer.rollout.RolloutRunner` is injected via
    :meth:`set_rollout_runner`, the strategy transparently switches to
    online mode: each ``__call__`` produces a :class:`RolloutResult`,
    converts it to a training batch via :meth:`prepare_from_rollout`, and
    then computes the loss.  Without a runner the strategy runs in
    offline mode and consumes the batch directly.
    """

    def __init__(
        self,
        model: Union[nn.Module, Callable[..., Dict[str, Tensor]]],
        device: str,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.executor = kwargs.pop("executor", None)
        self.extra_kwargs = kwargs
        self._rollout_runner = None
        self.last_metrics: Dict[str, float] = {}

    def combine_model_loss(
        self, outputs: Dict[str, Tensor], language_model_loss: Tensor
    ) -> Tensor:
        """Add optional MoE router losses and expose detached train metrics."""
        self.last_metrics = {"language_model_loss": language_model_loss.detach().item()}
        router_loss = outputs.get("router_loss")
        if router_loss is None:
            return language_model_loss

        expert_load = outputs["router_expert_load"].detach().float()
        load_mean = expert_load.mean()
        load_cv = expert_load.std(unbiased=False) / load_mean.clamp_min(1e-12)
        self.last_metrics.update(
            router_loss=router_loss.detach().item(),
            router_aux_loss=outputs["router_aux_loss"].detach().item(),
            router_z_loss=outputs["router_z_loss"].detach().item(),
            router_entropy=outputs["router_entropy"].detach().item(),
            expert_load_min=expert_load.min().item(),
            expert_load_max=expert_load.max().item(),
            expert_load_cv=load_cv.item(),
        )
        return language_model_loss + router_loss

    @abstractmethod
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute loss for the given batch.

        Args:
            batch: Dictionary containing batch tensors

        Returns:
            Computed loss tensor
        """
        raise NotImplementedError

    def supports_online(self) -> bool:
        """Whether this strategy can operate with a rollout runner.

        Base implementation returns ``False``; strategies that implement
        :meth:`prepare_from_rollout` should override to return ``True``.
        """
        return False

    def set_rollout_runner(self, runner):
        """Inject a :class:`RolloutRunner` to enable online rollout mode."""
        self._rollout_runner = runner

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

    def on_optimizer_step(self):
        """Advance online rollout state after a successful optimizer step."""
        if self._rollout_runner is not None:
            self._rollout_runner.step()

    def __call__(self, batch: Dict[str, Tensor]) -> Tensor:
        """Run offline or online forward depending on runner injection."""
        if self._rollout_runner is None:
            return self.compute_loss(batch)

        result, is_fresh = self._rollout_runner(batch)
        if is_fresh:
            self._on_rollout_refresh()

        train_batch = self.prepare_from_rollout(result)
        return self.compute_loss(train_batch)


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
    """

    def __init__(
        self,
        model: Union[nn.Module, Callable[..., Dict[str, Tensor]]],
        device: str,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(model, device, **kwargs)
        self.label_smoothing = label_smoothing

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        input_ids, target_ids = batch["input_ids"], batch["target_ids"]
        outputs = self.model(input_ids=input_ids)
        logits = outputs["logits"]

        loss = F.cross_entropy(
            input=logits.flatten(0, 1).float(),
            target=target_ids.flatten(),
            label_smoothing=self.label_smoothing,
        )

        return self.combine_model_loss(outputs, loss)


@StrategyFactory.register("sft")
class SFTStrategy(BaseStrategy):
    """Supervised Fine-tuning strategy with loss masking.

    Applies cross-entropy loss only to tokens where loss_mask is True.
    """

    def __init__(
        self,
        model: Union[nn.Module, Callable[..., Dict[str, Tensor]]],
        device: str,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(model, device, **kwargs)
        self.label_smoothing = label_smoothing

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        input_ids, target_ids, position_ids, loss_mask = (
            batch["input_ids"],
            batch["target_ids"],
            batch["position_ids"],
            batch["loss_mask"],
        )

        ignore_index = -100
        input_mask = make_doc_boundary_mask(position_ids)
        target_ids = target_ids.masked_fill(~loss_mask, ignore_index)
        outputs = self.model(
            input_ids=input_ids, position_ids=position_ids, input_mask=input_mask
        )
        logits = outputs["logits"]

        loss = F.cross_entropy(
            input=logits.flatten(0, 1).float(),
            target=target_ids.flatten(),
            ignore_index=ignore_index,
            label_smoothing=self.label_smoothing,
        )

        return self.combine_model_loss(outputs, loss)


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

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        chosen_ids, rejected_ids = batch["chosen"], batch["rejected"]
        chosen_mask, rejected_mask = batch["chosen_mask"], batch["rejected_mask"]

        concat_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        concat_loss_mask = torch.cat([chosen_mask, rejected_mask], dim=0)

        # Build full attention mask: key-padding + causal
        key_pad = concat_ids.bool()[:, None, None, :]  # [B*2, 1, 1, S]
        S = key_pad.shape[-1]
        causal = torch.tril(
            torch.ones(S, S, dtype=torch.bool, device=concat_ids.device)
        )[None, None, :, :]  # [1, 1, S, S]
        full_mask = key_pad & causal  # [B*2, 1, S, S] — composed

        log_pi = get_logprobs(
            self.model,
            concat_ids,
            full_mask,
            concat_loss_mask,
            self.reduction,
        )

        with torch.no_grad():
            log_ref = get_logprobs(
                self.ref_model,
                concat_ids,
                full_mask,
                concat_loss_mask,
                self.reduction,
            )

        log_pi_chosen = log_pi[: chosen_ids.shape[0]]
        log_pi_rejected = log_pi[chosen_ids.shape[0] :]
        log_ref_chosen = log_ref[: chosen_ids.shape[0]]
        log_ref_rejected = log_ref[chosen_ids.shape[0] :]

        pi_log_ratio = log_pi_chosen - log_pi_rejected
        ref_log_ratio = log_ref_chosen - log_ref_rejected

        ratio_diff = pi_log_ratio - ref_log_ratio
        dpo_loss = -F.logsigmoid(self.beta * ratio_diff).mean()

        return dpo_loss

    def supports_online(self) -> bool:
        return True

    def prepare_from_rollout(self, result: RolloutResult) -> Dict[str, Tensor]:
        """Pick best/worst response per prompt by reward as chosen/rejected."""
        rewards = result.rewards
        responses = result.responses
        masks = result.response_mask
        best = rewards.argmax(dim=-1)
        worst = rewards.argmin(dim=-1)
        B = responses.shape[0]
        idx = torch.arange(B, device=responses.device)
        chosen = responses[idx, best]
        chosen_mask = masks[idx, best].float()
        rejected = responses[idx, worst]
        rejected_mask = masks[idx, worst].float()
        return {
            "chosen": chosen,
            "chosen_mask": chosen_mask,
            "rejected": rejected,
            "rejected_mask": rejected_mask,
        }


@StrategyFactory.register("grpo")
class GRPOStrategy(BaseStrategy):
    """Group Relative Policy Optimization strategy.

    Implements GRPO following DeepSeek-R1 with token-level PPO clipping.
    Advantages are group-normalized from scalar per-response rewards and
    broadcast across all response tokens.  The loss is computed **only on
    response tokens** — prompt tokens are masked out.

    Three model roles are distinguished:

    * **Policy** ``self.model`` — the model being trained.
    * **Old policy** ``self.old_model`` — the behaviour policy that generated
      the responses.  Used for the importance sampling ratio
      ``ρ = π_θ / π_old``.  Synced externally after each data-generation round.
    * **Reference model** ``self.ref_model`` — a frozen copy of the initial
      policy (typically the SFT checkpoint) used **only** for the KL
      regularisation term.  It is never updated during training.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        old_model: nn.Module,
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
        self.old_model.load_state_dict(self.executor.unwrap_model(self.model))

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        prompts = batch["prompts"]
        responses = batch["responses"]
        masks = batch["masks"]
        rewards = batch["rewards"]

        batch_size, group_size, response_len = responses.shape
        responses_flat = responses.view(-1, response_len)
        masks_flat = masks.view(-1, response_len)
        prompt_expanded = prompts.unsqueeze(1).repeat(1, group_size, 1).flatten(0, 1)
        prompt_mask = batch.get("prompt_mask")
        if prompt_mask is None:
            prompt_mask = prompts.ne(0)
        prompt_mask_expanded = (
            prompt_mask.unsqueeze(1).expand(-1, group_size, -1).flatten(0, 1)
        )
        prompt_len = prompt_expanded.size(1)

        full_sequences = torch.cat([prompt_expanded, responses_flat], dim=-1)
        # Prompt tokens are masked out (0) so logprobs are computed only for
        # response tokens.  get_logprobs shifts the mask by one position, so
        # the first response token's logprob (predicted from the last prompt
        # token) is correctly included.
        full_masks = torch.cat(
            [torch.zeros_like(prompt_expanded, dtype=torch.bool), masks_flat], dim=-1
        )

        # Build full attention mask: key-padding + causal
        key_pad = torch.cat([prompt_mask_expanded, masks_flat.bool()], dim=-1)[
            :, None, None, :
        ]
        S = key_pad.shape[-1]
        causal = torch.tril(
            torch.ones(S, S, dtype=torch.bool, device=full_sequences.device)
        )[None, None, :, :]
        attn_mask = key_pad & causal

        # get_logprobs returns [B*G, S-1] (S = prompt_len + response_len).
        # Response token logprobs occupy the last ``response_len`` positions
        # (the first response token is predicted from the last prompt token).
        token_log_probs_policy = get_logprobs(
            self.model, full_sequences, attn_mask, full_masks, "none"
        )[:, prompt_len - 1 :]
        with torch.no_grad():
            token_log_probs_old = get_logprobs(
                self.old_model, full_sequences, attn_mask, full_masks, "none"
            )[:, prompt_len - 1 :]
            token_log_probs_ref = get_logprobs(
                self.ref_model, full_sequences, attn_mask, full_masks, "none"
            )[:, prompt_len - 1 :]

        # Reshape to [B, G, response_len]
        token_log_probs_policy = token_log_probs_policy.view(batch_size, group_size, -1)
        token_log_probs_old = token_log_probs_old.view(batch_size, group_size, -1)
        token_log_probs_ref = token_log_probs_ref.view(batch_size, group_size, -1)
        token_masks = masks_flat.view(batch_size, group_size, -1).float()

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

        total_loss = policy_loss + kl_penalty

        return total_loss

    def supports_online(self) -> bool:
        return True

    def prepare_from_rollout(self, result: RolloutResult) -> Dict[str, Tensor]:
        return {
            "prompts": result.prompts,
            "prompt_mask": result.prompt_mask,
            "responses": result.responses,
            "masks": result.response_mask,
            "rewards": result.rewards,
        }

    def _on_rollout_refresh(self):
        """Sync the behaviour policy whenever a fresh rollout arrives."""
        self.sync_old_model()


# Factory aliases: online variants use the same strategy class; the
# ``RolloutRunner`` is injected by ``TrainContextBuilder`` to enable
# online mode, so no separate subclass is needed.
StrategyFactory._entries["online_grpo"] = GRPOStrategy
StrategyFactory._entries["online_dpo"] = DPOStrategy
