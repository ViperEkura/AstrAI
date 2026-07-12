"""Training strategy implementations with factory pattern."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.factory import BaseFactory


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
    model: Union[nn.Module, Callable[..., Dict[str, Tensor]]],
    input_ids: Tensor,
    mask: Tensor,
    reduction: str,
) -> Tensor:
    """Compute token-wise log probabilities from model outputs.

    Args:
        model: The language model
        input_ids: Input token IDs of shape [batch_size, seq_len]
        mask: Attention mask of shape [batch_size, seq_len]
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
    shifted_mask = mask[:, 1:]

    logits = model(input_ids[:, :-1], mask[:, :-1])["logits"]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    token_logprobs = torch.gather(
        log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)

    if reduction == "mean":
        return (token_logprobs * shifted_mask).sum(dim=-1) / shifted_mask.sum(
            dim=-1
        ).clamp(min=1.0)
    elif reduction == "sum":
        return (token_logprobs * shifted_mask).sum(dim=-1)
    else:
        return token_logprobs * shifted_mask


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
    """Abstract base class for training strategies."""

    def __init__(
        self,
        model: Union[nn.Module, Callable[..., Dict[str, Tensor]]],
        device: str,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.executor = kwargs.pop("executor", None)
        self.model_fn = kwargs.pop("model_fn", None)
        self.extra_kwargs = kwargs

    @abstractmethod
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute loss for the given batch.

        Args:
            batch: Dictionary containing batch tensors

        Returns:
            Computed loss tensor
        """
        raise NotImplementedError

    def __call__(self, batch: Dict[str, Tensor]) -> Tensor:
        """Allow calling strategy directly as a callable."""
        return self.compute_loss(batch)


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
        logits = self.model(input_ids=input_ids)["logits"]

        loss = F.cross_entropy(
            input=logits.flatten(0, 1).float(),
            target=target_ids.flatten(),
            label_smoothing=self.label_smoothing,
        )

        return loss


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
        logits = self.model(
            input_ids=input_ids, position_ids=position_ids, input_mask=input_mask
        )["logits"]

        loss = F.cross_entropy(
            input=logits.flatten(0, 1).float(),
            target=target_ids.flatten(),
            ignore_index=ignore_index,
            label_smoothing=self.label_smoothing,
        )

        return loss


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
        beta: float = 0.1,
        reduction: str = "mean",
        **kwargs,
    ):
        super().__init__(model, device, **kwargs)
        self.ref_model = create_ref_model(
            self.model_fn, self.executor.unwrap_model(model)
        ).to(device=self.device)
        self.beta = beta
        self.reduction = reduction

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = move_to_device(batch, self.device)
        chosen_ids, rejected_ids = batch["chosen"], batch["rejected"]
        chosen_mask, rejected_mask = batch["chosen_mask"], batch["rejected_mask"]

        concat_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        concat_mask = torch.cat([chosen_mask, rejected_mask], dim=0)

        log_pi = get_logprobs(self.model, concat_ids, concat_mask, self.reduction)

        with torch.no_grad():
            log_ref = get_logprobs(
                self.ref_model, concat_ids, concat_mask, self.reduction
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


@StrategyFactory.register("grpo")
class GRPOStrategy(BaseStrategy):
    """Group Relative Policy Optimization strategy.

    Implements GRPO following DeepSeek-R1 with token-level PPO clipping.
    Advantages are group-normalized from scalar per-response rewards and
    broadcast across all response tokens.  The loss is computed **only on
    response tokens** — prompt tokens are masked out.

    The strategy expects offline-collected batches (``responses`` / ``rewards``
    pre-generated by the current or a recent policy).  Call ``sync_ref_model()``
    after each data-generation round so ``ref_model`` tracks the sampling policy.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        clip_eps: float = 0.2,
        kl_coef: float = 0.01,
        group_size: int = 4,
        sync_interval: int = 200,
        **kwargs,
    ):
        super().__init__(model, device, **kwargs)
        self.ref_model = create_ref_model(
            self.model_fn, self.executor.unwrap_model(model)
        ).to(device=self.device)
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.group_size = group_size
        self.sync_interval = sync_interval
        self._step = 0

    def sync_ref_model(self):
        """Copy current model weights to ref model."""
        self.ref_model.load_state_dict(self.executor.unwrap_model(self.model))

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        self._step += 1
        if self._step % self.sync_interval == 0:
            self.sync_ref_model()

        batch = move_to_device(batch, self.device)
        prompts = batch["prompts"]
        responses = batch["responses"]
        masks = batch["masks"]
        rewards = batch["rewards"]

        batch_size, group_size, response_len = responses.shape
        responses_flat = responses.view(-1, response_len)
        masks_flat = masks.view(-1, response_len)
        prompt_expanded = prompts.unsqueeze(1).repeat(1, group_size, 1).flatten(0, 1)
        prompt_len = prompt_expanded.size(1)

        full_sequences = torch.cat([prompt_expanded, responses_flat], dim=-1)
        # Prompt tokens are masked out (0) so logprobs are computed only for
        # response tokens.  get_logprobs shifts the mask by one position, so
        # the first response token's logprob (predicted from the last prompt
        # token) is correctly included.
        full_masks = torch.cat([torch.zeros_like(prompt_expanded), masks_flat], dim=-1)

        # get_logprobs returns [B*G, S-1] (S = prompt_len + response_len).
        # Response token logprobs occupy the last ``response_len`` positions
        # (the first response token is predicted from the last prompt token).
        token_log_probs_policy = get_logprobs(
            self.model, full_sequences, full_masks, "none"
        )[:, prompt_len - 1 :]
        with torch.no_grad():
            token_log_probs_ref = get_logprobs(
                self.ref_model, full_sequences, full_masks, "none"
            )[:, prompt_len - 1 :]

        # Reshape to [B, G, response_len]
        token_log_probs_policy = token_log_probs_policy.view(batch_size, group_size, -1)
        token_log_probs_ref = token_log_probs_ref.view(batch_size, group_size, -1)
        token_masks = masks_flat.view(batch_size, group_size, -1).float()

        # Group-normalized advantages from scalar per-response rewards.
        eps = 1e-8
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True)
        advantages = (rewards - mean) / (std + eps)
        # Broadcast scalar advantage to every response token: [B, G, 1]
        advantages = advantages.unsqueeze(-1)

        # Token-level ratio and PPO clipping.
        log_ratio = token_log_probs_policy - token_log_probs_ref
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        per_token_policy_loss = -torch.min(surr1, surr2)
        token_count = token_masks.sum().clamp(min=1.0)
        policy_loss = (per_token_policy_loss * token_masks).sum() / token_count

        # KL penalty with k1 estimator (non-negative): r - log(r) - 1, r=π_ref/π_θ.
        r = torch.exp(-log_ratio)
        kl_per_token = r - torch.log(r + eps) - 1.0
        kl_penalty = self.kl_coef * (kl_per_token * token_masks).sum() / token_count

        total_loss = policy_loss + kl_penalty

        return total_loss
