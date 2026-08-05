"""Training strategy implementations with factory pattern."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.factory import BaseFactory
from astrai.parallel.executor import broadcast_state_dict
from astrai.trainer.rollout import RolloutResult


class LossOutput(TypedDict):
    loss: Tensor
    metrics: Dict[str, float]


class LogprobsOutput(TypedDict):
    logprobs: Tensor
    aux_loss: Optional[Tensor]


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
    return {"logprobs": logprobs, "aux_loss": outputs.get("aux_loss")}


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


def _load_balancing_loss(router_probs: Tensor) -> Tensor:
    """Compute MoE load balancing auxiliary loss from router probabilities.

    Implements the Switch Transformer load balancing loss (eq. 4-6).
    Encourages tokens to be uniformly distributed across experts.

    Args:
        router_probs: (N, num_experts) tensor of softmax router probabilities.

    Returns:
        Scalar aux loss = num_experts * sum(f_i * P_i).
    """
    num_experts = router_probs.size(-1)
    # f_i: fraction of tokens dispatched to expert i (soft mean)
    f_i = router_probs.mean(dim=0)
    # P_i: average routing probability for expert i
    P_i = router_probs.mean(dim=0)
    return num_experts * torch.sum(f_i * P_i)


def _collect_moe_diagnostics(
    router_probs_list: List[Tensor],
    top_k: int,
) -> Dict[str, float]:
    """Collect MoE routing diagnostic metrics from router probabilities.

    Args:
        router_probs_list: List of (N, num_experts) router probability tensors,
            one per MoE layer.
        top_k: Number of top experts selected per token.

    Returns:
        Dict with keys: router_entropy, dead_expert_fraction,
        load_imbalance_mean, load_imbalance_max.  Values are averaged
        across layers.
    """
    layer_entropies: List[Tensor] = []
    layer_dead_fractions: List[Tensor] = []
    layer_imbalance_means: List[Tensor] = []
    layer_imbalance_maxs: List[Tensor] = []

    for probs in router_probs_list:
        probs = probs.detach().to(dtype=torch.float32)
        if probs.ndim == 0 or probs.shape[-1] == 0:
            continue
        probs = probs.reshape(-1, probs.shape[-1])
        if probs.numel() == 0:
            continue

        num_experts = probs.shape[-1]
        num_tokens = probs.shape[0]

        # Router entropy
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()

        # Top-k expert selection
        selected_experts = torch.topk(probs, top_k, dim=-1).indices  # [tokens, top_k]
        expert_mask = F.one_hot(selected_experts, num_experts)  # [tokens, top_k, E]
        expert_counts = expert_mask.sum(dim=(0, 1)).to(dtype=torch.float32)  # [E]

        # Ideal load: tokens * top_k / num_experts
        ideal_load = (num_tokens * top_k) / max(num_experts, 1)

        # Load imbalance ratios
        load_ratios = expert_counts / max(ideal_load, 1.0)
        imbalance_mean = (load_ratios - 1.0).abs().mean()
        imbalance_max = load_ratios.max()
        dead_fraction = (expert_counts == 0).to(dtype=torch.float32).mean()

        layer_entropies.append(entropy)
        layer_dead_fractions.append(dead_fraction)
        layer_imbalance_means.append(imbalance_mean)
        layer_imbalance_maxs.append(imbalance_max)

    if not layer_entropies:
        return {}

    return {
        "router_entropy": float(torch.stack(layer_entropies).mean().cpu().item()),
        "dead_expert_fraction": float(torch.stack(layer_dead_fractions).mean().cpu().item()),
        "load_imbalance_mean": float(torch.stack(layer_imbalance_means).mean().cpu().item()),
        "load_imbalance_max": float(torch.stack(layer_imbalance_maxs).mean().cpu().item()),
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
        self.extra_kwargs = kwargs
        self._rollout_runner = None

    @abstractmethod
    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Compute loss for the given batch.

        Args:
            batch: Dictionary containing batch tensors

        Returns:
            Computed loss tensor
        """
        raise NotImplementedError

    def compute_loss_output(self, batch: Dict[str, Tensor]) -> LossOutput:
        return self._normalize_output(self.compute_loss(batch))

    def _loss_output(
        self,
        task_loss: Tensor,
        metrics: Dict[str, Tensor],
        aux_loss: Optional[Tensor] = None,
    ) -> LossOutput:
        total_loss = task_loss
        if aux_loss is not None:
            weighted_aux_loss = self.moe_aux_loss_coef * aux_loss
            total_loss = total_loss + weighted_aux_loss
            metrics["moe_aux_loss"] = aux_loss
            metrics["moe_aux_loss_weighted"] = weighted_aux_loss
            self._refresh_moe_diagnostics(aux_loss)
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

    def _refresh_moe_diagnostics(self, aux_loss: Tensor) -> None:
        """Collect MoE routing diagnostics from model router probs.

        Populates ``self._moe_metrics`` with router entropy, dead expert
        fraction, load imbalance, and aux_loss.  Called from
        :meth:`_loss_output` when an MoE aux loss is present.
        """
        router_probs_list: List[Tensor] = self.model.get_moe_router_probs()
        if not router_probs_list:
            self._moe_metrics = {}
            return
        self._moe_metrics = _collect_moe_diagnostics(
            router_probs_list,
            self.model.config.n_activated_experts,
        )
        self._moe_metrics["aux_loss"] = float(aux_loss.detach().cpu().item())

    def on_optimizer_step(self):
        """Advance online rollout state after a successful optimizer step."""
        if self._rollout_runner is not None:
            self._rollout_runner.step()

    def __call__(self, batch: Dict[str, Tensor]) -> LossOutput:
        """Run offline or online forward depending on runner injection."""
        if self._rollout_runner is None:
            return self.compute_loss_output(batch)

        result, is_fresh = self._rollout_runner(batch)
        if is_fresh:
            self._on_rollout_refresh()

        train_batch = self.prepare_from_rollout(result)
        return self.compute_loss_output(train_batch)


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
        return self.compute_loss_output(batch)["loss"]

    def compute_loss_output(self, batch: Dict[str, Tensor]) -> LossOutput:
        batch = move_to_device(batch, self.device)
        input_ids, target_ids = batch["input_ids"], batch["target_ids"]
        outputs = self.model(input_ids=input_ids)
        logits = outputs["logits"]

        loss = F.cross_entropy(
            input=logits.flatten(0, 1).float(),
            target=target_ids.flatten(),
            label_smoothing=self.label_smoothing,
        )

        return self._loss_output(loss, {"task_loss": loss}, outputs.get("aux_loss"))


@StrategyFactory.register("sft")
class SFTStrategy(BaseStrategy):
    """Supervised Fine-tuning strategy with loss masking.

    Applies cross-entropy loss only to tokens where loss_mask is True.
    Optionally adds MoE load balancing auxiliary loss.
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
        return self.compute_loss_output(batch)["loss"]

    def compute_loss_output(self, batch: Dict[str, Tensor]) -> LossOutput:
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

        return self._loss_output(loss, {"task_loss": loss}, outputs.get("aux_loss"))


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
        return self.compute_loss_output(batch)["loss"]

    def compute_loss_output(self, batch: Dict[str, Tensor]) -> LossOutput:
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

        return self._loss_output(dpo_loss, {"dpo_loss": dpo_loss}, aux_loss)

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
        state_dict = self.executor.unwrap_model(self.model)
        if self.executor.use_distributed:
            state_dict = broadcast_state_dict(state_dict)
        if state_dict is not None:
            self.old_model.load_state_dict(state_dict)

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.compute_loss_output(batch)["loss"]

    def compute_loss_output(self, batch: Dict[str, Tensor]) -> LossOutput:
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
        policy_output = get_logprobs(
            self.model, full_sequences, attn_mask, full_masks, "none"
        )
        token_log_probs_policy = policy_output["logprobs"]
        aux_loss = policy_output["aux_loss"]
        token_log_probs_policy = token_log_probs_policy[:, prompt_len - 1 :]
        with torch.no_grad():
            old_output = get_logprobs(
                self.old_model, full_sequences, attn_mask, full_masks, "none"
            )
            token_log_probs_old = old_output["logprobs"]
            token_log_probs_old = token_log_probs_old[:, prompt_len - 1 :]
            ref_output = get_logprobs(
                self.ref_model, full_sequences, attn_mask, full_masks, "none"
            )
            token_log_probs_ref = ref_output["logprobs"]
            token_log_probs_ref = token_log_probs_ref[:, prompt_len - 1 :]

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

        task_loss = policy_loss + kl_penalty
        return self._loss_output(
            task_loss,
            {"policy_loss": policy_loss, "kl_loss": kl_penalty},
            aux_loss,
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
        }

    def _on_rollout_refresh(self):
        """Sync the behaviour policy whenever a fresh rollout arrives."""
        self.sync_old_model()


# Factory aliases: online variants use the same strategy class; the
# ``RolloutRunner`` is injected by ``TrainContextBuilder`` to enable
# online mode, so no separate subclass is needed.
StrategyFactory.register("online_grpo")(GRPOStrategy)
StrategyFactory.register("online_dpo")(DPOStrategy)
