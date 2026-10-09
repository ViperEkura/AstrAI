"""Proximal policy optimization with a learned critic."""

from typing import (
    Dict,
    Optional,
    Tuple,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from astrai.trainer.rollout import RolloutResult
from astrai.trainer.strategy.base import BaseStrategy
from astrai.trainer.strategy.factory import StrategyCapabilities, StrategyFactory
from astrai.trainer.strategy.ops import (
    LossOutput,
    _importance_ratio_metrics,
    _truncation_metric,
    _validate_behavior_logprobs,
    compute_gae,
    move_to_device,
    rollout_token_logprobs,
    rollout_token_values,
)


@StrategyFactory.register(
    "online_ppo",
    capabilities=StrategyCapabilities(online=True, reference_model=True, critic=True),
)
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
            "finish_reasons": result.finish_reasons,
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
            self.model,
            prompts,
            prompt_mask,
            responses,
            masks,
            grad_chunked=self.gradient_chunked_logprobs,
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
                **_importance_ratio_metrics(
                    ratio, token_masks, self.clip_eps, self.clip_eps
                ),
                **_truncation_metric(batch.get("finish_reasons") or []),
            },
            policy_output["aux_loss"],
            policy_output.get("router_stats"),
        )
