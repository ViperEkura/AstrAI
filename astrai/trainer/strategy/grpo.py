"""Group relative policy optimization objective."""

import math
from numbers import Real
from typing import (
    Any,
    Dict,
    Iterator,
    Optional,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from astrai.parallel.executor import broadcast_state_dict
from astrai.trainer.rollout import RolloutResult
from astrai.trainer.rollout.batching import slice_batch
from astrai.trainer.strategy.base import BaseStrategy
from astrai.trainer.strategy.factory import StrategyFactory
from astrai.trainer.strategy.ops import (
    LossOutput,
    _importance_ratio_metrics,
    _truncation_metric,
    _validate_behavior_logprobs,
    move_to_device,
    rollout_token_logprobs,
)


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
        clip_eps_low: Optional[float] = None,
        clip_eps_high: Optional[float] = None,
        kl_coef: float = 0.01,
        group_size: int = 4,
        loss_aggregation: str = "token",
        overlong_max_len: Optional[int] = None,
        overlong_buffer_len: int = 0,
        overlong_penalty_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(model, device, **kwargs)
        self.old_model = old_model
        self.ref_model = ref_model
        self.clip_eps = self._validate_clip_epsilon(clip_eps, "clip_eps", upper=True)
        self.clip_eps_low = self._validate_clip_epsilon(
            self.clip_eps if clip_eps_low is None else clip_eps_low,
            "clip_eps_low",
            upper=True,
        )
        self.clip_eps_high = self._validate_clip_epsilon(
            self.clip_eps if clip_eps_high is None else clip_eps_high,
            "clip_eps_high",
        )
        if self.clip_eps_high < self.clip_eps_low:
            raise ValueError(
                "clip_eps_high must be greater than or equal to clip_eps_low"
            )
        if loss_aggregation not in {"token", "sequence"}:
            raise ValueError("loss_aggregation must be 'token' or 'sequence'")
        self.loss_aggregation = loss_aggregation
        self.overlong_max_len, self.overlong_buffer_len = (
            self._validate_overlong_window(overlong_max_len, overlong_buffer_len)
        )
        self.overlong_penalty_scale = self._validate_non_negative_real(
            overlong_penalty_scale, "overlong_penalty_scale"
        )
        self.kl_coef = kl_coef
        self.group_size = group_size

    @staticmethod
    def _validate_clip_epsilon(value: float, name: str, upper: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        value = float(value)
        if not math.isfinite(value) or value < 0 or (upper and value >= 1):
            interval = "[0, 1)" if upper else "[0, infinity)"
            raise ValueError(f"{name} must be finite and in {interval}")
        return value

    @staticmethod
    def _validate_non_negative_real(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    @staticmethod
    def _validate_overlong_window(
        max_len: Optional[int], buffer_len: int
    ) -> tuple[Optional[int], int]:
        if max_len is None:
            if buffer_len != 0:
                raise ValueError(
                    "overlong_buffer_len requires overlong_max_len to be set"
                )
            return None, 0
        if isinstance(max_len, bool) or not isinstance(max_len, int) or max_len <= 0:
            raise ValueError("overlong_max_len must be a positive integer or None")
        if (
            isinstance(buffer_len, bool)
            or not isinstance(buffer_len, int)
            or buffer_len <= 0
            or buffer_len > max_len
        ):
            raise ValueError(
                "overlong_buffer_len must be a positive integer no greater "
                "than overlong_max_len"
            )
        return max_len, buffer_len

    def _reduce_token_loss(self, loss: Tensor, mask: Tensor) -> Tensor:
        """Reduce response-token losses with GRPO or DAPO weighting."""
        mask = mask.float()
        if self.loss_aggregation == "token":
            return (loss * mask).sum() / mask.sum().clamp(min=1.0)

        lengths = mask.sum(dim=-1)
        valid_sequences = lengths > 0
        per_sequence = (loss * mask).sum(dim=-1) / lengths.clamp(min=1.0)
        return (per_sequence * valid_sequences).sum() / valid_sequences.sum().clamp(
            min=1
        )

    def _shape_overlong_rewards(
        self, rewards: Tensor, token_masks: Tensor
    ) -> tuple[Tensor, Optional[Tensor]]:
        if self.overlong_max_len is None:
            return rewards, None

        lengths = token_masks.sum(dim=-1)
        penalty_start = self.overlong_max_len - self.overlong_buffer_len
        penalty = ((penalty_start - lengths) / self.overlong_buffer_len).clamp(
            min=-1.0, max=0.0
        )
        return rewards + self.overlong_penalty_scale * penalty, penalty

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
            self.model,
            prompts,
            prompt_mask,
            responses,
            masks,
            grad_chunked=self.gradient_chunked_logprobs,
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
        rewards, overlong_penalty = self._shape_overlong_rewards(rewards, token_masks)
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True, unbiased=False)
        advantages = (rewards - mean) / (std + eps)
        # Broadcast scalar advantage to every response token: [B, G, 1]
        advantages = advantages.unsqueeze(-1)

        # Token-level ratio (π_θ / π_old) and PPO clipping.
        log_ratio = token_log_probs_policy - token_log_probs_old
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(
                ratio,
                1 - self.clip_eps_low,
                1 + self.clip_eps_high,
            )
            * advantages
        )
        per_token_policy_loss = -torch.min(surr1, surr2)
        policy_loss = self._reduce_token_loss(per_token_policy_loss, token_masks)

        # KL penalty to frozen reference model with k1 estimator (non-negative):
        # k1 = π_ref / π_θ - log(π_ref / π_θ) - 1, where π_ref / π_θ = exp(log_ref - log_policy).
        log_ref_ratio = token_log_probs_ref - token_log_probs_policy
        r = torch.exp(log_ref_ratio)
        kl_per_token = r - torch.log(r + eps) - 1.0
        kl_penalty = self.kl_coef * self._reduce_token_loss(kl_per_token, token_masks)

        task_loss = policy_loss + kl_penalty
        metrics = {
            "policy_loss": policy_loss,
            "kl_loss": kl_penalty,
        }
        metrics.update(
            _importance_ratio_metrics(
                ratio, token_masks, self.clip_eps_low, self.clip_eps_high
            )
        )
        if overlong_penalty is not None:
            metrics["overlong_penalty_mean"] = overlong_penalty.mean()
            metrics["overlong_fraction"] = (overlong_penalty < 0).float().mean()
        metrics.update(_truncation_metric(batch.get("finish_reasons") or []))
        return self._loss_output(
            task_loss,
            metrics,
            aux_loss,
            policy_output.get("router_stats"),
        )

    def supports_online(self) -> bool:
        return True

    def training_updates(
        self, batch: Any, microbatch_prompts: Optional[int] = None
    ) -> Iterator[Iterator[LossOutput]]:
        if not isinstance(batch, RolloutResult):
            yield from super().training_updates(batch, microbatch_prompts)
            return

        self._on_rollout_refresh()
        prepared = self.prepare_from_rollout(batch)
        total = prepared["prompts"].shape[0]
        masks = prepared["masks"]
        denominator = self._response_count(masks)
        if denominator == 0:
            raise RuntimeError("async GRPO round has no valid response tokens")

        def microbatches() -> Iterator[LossOutput]:
            size = microbatch_prompts or total
            for begin in range(0, total, size):
                indices = list(range(begin, min(begin + size, total)))
                chunk = slice_batch(prepared, indices, total)
                count = self._response_count(chunk["masks"])
                if count == 0:
                    continue
                weight = count / denominator
                output = self.compute_loss_output(chunk)
                metrics = {}
                for name, value in output["metrics"].items():
                    metric = value.detach() if isinstance(value, Tensor) else value
                    metrics[name] = float(metric) * weight
                yield {"loss": output["loss"] * weight, "metrics": metrics}

        yield microbatches()

    def _response_count(self, masks: Tensor) -> int:
        if self.loss_aggregation == "sequence":
            return int(masks.any(dim=-1).sum().item())
        return int(masks.sum().item())

    def prepare_from_rollout(self, result: RolloutResult) -> Dict[str, Tensor]:
        return {
            "prompts": result.prompts,
            "prompt_mask": result.prompt_mask,
            "responses": result.responses,
            "masks": result.response_mask,
            "rewards": result.rewards,
            "logprobs_old": result.logprobs_old,
            "finish_reasons": result.finish_reasons,
        }


# Online GRPO uses the same objective with a rollout runner.
StrategyFactory.register("online_grpo")(GRPOStrategy)
