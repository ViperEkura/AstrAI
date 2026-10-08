"""Group relative policy optimization objective."""

import math
from contextlib import nullcontext
from numbers import Real
from typing import (
    Dict,
    Iterator,
    Optional,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from astrai.parallel.executor import broadcast_state_dict
from astrai.trainer.rollout import RolloutResult
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
        rl_microbatch_prompts: Optional[int] = None,
        loss_process_group=None,
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
        if rl_microbatch_prompts is not None and (
            isinstance(rl_microbatch_prompts, bool)
            or not isinstance(rl_microbatch_prompts, int)
            or rl_microbatch_prompts < 1
        ):
            raise ValueError("rl_microbatch_prompts must be a positive integer or None")
        self.rl_microbatch_prompts = rl_microbatch_prompts
        self.loss_process_group = loss_process_group
        self._window_denominator = None
        self._window_reject_aux = False
        self.training_global_prompts = 0

    @property
    def loss_dp_size(self):
        return (
            dist.get_world_size(self.loss_process_group) if dist.is_initialized() else 1
        )

    def _sum(self, tensor):
        if self.loss_dp_size > 1:
            dist.all_reduce(tensor, group=self.loss_process_group)
        return tensor

    def _response_count(self, mask):
        if self.loss_aggregation == "token":
            return mask.sum(dtype=torch.int64)
        return (mask.sum(dim=-1) > 0).sum().to(dtype=torch.int64)

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
        denominator = self._window_denominator
        if denominator is None:
            denominator = self._sum(self._response_count(mask))
        denominator = denominator.clamp(min=1).to(dtype=loss.dtype)
        if self.loss_aggregation == "token":
            return self.loss_dp_size * (loss * mask).sum() / denominator

        lengths = mask.sum(dim=-1)
        valid_sequences = lengths > 0
        per_sequence = (loss * mask).sum(dim=-1) / lengths.clamp(min=1.0)
        return self.loss_dp_size * (per_sequence * valid_sequences).sum() / denominator

    @staticmethod
    def _slice_prompt_groups(batch, begin, end):
        total = batch["responses"].shape[0]
        return {
            key: value[begin:end]
            if (isinstance(value, Tensor) and value.ndim and value.shape[0] == total)
            or key == "finish_reasons"
            else value
            for key, value in batch.items()
        }

    @staticmethod
    def _padding_group(batch):
        # Padding only schedules the same number of DDP forwards/backwards;
        # it contributes no responses to the frozen global denominator.
        total = batch["responses"].shape[0]
        padded = {
            key: value.new_zeros((1, *value.shape[1:]))
            if isinstance(value, Tensor) and value.ndim and value.shape[0] == total
            else value
            for key, value in batch.items()
        }
        if padded["responses"].shape[-1] == 0:
            for key in ("responses", "masks", "logprobs_old"):
                if key in padded:
                    value = padded[key]
                    padded[key] = value.new_zeros((*value.shape[:-1], 1))
        if "prompt_mask" in padded:
            padded["prompt_mask"].fill_(True)
        padded["finish_reasons"] = [[]]
        return padded

    def _window_outputs(self, batch):
        count = self._response_count(batch["masks"].to(self.device))
        count = self._sum(count)
        size = self.rl_microbatch_prompts or max(batch["responses"].shape[0], 1)
        chunks = math.ceil(batch["responses"].shape[0] / size)
        slots = torch.tensor(max(chunks, 1), device=self.device, dtype=torch.int64)
        empty_rank = torch.tensor(chunks == 0, device=self.device, dtype=torch.int64)
        if self.loss_dp_size > 1:
            dist.all_reduce(slots, op=dist.ReduceOp.MAX, group=self.loss_process_group)
            dist.all_reduce(
                empty_rank, op=dist.ReduceOp.MAX, group=self.loss_process_group
            )
        if count.item() == 0:
            return
        self._window_denominator = count
        self._window_reject_aux = slots.item() > 1 or bool(empty_rank.item())
        try:
            for index in range(slots.item()):
                padding = index >= chunks
                chunk = (
                    self._padding_group(batch)
                    if padding
                    else self._slice_prompt_groups(
                        batch, index * size, (index + 1) * size
                    )
                )
                if chunk["responses"].shape[-1] == 0:
                    chunk = self._padding_group(chunk)
                    padding = True
                sync = nullcontext()
                if index + 1 < slots.item():
                    if self.executor is not None and hasattr(self.executor, "_no_sync"):
                        sync = self.executor._no_sync(self.model)
                    elif hasattr(self.model, "no_sync"):
                        sync = self.model.no_sync()
                with sync:
                    output = self.compute_loss_output(chunk)
                    output["metrics"]["padding_microbatches"] = float(padding)
                    yield output
        finally:
            self._window_denominator = None
            self._window_reject_aux = False

    def training_updates(self, batch) -> Iterator[Iterator[LossOutput]]:
        if self._rollout_runner is not None:
            result, is_fresh = self._rollout_runner(batch)
            if is_fresh:
                self._on_rollout_refresh()
            prepared = self.prepare_from_rollout(result)
            epochs = self.rl_update_epochs
        else:
            prepared, epochs = batch, 1
        prompts = prepared["responses"].shape[0]
        self.training_global_prompts = self._sum(
            torch.tensor(prompts, device=self.device, dtype=torch.int64)
        ).item()
        size = self.rl_minibatch_prompts or max(prompts, 1)
        updates = torch.tensor(
            max(math.ceil(prompts / size), 1), device=self.device, dtype=torch.int64
        )
        if self.loss_dp_size > 1:
            dist.all_reduce(
                updates, op=dist.ReduceOp.MAX, group=self.loss_process_group
            )
        for _ in range(epochs):
            for index in range(updates.item()):
                chunk = self._slice_prompt_groups(
                    prepared, index * size, (index + 1) * size
                )
                yield self._window_outputs(chunk)

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
        if (
            self._window_reject_aux
            and aux_loss is not None
            and self.moe_aux_loss_coef != 0
        ):
            raise NotImplementedError(
                "microbatched or empty-rank MoE auxiliary loss needs its own "
                "global router-statistics reduction; use an unchunked update "
                "or an explicitly zero auxiliary coefficient"
            )
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
        output = self._loss_output(
            task_loss,
            metrics,
            aux_loss,
            policy_output.get("router_stats"),
        )
        # Backward uses D * local_sum / N_global. Report the DP mean of
        # those scaled contributions, leaving rank-local router diagnostics.
        if self.loss_dp_size > 1:
            names = ("loss", "policy_loss", "kl_loss")
            reported = torch.tensor(
                [output["metrics"][name] for name in names],
                dtype=torch.float64,
                device=self.device,
            )
            self._sum(reported)
            for name, value in zip(names, reported / self.loss_dp_size):
                output["metrics"][name] = value.item()
        return output

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
            "finish_reasons": result.finish_reasons,
        }


# Online GRPO uses the same objective with a rollout runner.
StrategyFactory.register("online_grpo")(GRPOStrategy)
