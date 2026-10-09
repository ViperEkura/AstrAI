"""Direct preference optimization objective."""

from typing import (
    Dict,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.trainer.rollout import RolloutResult
from astrai.trainer.strategy.base import BaseStrategy
from astrai.trainer.strategy.factory import StrategyCapabilities, StrategyFactory
from astrai.trainer.strategy.ops import (
    LossOutput,
    get_logprobs,
    move_to_device,
)


@StrategyFactory.register(
    "dpo", capabilities=StrategyCapabilities(reference_model=True)
)
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
            grad_chunked=self.gradient_chunked_logprobs,
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


# Online DPO uses the same objective with a rollout runner.
StrategyFactory.register(
    "online_dpo", capabilities=StrategyCapabilities(online=True, reference_model=True)
)(DPOStrategy)
