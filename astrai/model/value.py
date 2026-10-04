"""Value (critic) model for actor-critic RL training."""

from typing import Any, Dict, Mapping, Optional

import torch.nn as nn
from torch import Tensor

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.automodel import AutoModel, ModelFactory
from astrai.model.components.linear import Linear
from astrai.model.transformer import TransformerModel, init_module_weights


@ModelFactory.register("value_model")
class ValueModel(AutoModel):
    """Critic scoring each state with a scalar instead of vocab logits.

    Wraps the shared trunk with a scalar head (the HF task-wrapper shape),
    so the critic carries no ``lm_head`` — the policy's vocab projection
    (and its optimizer state) was pure dead weight here.  A policy
    checkpoint warm-starts the trunk with ``strict=False``: the
    ``lm_head.*`` key it carries is the one tolerated difference, dropped
    by :meth:`load_state_dict`.  Only ``value_head`` keeps its fresh
    initialization, and the trunk pass is the same ``TransformerModel`` the
    policy runs, so the hidden states stay identical by construction
    (``tests/trainer/test_ppo_strategy.py`` pins them).
    """

    def __init__(self, config: AutoRegressiveLMConfig):
        super().__init__(config)
        self.model = TransformerModel(config)
        self.value_head = Linear(config.hidden_size, 1, bias=True)
        self.apply(init_module_weights)
        # Zero head so training starts from V(s) == 0 and the first GAE
        # advantages are driven purely by rewards.
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict=True, assign=False):
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("lm_head.")
        }
        return super().load_state_dict(state_dict, strict, assign)

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("critic input_ids must be [batch, seq_len]")
        hidden_states = self.model(
            input_ids, input_mask=input_mask, position_ids=position_ids
        )["hidden_states"]
        values = self.value_head(hidden_states).squeeze(-1)
        return {"values": values}
