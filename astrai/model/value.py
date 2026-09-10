"""Value (critic) model for actor-critic RL training."""

from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.automodel import ModelFactory
from astrai.model.components.linear import Linear
from astrai.model.transformer import AutoRegressiveLM, process_attention_mask


@ModelFactory.register("value_model")
class ValueModel(AutoRegressiveLM):
    """Critic scoring each state with a scalar instead of vocab logits.

    Inherits the ``AutoRegressiveLM`` components so a policy checkpoint can
    warm-start the critic backbone (``load_state_dict(..., strict=False)``);
    only ``value_head`` keeps its fresh initialization.  The inherited
    ``lm_head`` parameters stay dormant — the forward below never projects
    through them — so checkpoints round-trip with stable keys.  The trunk
    pass mirrors ``AutoRegressiveLM.forward`` for training-style input;
    ``tests/trainer/test_ppo_strategy.py`` pins the two to identical
    hidden states.
    """

    def __init__(self, config: AutoRegressiveLMConfig):
        super().__init__(config)
        self.value_head = Linear(config.hidden_size, 1, bias=True)
        # Zero head so training starts from V(s) == 0 and the first GAE
        # advantages are driven purely by rewards.
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("critic input_ids must be [batch, seq_len]")
        x = self.embed_tokens(input_ids)
        rotary_emb = self.rotary_embedding(x, position_ids)
        attn_mask = process_attention_mask(input_mask)
        use_sdpa_causal_mask = attn_mask is None

        for layer in self.layers:
            x = layer(x, rotary_emb, attn_mask, None, use_sdpa_causal_mask, None)[
                "hidden_states"
            ]
        hidden_states = self.norm(x)
        values = self.value_head(hidden_states).squeeze(-1)
        return {"values": values}
