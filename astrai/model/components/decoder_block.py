from dataclasses import asdict
from typing import Optional

import torch.nn as nn
from torch import Tensor

from astrai.inference.core.cache import KvcacheView
from astrai.model.components.attention import AttnFactory
from astrai.model.components.mlp import FFNFactory
from astrai.model.components.norm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        cfg = asdict(config)
        cfg["down_init_std"] = 0.02 / (2 * config.n_layers) ** 0.5
        self.attention = AttnFactory.create(config.attn_type, **cfg, layer_id=layer_id)
        self.input_norm = RMSNorm(config.dim, config.norm_eps)
        self.post_attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.mlp = FFNFactory.create(config.ffn_type, **cfg)

    def forward(
        self,
        x: Tensor,
        rotary_emb: Tensor,
        attention_mask: Optional[Tensor] = None,
        paged_cache: Optional[KvcacheView] = None,
    ) -> Tensor:
        attn_output = self.attention(
            self.input_norm(x),
            rotary_emb,
            attention_mask,
            paged_cache,
        )
        x = attn_output + x
        x = self.mlp(self.post_attention_norm(x)) + x

        return x
