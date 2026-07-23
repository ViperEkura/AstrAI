from dataclasses import asdict
from typing import Optional

import torch.nn as nn
from torch import Tensor

from astrai.inference.core.cache import CacheView
from astrai.model.components.attention import AttnFactory
from astrai.model.components.mlp import FFNFactory
from astrai.model.components.norm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        cfg = asdict(config)
        cfg.update(
            dim=config.hidden_size,
            dim_ffn=config.intermediate_size,
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            norm_eps=config.rms_norm_eps,
            down_init_std=0.02 / (2 * config.num_hidden_layers) ** 0.5,
        )
        self.attention = AttnFactory.create(config.attn_type, **cfg, layer_id=layer_id)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = FFNFactory.create(config.ffn_type, **cfg)

    def forward(
        self,
        x: Tensor,
        rotary_emb: Tensor,
        attention_mask: Optional[Tensor] = None,
        paged_cache: Optional[CacheView] = None,
        is_causal: bool = False,
        return_router_losses: bool = False,
    ):
        attn_output = self.attention(
            self.input_norm(x),
            rotary_emb,
            attention_mask,
            paged_cache,
            is_causal,
        )
        x = attn_output + x
        mlp_output = self.mlp(self.post_attention_norm(x))
        router_outputs = None
        if isinstance(mlp_output, tuple):
            mlp_output, *router_outputs = mlp_output
        x = mlp_output + x

        if return_router_losses:
            return x, router_outputs
        return x
