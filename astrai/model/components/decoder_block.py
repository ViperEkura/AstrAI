from dataclasses import asdict
from typing import Optional, TypedDict

import torch.nn as nn
from torch import Tensor

from astrai.model.components.attention import AttnFactory
from astrai.model.components.mlp import FFNFactory, RouterStats
from astrai.model.components.norm import RMSNorm
from astrai.model.kv_cache import KVCache


class DecoderOutput(TypedDict):
    hidden_states: Tensor
    aux_loss: Optional[Tensor]
    router_stats: Optional[RouterStats]


class DecoderBlock(nn.Module):
    def __init__(self, config, layer_id: int, attention_type: Optional[str] = None):
        super().__init__()
        cfg = asdict(config)
        attention_config = getattr(config, "attention", None)
        if attention_config is None:
            cfg.update(
                n_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads,
            )
            attn_type = attention_type or config.attn_type
        else:
            cfg.update(attention_config.module_kwargs())
            attn_type = attention_type or attention_config.type_for_layer(layer_id)
        cfg.update(
            dim=config.hidden_size,
            dim_ffn=config.intermediate_size,
            n_layers=config.num_hidden_layers,
            norm_eps=config.rms_norm_eps,
            down_init_std=0.02 / (2 * config.num_hidden_layers) ** 0.5,
        )
        self.attention_type = attn_type
        self.attention = AttnFactory.create(attn_type, **cfg, layer_id=layer_id)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        ffn_type = self._resolve_ffn_type(config, layer_id)
        self.mlp = FFNFactory.create(ffn_type, **cfg)

    @staticmethod
    def _resolve_ffn_type(config, layer_id: int) -> str:
        if config.ffn_type != "moe":
            return config.ffn_type
        mlp_only = config.mlp_only_layers or []
        if layer_id in mlp_only:
            return "mlp"
        if config.decoder_sparse_step > 1:
            if (layer_id + 1) % config.decoder_sparse_step != 0:
                return "mlp"
        return "moe"

    def forward(
        self,
        x: Tensor,
        rotary_emb: Optional[Tensor],
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        is_causal: bool = False,
        fwd: Optional[str] = None,
    ) -> DecoderOutput:
        attn_output = self.attention(
            self.input_norm(x),
            rotary_emb,
            attention_mask,
            kv_cache,
            is_causal,
            fwd,
        )
        x = attn_output + x
        normalized = self.post_attention_norm(x)
        mlp_output = self.mlp(normalized)
        x = mlp_output["hidden_states"] + x

        return {
            "hidden_states": x,
            "aux_loss": mlp_output["aux_loss"],
            "router_stats": mlp_output.get("router_stats"),
        }
