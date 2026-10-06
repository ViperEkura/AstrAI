from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.components.decoder_block import DecoderBlock
from astrai.model.components.embedding import Embedding
from astrai.model.components.norm import RMSNorm
from astrai.model.components.rope import RotaryEmbedding
from astrai.model.kv_cache import KVCache
from astrai.model.masking import (
    prepare_decoder_masks,
)
from astrai.model.masking import (
    process_attention_mask as process_attention_mask,
)


def init_module_weights(module: nn.Module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


class TransformerModel(nn.Module):
    """The base model every head attaches to: embed → layers → final norm.

    Task wrappers (:class:`AutoRegressiveLM`, :class:`ValueModel`, …) hold
    it as ``self.model``, mirroring HF's ``LlamaForCausalLM.model``.  The
    checkpoint namespace splits accordingly: trunk weights live under
    ``model.*`` and each wrapper's head sits at the top level, so a policy
    checkpoint warm-starts any head by loading its ``model.*`` subset.  The
    forward returns the post-norm hidden states (plus MoE aux statistics);
    input validation stays on the wrappers, which own the calling contract.
    """

    def __init__(self, config: AutoRegressiveLMConfig):
        super().__init__()
        self.config = config
        attention = config.attention
        self.layer_types = tuple(
            attention.type_for_layer(layer_id)
            for layer_id in range(config.num_hidden_layers)
        )
        rope_dims = {}
        if "gqa" in self.layer_types:
            rope_dims["gqa"] = (
                attention.gqa.rotary_dim
                or attention.gqa.head_dim
                or config.hidden_size // attention.num_heads
            )
        if "mla" in self.layer_types:
            rope_dims["mla"] = attention.mla.qk_rope_head_dim
        rope_base = config.rope_theta if config.rope_theta is not None else 10000
        self.rotary_embeddings = nn.ModuleDict(
            {
                kind: RotaryEmbedding(
                    dim,
                    config.max_position_embeddings,
                    rope_base,
                    rope_scaling=config.rope_scaling,
                )
                for kind, dim in rope_dims.items()
            }
        )
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            neftune_alpha=config.neftune_alpha,
        )

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    config,
                    layer_id,
                    attention_type=self.layer_types[layer_id],
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        position_ids: Optional[Tensor] = None,
        fwd: Optional[str] = None,
        logits_positions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        x = self.embed_tokens(input_ids)
        rotary_by_type = {
            kind: embedding(x, position_ids)
            for kind, embedding in self.rotary_embeddings.items()
        }
        masks = prepare_decoder_masks(input_mask, self.layer_types)

        aux_losses = []
        router_stats_list = []
        for layer, kind in zip(self.layers, self.layer_types):
            mask = masks[kind]
            layer_output = layer(
                x,
                rotary_by_type.get(kind),
                mask.tensor,
                kv_cache,
                mask.is_causal,
                fwd,
            )
            x = layer_output["hidden_states"]
            stats = layer_output.get("router_stats")
            if stats is not None:
                aux_losses.append(layer_output["aux_loss"])
                router_stats_list.append(stats)

        if logits_positions is not None:
            # RMSNorm is per-row, so gathering before it matches gathering after.
            hidden_states = self.norm(x[logits_positions])
        else:
            hidden_states = self.norm(x)

        output = {"hidden_states": hidden_states}
        if aux_losses:
            output["aux_loss"] = torch.stack(aux_losses).mean()
            output["router_stats"] = router_stats_list
        return output
