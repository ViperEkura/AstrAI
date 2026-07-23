from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.inference.core.cache import CacheView
from astrai.model.automodel import AutoModel
from astrai.model.components.decoder_block import DecoderBlock
from astrai.model.components.embedding import Embedding
from astrai.model.components.linear import Linear
from astrai.model.components.norm import RMSNorm
from astrai.model.components.rope import RotaryEmbedding


def process_attention_mask(
    input_mask: Optional[Tensor],
) -> Optional[Tensor]:
    if input_mask is None:
        return None
    if input_mask.dim() == 2:
        return input_mask[:, None, None, :]
    if input_mask.dim() == 3:
        return input_mask[:, None, :, :]
    return input_mask


@AutoModel.register("autoregressive_lm")
class AutoRegressiveLM(AutoModel):
    """Autoregressive language model with paged KV cache."""

    def __init__(self, config: AutoRegressiveLMConfig):
        super().__init__(config)
        self.config = config
        rope_dim = (
            config.qk_rope_head_dim
            if config.attn_type == "mla"
            else config.hidden_size // config.num_attention_heads
        )
        rope_base = config.rope_theta if config.rope_theta is not None else 10000
        self.rotary_embedding = RotaryEmbedding(
            rope_dim,
            config.max_position_embeddings,
            rope_base,
            rope_scaling=config.rope_scaling,
        )
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            neftune_alpha=config.neftune_alpha,
        )

        self.layers = nn.ModuleList(
            [
                DecoderBlock(config, layer_id)
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = Linear(config.hidden_size, config.vocab_size)

        if self.config.tie_word_embeddings is True:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict=True, assign=False):
        lm_head_key = "lm_head.weight"
        embed_key = "embed_tokens.weight"

        state_dict = dict(state_dict)

        if self.config.tie_word_embeddings is True:
            # same tensor for embed and lm_head
            if embed_key in state_dict:
                state_dict[lm_head_key] = state_dict[embed_key]
        else:
            if lm_head_key not in state_dict and embed_key in state_dict:
                # clone to avoid sharing gradients
                state_dict[lm_head_key] = torch.clone(state_dict[embed_key])

        return super().load_state_dict(state_dict, strict, assign)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        if self.config.tie_word_embeddings is True:
            lm_head_key = prefix + "lm_head.weight"
            if lm_head_key in state_dict:
                del state_dict[lm_head_key]

        return state_dict

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Optional[Tensor] = None,
        paged_cache: Optional[CacheView] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        assert input_ids.ndim == 2

        x = self.embed_tokens(input_ids)
        rotary_emb = self.rotary_embedding(x, position_ids)
        attn_mask = process_attention_mask(input_mask)
        use_sdpa_causal_mask = attn_mask is None

        router_outputs = []
        for layer in self.layers:
            x, layer_router_outputs = layer(
                x,
                rotary_emb,
                attn_mask,
                paged_cache,
                use_sdpa_causal_mask,
                return_router_losses=True,
            )
            if layer_router_outputs is not None:
                router_outputs.append(layer_router_outputs)

        hidden_states = self.norm(x)
        logits = self.lm_head(hidden_states)

        output = {"logits": logits, "hidden_states": hidden_states}
        if router_outputs:
            aux_losses, z_losses, expert_loads, router_entropies = zip(
                *router_outputs
            )
            router_aux_loss = torch.stack(aux_losses).mean()
            router_z_loss = torch.stack(z_losses).mean()
            output.update(
                router_aux_loss=router_aux_loss,
                router_z_loss=router_z_loss,
                router_loss=(
                    self.config.router_aux_loss_coef * router_aux_loss
                    + self.config.router_z_loss_coef * router_z_loss
                ),
                router_expert_load=torch.stack(expert_loads).mean(dim=0),
                router_entropy=torch.stack(router_entropies).mean(),
            )

        return output
