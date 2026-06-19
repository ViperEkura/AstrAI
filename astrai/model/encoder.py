from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor

from astrai.config.model_config import EncoderConfig
from astrai.model.automodel import AutoModel
from astrai.model.components.decoder_block import DecoderBlock
from astrai.model.components.embedding import Embedding
from astrai.model.components.norm import RMSNorm
from astrai.model.components.rope import RotaryEmbedding
from astrai.model.transformer import process_attention_mask


@AutoModel.register("embedding")
class EmbeddingEncoder(AutoModel):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.config = config
        rope_dim = config.dim // config.n_heads
        rope_base = config.rope_theta if config.rope_theta is not None else 10000
        self.rotary_embedding = RotaryEmbedding(
            rope_dim, config.max_len, rope_base, rope_scaling=config.rope_scaling
        )
        self.embed_tokens = Embedding(
            config.vocab_size, config.dim, neftune_alpha=config.neftune_alpha
        )

        self.layers = nn.ModuleList(
            [DecoderBlock(config, layer_id) for layer_id in range(config.n_layers)]
        )

        self.norm = RMSNorm(config.dim, config.norm_eps)

        self.pooling_type = config.pooling_type or "mean"
        self.normalize_embeddings = config.normalize_embeddings or False

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict=True, assign=False):
        state_dict = dict(state_dict)
        state_dict.pop("lm_head.weight", None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        assert input_ids.ndim == 2
        B, S = input_ids.shape

        x = self.embed_tokens(input_ids)

        rotary_emb = self.rotary_embedding(x, position_ids)
        attn_mask = process_attention_mask(x, position_ids, input_mask, is_causal=False)

        for layer in self.layers:
            x = layer(x, rotary_emb, attn_mask, paged_cache=None)

        hidden_states = self.norm(x)

        if self.pooling_type == "cls":
            pooled = hidden_states[:, 0]
        elif self.pooling_type == "last":
            if input_mask is not None:
                lengths = input_mask.sum(dim=1) - 1
                pooled = hidden_states[torch.arange(B, device=x.device), lengths]
            else:
                pooled = hidden_states[:, -1]
        else:
            if input_mask is not None:
                mask = input_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                    min=1.0
                )
            else:
                pooled = hidden_states.mean(dim=1)

        if self.normalize_embeddings:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)

        return pooled
