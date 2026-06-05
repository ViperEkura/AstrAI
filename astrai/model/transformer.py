from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.inference.core.cache import KvcacheView
from astrai.model.automodel import AutoModel
from astrai.model.components.decoder_block import DecoderBlock
from astrai.model.components.embedding import Embedding
from astrai.model.components.linear import Linear
from astrai.model.components.norm import RMSNorm
from astrai.model.components.rope import RotaryEmbedding


def process_attention_mask(
    input_tensor: Tensor,
    position_ids: Optional[Tensor],
    input_mask: Optional[Tensor] = None,
    is_causal: bool = False,
) -> Optional[Tensor]:
    if position_ids is None:
        return None
    if input_mask is not None and input_mask.dim() > 2:
        return input_mask

    device = input_tensor.device
    B = input_tensor.size(0)
    T = position_ids.max().item() + 1

    if input_mask is None:
        if position_ids.min().item() == 0 and is_causal:
            return None
        attend = torch.ones(B, 1, T, dtype=torch.bool, device=device)
    else:
        attend = input_mask[:, :T].to(device=device, dtype=torch.bool).unsqueeze(1)

    if is_causal:
        causal = position_ids.unsqueeze(-1) >= torch.arange(T, device=device)
        attend = attend & causal

    return attend.unsqueeze(1)


@AutoModel.register("autoregressive_lm")
class AutoRegressiveLM(AutoModel):
    """Autoregressive language model with paged KV cache."""

    def __init__(self, config: AutoRegressiveLMConfig):
        super().__init__(config)
        self.config = config
        rope_dim = (
            config.qk_rope_head_dim
            if config.attn_type == "mla"
            else config.dim // config.n_heads
        )
        rope_base = config.rope_theta if config.rope_theta is not None else 10000
        self.rotary_embedding = RotaryEmbedding(
            rope_dim, config.max_len, rope_base, rope_scaling=config.rope_scaling
        )
        self.embed_tokens = Embedding(config.vocab_size, config.dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    config.dim,
                    config.n_heads,
                    config.dim_ffn,
                    config.n_kv_heads,
                    config.norm_eps,
                    config.use_qk_norm,
                    config.use_gated_attention,
                    layer_id,
                    attn_type=config.attn_type,
                    ffn_type=config.ffn_type,
                    n_routed_experts=config.n_routed_experts,
                    n_shared_experts=config.n_shared_experts,
                    n_activated_experts=config.n_activated_experts,
                    topk_method=config.topk_method,
                    kv_lora_rank=config.kv_lora_rank,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                )
                for layer_id in range(config.n_layers)
            ]
        )

        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = Linear(config.dim, config.vocab_size)

        if self.config.tie_weight is True:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict=True, assign=False):
        lm_head_key = "lm_head.weight"
        embed_key = "embed_tokens.weight"

        state_dict = dict(state_dict)

        if self.config.tie_weight is True:
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

        if self.config.tie_weight is True:
            lm_head_key = prefix + "lm_head.weight"
            if lm_head_key in state_dict:
                del state_dict[lm_head_key]

        return state_dict

    def forward(
        self,
        input_ids: Tensor,
        input_mask: Optional[Tensor] = None,
        paged_cache: Optional[KvcacheView] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        assert input_ids.ndim == 2

        x = self.embed_tokens(input_ids)
        rotary_emb = self.rotary_embedding(x, position_ids)
        attn_mask = process_attention_mask(x, position_ids, input_mask, is_causal=True)

        for layer in self.layers:
            x = layer(x, rotary_emb, attn_mask, paged_cache)

        hidden_states = self.norm(x)
        logits = self.lm_head(hidden_states)

        return {"logits": logits, "hidden_states": hidden_states}
