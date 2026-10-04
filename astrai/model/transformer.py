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


def process_attention_mask(
    input_mask: Optional[Tensor],
    *,
    causal: bool = False,
) -> Optional[Tensor]:
    """Expand masks, optionally adding causality to a 2-D key-padding mask.

    Explicit 3-D/4-D masks already define which query/key pairs may attend.
    """
    if input_mask is None:
        return None
    if input_mask.dim() == 2:
        mask = input_mask[:, None, None, :]
        if causal:
            seq_len = input_mask.size(-1)
            causal_mask = torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=input_mask.device
            ).tril()
            if input_mask.dtype == torch.bool:
                mask = mask & causal_mask
            else:
                # Preserve additive attention biases on allowed keys.
                mask = mask.expand(-1, 1, seq_len, -1).masked_fill(
                    ~causal_mask, float("-inf")
                )
        return mask
    if input_mask.dim() == 3:
        return input_mask[:, None, :, :]
    return input_mask


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
        rotary_emb = self.rotary_embedding(x, position_ids)
        # GDN supplies causality through its recurrence and needs the compact
        # key-padding mask; softmax attention needs both constraints combined.
        attn_mask = process_attention_mask(
            input_mask, causal=self.config.attn_type != "gdn"
        )
        use_sdpa_causal_mask = attn_mask is None

        aux_losses = []
        router_stats_list = []
        for layer in self.layers:
            layer_output = layer(
                x,
                rotary_emb,
                attn_mask,
                kv_cache,
                use_sdpa_causal_mask,
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
