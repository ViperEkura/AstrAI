"""Autoregressive LM: the shared base model plus a vocab head."""

from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.automodel import AutoModel, ModelFactory
from astrai.model.components.linear import Linear
from astrai.model.kv_cache import KVCache
from astrai.model.transformer import TransformerModel, init_module_weights


@ModelFactory.register("autoregressive_lm")
class AutoRegressiveLM(AutoModel):
    """Autoregressive language model with paged KV cache."""

    def __init__(self, config: AutoRegressiveLMConfig):
        super().__init__(config)
        self.model = TransformerModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size)

        if self.config.tie_word_embeddings is True:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.apply(init_module_weights)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict=True, assign=False):
        lm_head_key = "lm_head.weight"
        embed_key = "model.embed_tokens.weight"

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
        kv_cache: Optional[KVCache] = None,
        position_ids: Optional[Tensor] = None,
        fwd: Optional[str] = None,
        logits_positions: Optional[Tensor] = None,
        skip_lm_head: bool = False,
        return_lm_head_weight: bool = False,
    ) -> Dict[str, Tensor]:
        if fwd is None:
            if input_ids.ndim != 2:
                raise ValueError("training input_ids must be [batch, seq_len]")
            if kv_cache is not None:
                raise ValueError("training forward does not accept a KV cache")
        elif fwd in ("prefill", "decode"):
            if input_ids.ndim != 1:
                raise ValueError("inference input_ids must be packed [tokens]")
            if kv_cache is None:
                raise ValueError("inference forward requires a KV cache")
        else:
            raise ValueError(f"unsupported forward mode: {fwd}")

        if return_lm_head_weight and not skip_lm_head:
            raise ValueError("return_lm_head_weight requires skip_lm_head")
        output = self.model(
            input_ids,
            input_mask=input_mask,
            kv_cache=kv_cache,
            position_ids=position_ids,
            fwd=fwd,
            logits_positions=logits_positions,
        )
        # skip_lm_head returns post-norm hidden states only, with logits
        # set to None: no-grad consumers compute per-token log-probs from
        # hidden @ lm_head.T in row chunks instead of materializing the
        # full [batch, seq, vocab] tensor (see trainer.strategy.get_logprobs).
        output["logits"] = (
            None if skip_lm_head else self.lm_head(output["hidden_states"])
        )
        if return_lm_head_weight:
            # A view keeps the head reachable from DDP's forward outputs when
            # a training strategy applies the projection outside this module.
            output["lm_head_weight"] = self.lm_head.weight.view_as(self.lm_head.weight)
            if self.lm_head.bias is not None:
                output["lm_head_bias"] = self.lm_head.bias.view_as(self.lm_head.bias)

        return output
