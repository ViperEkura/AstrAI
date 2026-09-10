"""HuggingFace checkpoint adaptation for LLaMA-style decoder models.

AstrAI stores weights with its own key names (``layers.<i>.input_norm``,
``layers.<i>.mlp.gate``), while HuggingFace decoder-only checkpoints use
``model.layers.<i>.input_layernorm`` / ``model.layers.<i>.mlp.gate_proj``.
This module translates HF configs and state dicts so external checkpoints
can be loaded directly.

Supported families (LLaMA layout, dense and MoE):
- dense FFN: llama, mistral, qwen2, gemma, gemma2, phi3
- MoE FFN (Mixtral / Qwen2-MoE / DeepSeek-V3 layout): router
  ``mlp.gate``, routed experts ``mlp.experts.<j>``, shared experts
  ``mlp.shared_experts.<j>``

Not supported:
- MLA attention (DeepSeek-V2/V3 ``kv_a_proj_with_mqa``) uses a different
  KV factorization and cannot be converted numerically.
- Attention/MLP bias (``attention_bias`` / ``mlp_bias``) — AstrAI
  projections are bias-free.
"""

import logging
import re
from typing import Any, Dict, Mapping

import torch

from astrai.config.base import BaseConfig

logger = logging.getLogger(__name__)

HF_MODEL_TYPES = frozenset(
    {
        "llama",
        "mistral",
        "mixtral",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "gemma",
        "gemma2",
        "phi3",
    }
)

_EMBED = re.compile(r"^model\.embed_tokens\.weight$")
_ATTN = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)$")
_Q_NORM = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_norm\.weight$")
_K_NORM = re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_norm\.weight$")
_INPUT_NORM = re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.weight$")
_POST_NORM = re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$")
_FINAL_NORM = re.compile(r"^model\.norm\.weight$")
_LM_HEAD = re.compile(r"^lm_head\.weight$")
_DENSE_MLP = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.(weight|bias)$"
)
_MOE_ROUTER = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$")
_MOE_EXPERTS = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(weight|bias)$"
)
_MOE_SHARED = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.shared_expert(?:s)?(?:\.(\d+))?\."
    r"(gate|up|down)_proj\.(weight|bias)$"
)

_ASTR_PREFIXES = ("embed_tokens.", "layers.", "norm.", "lm_head.")


def _half_to_interleaved(head_dim: int) -> torch.Tensor:
    """Row permutation converting HF half-split RoPE coordinates to
    AstrAI interleaved coordinates.

    HF rotate_half pairs channels ``(i, i + head_dim/2)``; AstrAI pairs
    ``(2i, 2i + 1)``. Both use frequency ``i`` for the pair, so AstrAI
    channel ``2i`` takes the HF value at channel ``i`` and AstrAI
    ``2i + 1`` takes HF ``i + head_dim/2``.
    """
    half = head_dim // 2
    perm = torch.empty(head_dim, dtype=torch.long)
    perm[0::2] = torch.arange(half)
    perm[1::2] = torch.arange(half, head_dim)
    return perm


def looks_like_hf_state_dict(state_dict: Mapping[str, Any]) -> bool:
    """Return True if *state_dict* uses HuggingFace key names."""
    return any(
        key.startswith("model.")
        or "self_attn." in key
        or "input_layernorm" in key
        or "mlp.experts." in key
        for key in state_dict
    )


def _is_dense_mlp_layer(config: BaseConfig, layer_id: int) -> bool:
    """Return whether a layer uses dense MLP instead of routed experts."""
    if getattr(config, "ffn_type", "mlp") != "moe":
        return True
    mlp_only = getattr(config, "mlp_only_layers", None) or []
    if layer_id in mlp_only:
        return True
    step = getattr(config, "decoder_sparse_step", 1) or 1
    return step > 1 and (layer_id + 1) % step != 0


def adapt_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Translate *raw* for AstrAI if it looks like an HF model config."""
    if raw.get("model_type") in HF_MODEL_TYPES:
        return convert_hf_config(raw)
    return raw


def convert_hf_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an HF LLaMA-style config dict to AstrAI field names."""
    if raw.get("attention_bias") or raw.get("mlp_bias"):
        raise NotImplementedError(
            "attention_bias / mlp_bias checkpoints are not supported; "
            "AstrAI projections are bias-free"
        )

    cfg: Dict[str, Any] = {}
    for key in (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "rms_norm_eps",
        "tie_word_embeddings",
        "max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "num_attention_heads",
        "num_key_value_heads",
        "use_qk_norm",
        "use_gated_attention",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "topk_method",
        "norm_topk_prob",
        "moe_aux_loss_coef",
        "decoder_sparse_step",
        "mlp_only_layers",
        "neftune_alpha",
    ):
        if key in raw:
            cfg[key] = raw[key]

    if "qk_norm" in raw and "use_qk_norm" not in cfg:
        cfg["use_qk_norm"] = raw["qk_norm"]
    if (
        raw.get("model_type") in ("gemma", "gemma2")
        and "use_qk_norm" not in cfg
        and "qk_norm" not in raw
    ):
        # Gemma/Gemma2 always apply RMSNorm to Q and K before attention.
        cfg["use_qk_norm"] = True

    n_heads = raw.get("num_attention_heads")
    if cfg.get("num_key_value_heads") is None and n_heads is not None:
        cfg["num_key_value_heads"] = n_heads

    if raw.get("head_dim") is not None and n_heads and raw.get("hidden_size"):
        expected = raw["hidden_size"] // n_heads
        if raw["head_dim"] != expected:
            raise NotImplementedError(
                f"HF head_dim={raw['head_dim']} differs from the computed "
                f"head dim {expected}; AstrAI derives head_dim from "
                "hidden_size / num_attention_heads"
            )

    if "kv_lora_rank" in raw:
        cfg["attn_type"] = "mla"

    n_experts = raw.get("num_local_experts") or raw.get("n_routed_experts")
    if n_experts:
        cfg["ffn_type"] = "moe"
        cfg["n_routed_experts"] = n_experts
        if "num_experts_per_tok" in raw:
            cfg["n_activated_experts"] = raw["num_experts_per_tok"]
        if "n_activated_experts" in raw:
            cfg["n_activated_experts"] = raw["n_activated_experts"]
        if "n_shared_experts" in raw:
            cfg["n_shared_experts"] = raw["n_shared_experts"]
        elif raw.get("shared_expert_intermediate_size"):
            # Qwen2-MoE exposes a single un-indexed shared expert.
            cfg["n_shared_experts"] = 1
        else:
            # Mixtral has no shared experts.
            cfg["n_shared_experts"] = 0
        if cfg.get("moe_intermediate_size") is None and "intermediate_size" in raw:
            # MoE configs store the per-expert FFN size in intermediate_size.
            cfg["moe_intermediate_size"] = raw["intermediate_size"]
        first_k_dense = raw.get("first_k_dense_replace")
        if isinstance(first_k_dense, int) and first_k_dense > 0:
            cfg["mlp_only_layers"] = list(range(first_k_dense))
            cfg["decoder_sparse_step"] = 1

    cfg["model_type"] = "autoregressive_lm"
    return cfg


def convert_hf_weights(
    state_dict: Mapping[str, Any],
    config: BaseConfig,
) -> Dict[str, torch.Tensor]:
    """Rename HF state dict keys to AstrAI names.

    Keys that are already AstrAI-style pass through unchanged; unmapped
    HF keys are dropped with a warning. Use with ``strict=True`` to fail
    loudly when the checkpoint does not match the config.
    """
    if getattr(config, "attn_type", "gqa") == "mla":
        if any("kv_a_proj_with_mqa" in key for key in state_dict):
            raise NotImplementedError(
                "MLA attention (DeepSeek-V2/V3 kv_a_proj_with_mqa) uses a "
                "different KV factorization and cannot be converted"
            )

    ffn_type = getattr(config, "ffn_type", "mlp")
    permute_rope = getattr(config, "attn_type", "gqa") != "mla"
    head_dim = None
    if permute_rope:
        head_dim = config.hidden_size // config.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim={head_dim} is odd; rotary permutation requires even"
            )
    converted: Dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, tensor in state_dict.items():
        if key.startswith(_ASTR_PREFIXES):
            converted[key] = tensor
            continue

        new_key = None
        if ffn_type == "moe":
            m = _MOE_ROUTER.match(key)
            if m:
                new_key = f"layers.{m.group(1)}.mlp.router.weight"
            else:
                m = _MOE_EXPERTS.match(key)
                if m:
                    new_key = (
                        f"layers.{m.group(1)}.mlp.routed_experts.{m.group(2)}."
                        f"{m.group(3)}.{m.group(4)}"
                    )
                else:
                    m = _MOE_SHARED.match(key)
                    if m:
                        shared_idx = m.group(2) if m.group(2) is not None else "0"
                        new_key = (
                            f"layers.{m.group(1)}.mlp.shared_experts.{shared_idx}."
                            f"{m.group(3)}.{m.group(4)}"
                        )
            if new_key is None:
                m = _DENSE_MLP.match(key)
                if m and _is_dense_mlp_layer(config, int(m.group(1))):
                    new_key = f"layers.{m.group(1)}.mlp.{m.group(2)}.{m.group(3)}"
        else:
            m = _DENSE_MLP.match(key)
            if m:
                new_key = f"layers.{m.group(1)}.mlp.{m.group(2)}.{m.group(3)}"

        if new_key is None:
            m = _ATTN.match(key)
            if m:
                new_key = (
                    f"layers.{m.group(1)}.attention.{m.group(2)}_proj.{m.group(3)}"
                )
                if permute_rope and m.group(2) in ("q", "k"):
                    rows = tensor.shape[0]
                    if rows % head_dim != 0:
                        raise ValueError(
                            f"{key}: {rows} output rows not divisible by "
                            f"head_dim={head_dim}"
                        )
                    base = _half_to_interleaved(head_dim).to(tensor.device)
                    blocks = (
                        torch.arange(rows // head_dim, device=tensor.device) * head_dim
                    )
                    perm = (blocks[:, None] + base[None, :]).flatten()
                    tensor = tensor.index_select(0, perm)
            elif (m := _Q_NORM.match(key)) is not None:
                new_key = f"layers.{m.group(1)}.attention.q_norm.weight"
                if permute_rope and tensor.shape[0] == head_dim:
                    tensor = tensor.index_select(
                        0, _half_to_interleaved(head_dim).to(tensor.device)
                    )
            elif (m := _K_NORM.match(key)) is not None:
                new_key = f"layers.{m.group(1)}.attention.k_norm.weight"
                if permute_rope and tensor.shape[0] == head_dim:
                    tensor = tensor.index_select(
                        0, _half_to_interleaved(head_dim).to(tensor.device)
                    )
            elif (m := _INPUT_NORM.match(key)) is not None:
                new_key = f"layers.{m.group(1)}.input_norm.weight"
            elif (m := _POST_NORM.match(key)) is not None:
                new_key = f"layers.{m.group(1)}.post_attention_norm.weight"
            elif (m := _EMBED.match(key)) is not None:
                new_key = "embed_tokens.weight"
            elif (m := _FINAL_NORM.match(key)) is not None:
                new_key = "norm.weight"
            elif (m := _LM_HEAD.match(key)) is not None:
                new_key = "lm_head.weight"

        if new_key is None:
            skipped.append(key)
        else:
            converted[new_key] = tensor

    if skipped:
        logger.warning(
            "Dropped %d unmapped HuggingFace weight key(s): %s",
            len(skipped),
            ", ".join(sorted(skipped)[:10]),
        )
    return converted
