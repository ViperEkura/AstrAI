"""Translate compatible Hugging Face decoder checkpoint weight keys.

HF configuration fields are mapped by hf_mapping.json in each model
directory. Tensor conversion here handles layouts supported by AstrAI.
"""

import logging
import re
from typing import Any, Dict, Mapping

import torch

from astrai.config.base import BaseConfig
from astrai.serialization.hf_config import convert_hf_config, load_hf_mapping

logger = logging.getLogger(__name__)

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

_ASTR_KEY = re.compile(
    r"^(?:model\.(?:embed_tokens|norm)\."
    r"|model\.layers\.\d+\.(?:attention|input_norm|post_attention_norm|mlp)\."
    r"|lm_head\.)"
)


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


#: Component names only HuggingFace uses; AstrAI keys share the ``model.*``
#: trunk prefix, so the prefix alone cannot identify an HF checkpoint.
_HF_MARKERS = (
    "self_attn.",
    "linear_attn.",
    "input_layernorm",
    "post_attention_layernorm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.experts.",
)


def looks_like_hf_state_dict(state_dict: Mapping[str, Any]) -> bool:
    """Return True if *state_dict* uses HuggingFace key names."""
    return any(marker in key for key in state_dict for marker in _HF_MARKERS)


def _is_dense_mlp_layer(config: BaseConfig, layer_id: int) -> bool:
    """Return whether a layer uses dense MLP instead of routed experts."""
    if getattr(config, "ffn_type", "mlp") != "moe":
        return True
    mlp_only = getattr(config, "mlp_only_layers", None) or []
    if layer_id in mlp_only:
        return True
    step = getattr(config, "decoder_sparse_step", 1) or 1
    return step > 1 and (layer_id + 1) % step != 0


def adapt_config(raw: Dict[str, Any], model_dir: str | None = None) -> Dict[str, Any]:
    """Import an HF config when its model directory provides a mapping."""
    mapping = load_hf_mapping(model_dir) if model_dir is not None else None
    return convert_hf_config(raw, mapping) if mapping is not None else raw


def convert_hf_weights(
    state_dict: Mapping[str, Any],
    config: BaseConfig,
    mapping: Mapping[str, Any] | None = None,
    *,
    strict: bool = False,
) -> Dict[str, torch.Tensor]:
    """Rename HF state dict keys to AstrAI names.

    Keys that are already AstrAI-style pass through unchanged; unmapped
    HF keys are dropped with a warning, or rejected when ``strict=True``.
    Mapping-declared ``skip_prefixes`` remain explicit exclusions.
    """
    if getattr(config, "attn_type", "gqa") == "mla":
        if any("kv_a_proj_with_mqa" in key for key in state_dict):
            raise NotImplementedError(
                "MLA attention (DeepSeek-V2/V3 kv_a_proj_with_mqa) uses a "
                "different KV factorization and cannot be converted"
            )

    weight_options = mapping.get("weights", {}) if mapping is not None else {}
    norm_offset = weight_options.get("norm_weight_offset", 0)
    skip_prefixes = tuple(weight_options.get("skip_prefixes", []))
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
    for source_key, tensor in state_dict.items():
        if source_key.startswith(skip_prefixes):
            continue
        key = source_key
        for prefix in ("model.language_model.", "language_model."):
            if key.startswith(prefix):
                key = "model." + key[len(prefix) :]
                break
        new_key = None
        linear = re.match(
            r"^model\.layers\.(\d+)\.linear_attn\."
            r"(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|conv1d|norm|A_log|dt_bias|out_proj)"
            r"(?:\.(weight|bias))?$",
            key,
        )
        if linear:
            layer, name, suffix = linear.groups()
            root = f"model.layers.{layer}.attention."
            if name == "in_proj_qkv" and suffix == "weight":
                q_size = config.gdn_num_key_heads * config.gdn_key_head_dim
                v_size = config.gdn_num_value_heads * config.gdn_value_head_dim
                q, k, v = tensor.split((q_size, q_size, v_size), dim=0)
                converted[root + "q_proj.weight"] = q
                converted[root + "k_proj.weight"] = k
                converted[root + "v_proj.weight"] = v
                continue
            names = {
                "in_proj_z": "z_proj",
                "in_proj_a": "gate_proj",
                "in_proj_b": "beta_proj",
                "conv1d": "conv",
                "norm": "g_norm",
                "A_log": "A_log",
                "dt_bias": "dt_bias",
                "out_proj": "o_proj",
            }
            target = names[name]
            if name == "conv1d":
                new_key = root + "conv.weight"
            elif name == "norm":
                new_key = root + "g_norm.weight"
            elif name in ("A_log", "dt_bias"):
                new_key = root + name
            else:
                new_key = root + target + "." + (suffix or "weight")

        if new_key is None and ffn_type == "moe":
            m = _MOE_ROUTER.match(key)
            if m:
                new_key = f"model.layers.{m.group(1)}.mlp.router.weight"
            else:
                m = _MOE_EXPERTS.match(key)
                if m:
                    new_key = (
                        f"model.layers.{m.group(1)}.mlp.routed_experts.{m.group(2)}."
                        f"{m.group(3)}.{m.group(4)}"
                    )
                else:
                    m = _MOE_SHARED.match(key)
                    if m:
                        shared_idx = m.group(2) if m.group(2) is not None else "0"
                        new_key = (
                            f"model.layers.{m.group(1)}.mlp.shared_experts.{shared_idx}."
                            f"{m.group(3)}.{m.group(4)}"
                        )
            if new_key is None:
                m = _DENSE_MLP.match(key)
                if m and _is_dense_mlp_layer(config, int(m.group(1))):
                    new_key = f"model.layers.{m.group(1)}.mlp.{m.group(2)}.{m.group(3)}"
        elif new_key is None:
            m = _DENSE_MLP.match(key)
            if m:
                new_key = f"model.layers.{m.group(1)}.mlp.{m.group(2)}.{m.group(3)}"

        if new_key is None:
            m = _ATTN.match(key)
            if m:
                layer, projection, suffix = m.groups()
                root = f"model.layers.{layer}.attention."
                hd = getattr(config, "head_dim", None) or (
                    config.hidden_size // config.num_attention_heads
                    if permute_rope
                    else None
                )
                rd = getattr(config, "rotary_dim", None) or hd
                new_key = root + f"{projection}_proj.{suffix}"
                if projection == "q" and getattr(config, "use_gated_attention", False):
                    expected = config.num_attention_heads * hd
                    if tensor.shape[0] == 2 * expected:
                        if suffix != "weight":
                            raise ValueError(f"{key}: gated Q bias is not supported")
                        # HF stores [query, gate] for each attention head.
                        paired = tensor.reshape(config.num_attention_heads, 2, hd, -1)
                        converted[root + "gate.weight"] = (
                            paired[:, 1].reshape(expected, -1).contiguous()
                        )
                        tensor = paired[:, 0].reshape(expected, -1).contiguous()
                    elif tensor.shape[0] != expected:
                        raise ValueError(
                            f"{key}: expected {expected} or {2 * expected} rows"
                        )
                if permute_rope and projection in ("q", "k"):
                    rows = tensor.shape[0]
                    if rows % hd != 0:
                        raise ValueError(
                            f"{key}: {rows} rows not divisible by head_dim={hd}"
                        )
                    base = _half_to_interleaved(rd).to(tensor.device)
                    blocks = torch.arange(rows // hd, device=tensor.device) * hd
                    indices = (
                        torch.arange(rows, device=tensor.device).reshape(-1, hd).clone()
                    )
                    indices[:, :rd] = blocks[:, None] + base[None, :]
                    tensor = tensor.index_select(0, indices.flatten())
            elif (m := _Q_NORM.match(key)) is not None:
                new_key = f"model.layers.{m.group(1)}.attention.q_norm.weight"
                hd = getattr(config, "head_dim", None) or head_dim
                rd = getattr(config, "rotary_dim", None) or hd
                if permute_rope and tensor.shape[0] == hd:
                    indices = torch.arange(hd, device=tensor.device)
                    indices[:rd] = _half_to_interleaved(rd).to(tensor.device)
                    tensor = tensor.index_select(0, indices)
            elif (m := _K_NORM.match(key)) is not None:
                new_key = f"model.layers.{m.group(1)}.attention.k_norm.weight"
                hd = getattr(config, "head_dim", None) or head_dim
                rd = getattr(config, "rotary_dim", None) or hd
                if permute_rope and tensor.shape[0] == hd:
                    indices = torch.arange(hd, device=tensor.device)
                    indices[:rd] = _half_to_interleaved(rd).to(tensor.device)
                    tensor = tensor.index_select(0, indices)
            elif (m := _INPUT_NORM.match(key)) is not None:
                new_key = f"model.layers.{m.group(1)}.input_norm.weight"
            elif (m := _POST_NORM.match(key)) is not None:
                new_key = f"model.layers.{m.group(1)}.post_attention_norm.weight"
            elif (m := _EMBED.match(key)) is not None:
                new_key = "model.embed_tokens.weight"
            elif (m := _FINAL_NORM.match(key)) is not None:
                new_key = "model.norm.weight"
            elif (m := _LM_HEAD.match(key)) is not None:
                new_key = "lm_head.weight"

        if new_key is None:
            # Already-AstrAI keys pass through untouched; anything else is
            # an unmapped HF key and is dropped with a warning.
            if _ASTR_KEY.match(key):
                converted[key] = tensor
            else:
                skipped.append(key)
            continue
        else:
            if norm_offset and (
                new_key.endswith(".input_norm.weight")
                or new_key.endswith(".post_attention_norm.weight")
                or new_key.endswith(".q_norm.weight")
                or new_key.endswith(".k_norm.weight")
                or new_key == "model.norm.weight"
            ):
                tensor = tensor + norm_offset
            converted[new_key] = tensor

    if skipped:
        if strict:
            raise ValueError(
                f"Unmapped HuggingFace weight key(s) ({len(skipped)}): "
                + ", ".join(sorted(skipped)[:10])
            )
        logger.warning(
            "Dropped %d unmapped HuggingFace weight key(s): %s",
            len(skipped),
            ", ".join(sorted(skipped)[:10]),
        )
    return converted
