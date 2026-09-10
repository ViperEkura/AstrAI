from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.backend import apply_rotary_emb, attention
from astrai.factory import BaseFactory
from astrai.inference.cache import KVCache
from astrai.model.components.linear import Linear
from astrai.model.components.norm import RMSNorm


class AttnFactory(BaseFactory[nn.Module]):
    pass


@AttnFactory.register("gqa")
class GQA(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        use_qk_norm: bool,
        norm_eps: float,
        use_gated_attention: bool,
        layer_id: int,
        n_layers: int = 1,
    ):
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.use_qk_norm = use_qk_norm
        self.use_gated_attention = use_gated_attention

        self.q_proj = Linear(dim, n_heads * self.head_dim)
        self.k_proj = Linear(dim, n_kv_heads * self.head_dim)
        self.v_proj = Linear(dim, n_kv_heads * self.head_dim)
        self.o_proj = Linear(dim, dim, init_std=0.02 / (2 * n_layers) ** 0.5)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, norm_eps)
            self.k_norm = RMSNorm(self.head_dim, norm_eps)

        if self.use_gated_attention:
            self.gate = Linear(dim, dim)

    def _split_heads(self, x: Tensor, n_heads) -> Tensor:
        return x.reshape(*x.shape[:-1], n_heads, self.head_dim)

    def forward(
        self,
        x: Tensor,
        rotary_emb: Tensor,
        attn_mask: Tensor = None,
        kv_cache: Optional[KVCache] = None,
        is_causal: bool = False,
        fwd: Optional[str] = None,
    ) -> Tensor:
        q = self._split_heads(self.q_proj(x), self.n_heads)
        k = self._split_heads(self.k_proj(x), self.n_kv_heads)
        v = self._split_heads(self.v_proj(x), self.n_kv_heads)

        # Match the HuggingFace convention (Qwen2/Gemma): normalize Q/K
        # before RoPE. RMSNorm's per-channel gain does not commute with
        # the rotation, so the order changes numerics.
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        q, k = apply_rotary_emb(q, rotary_emb), apply_rotary_emb(k, rotary_emb)

        sdqa_out = attention(
            q, k, v, kv_cache, self.layer_id, attn_mask, is_causal, fwd
        ).reshape(*x.shape[:-1], self.n_heads * self.head_dim)

        if self.use_gated_attention:
            sdqa_out = sdqa_out * F.sigmoid(self.gate(x))

        out = self.o_proj(sdqa_out)
        return out


@AttnFactory.register("mla")
class MLA(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        norm_eps: float,
        use_qk_norm: bool,
        use_gated_attention: bool,
        layer_id: int,
        n_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.layer_id = layer_id
        self.n_rep = n_heads // n_kv_heads
        self.use_qk_norm = use_qk_norm
        self.use_gated_attention = use_gated_attention

        self.q_proj = Linear(dim, n_heads * self.head_dim, bias=False)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, norm_eps)
            self.k_norm = RMSNorm(self.head_dim, norm_eps)
        self.kv_a_proj = Linear(dim, kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank, norm_eps)

        self.kv_b_proj = Linear(
            kv_lora_rank,
            n_kv_heads * (2 * self.head_dim),
        )

        self.o_proj = Linear(
            dim, dim, bias=False, init_std=0.02 / (2 * n_layers) ** 0.5
        )

        if use_gated_attention:
            self.gate = Linear(dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        rotary_emb: Tensor,
        attn_mask: Tensor = None,
        kv_cache: Optional[KVCache] = None,
        is_causal: bool = False,
        fwd: Optional[str] = None,
    ) -> Tensor:
        q = self.q_proj(x)
        q = q.reshape(*x.shape[:-1], self.n_heads, self.head_dim)

        kv_compressed = self.kv_a_proj(x)
        kv_compressed = self.kv_norm(kv_compressed)

        kv = self.kv_b_proj(kv_compressed)
        kv = kv.reshape(*x.shape[:-1], self.n_kv_heads, -1)

        k_nope, k_rope, v = torch.split(
            kv, [self.qk_nope_head_dim, self.qk_rope_head_dim, self.head_dim], dim=-1
        )

        q_nope, q_rope = (
            q[..., : self.qk_nope_head_dim],
            q[..., self.qk_nope_head_dim :],
        )
        q_rope = apply_rotary_emb(q_rope, rotary_emb)
        k_rope = apply_rotary_emb(k_rope, rotary_emb)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_out = attention(
            q, k, v, kv_cache, self.layer_id, attn_mask, is_causal, fwd
        ).reshape(*x.shape[:-1], self.n_heads * self.head_dim)

        if self.use_gated_attention:
            attn_out = attn_out * F.sigmoid(self.gate(x))

        out = self.o_proj(attn_out)
        return out
