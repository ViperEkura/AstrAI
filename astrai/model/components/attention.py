from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.backend import apply_rotary_emb, attention
from astrai.factory import BaseFactory
from astrai.model.components.gdn_ops import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule_step,
)
from astrai.model.components.linear import Linear
from astrai.model.components.norm import RMSNorm
from astrai.model.kv_cache import KVCache


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


class GDNState(NamedTuple):
    """Inference state of one Gated DeltaNet layer.

    Attributes:
        conv: ``[B, conv_kernel_size - 1, C]`` — the trailing mixed q/k/v inputs
            the causal convolution still needs. The width follows the
            sglang/FLA convention (``width - 1``); HF stores ``width`` instead
            and the two are not interchangeable.
        recurrent: ``[B, HV, K, V]`` float32 — the delta-rule state matrix. Its
            shape is fixed, so decode memory does not grow with history.
    """

    conv: Tensor
    recurrent: Tensor


@AttnFactory.register("gdn")
class GDN(nn.Module):
    """Gated DeltaNet layer: chunked training forward plus recurrent inference.

    Training (:meth:`forward`) runs the chunked operator, whose sequential cost
    is O(T/64) rather than O(T). Inference runs through :meth:`prefill` and
    :meth:`decode_step`, which carry an explicit :class:`GDNState`; ``forward``
    rejects a KV cache because that cache is token-indexed K/V, not a recurrent
    matrix, and the state pool is not wired up yet.

    Position information comes from the recurrence and the local convolution, so
    ``rotary_emb`` is accepted for interface compatibility and ignored.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        gdn_num_key_heads: Optional[int] = None,
        gdn_num_value_heads: Optional[int] = None,
        gdn_key_head_dim: Optional[int] = None,
        gdn_value_head_dim: Optional[int] = None,
        gdn_conv_kernel_size: int = 4,
        norm_eps: float = 1e-6,
        n_layers: int = 1,
        **_,
    ):
        super().__init__()
        self.dim = dim
        self.n_key_heads = gdn_num_key_heads or n_heads
        self.n_value_heads = gdn_num_value_heads or n_heads
        self.key_dim = gdn_key_head_dim or dim // n_heads
        self.value_dim = gdn_value_head_dim or dim // n_heads
        self.conv_kernel_size = gdn_conv_kernel_size
        if self.n_value_heads % self.n_key_heads:
            raise ValueError(
                "gdn_num_value_heads must be divisible by gdn_num_key_heads"
            )
        if dim % n_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if min(self.n_key_heads, self.n_value_heads, self.key_dim, self.value_dim) < 1:
            raise ValueError(
                "Gated DeltaNet head counts and dimensions must be positive"
            )
        if self.conv_kernel_size < 1:
            raise ValueError("gdn_conv_kernel_size must be at least 1")

        q_size = self.n_key_heads * self.key_dim
        k_size = q_size
        v_size = self.n_value_heads * self.value_dim
        self.conv_channels = q_size + k_size + v_size
        self.conv_width = self.conv_kernel_size - 1
        self.q_proj = Linear(dim, q_size)
        self.k_proj = Linear(dim, k_size)
        self.v_proj = Linear(dim, v_size)
        self.conv = nn.Conv1d(
            self.conv_channels,
            self.conv_channels,
            self.conv_kernel_size,
            groups=self.conv_channels,
            bias=False,
        )
        self.gate_proj = Linear(dim, self.n_value_heads)
        self.beta_proj = Linear(dim, self.n_value_heads)
        self.z_proj = Linear(dim, v_size)
        self.g_norm = RMSNorm(self.value_dim, norm_eps)
        self.A_log = nn.Parameter(torch.empty(self.n_value_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.n_value_heads))
        self.o_proj = Linear(v_size, dim, init_std=0.02 / (2 * n_layers) ** 0.5)
        self.reset_parameters()

    def reset_parameters(self):
        # Qwen's parameterization: decay = exp(-exp(A_log) * softplus(a + dt_bias))
        # with A = exp(A_log) drawn from U(0, 16) and dt_bias = 1. The lower bound
        # is kept just above zero because log(0) is -inf.
        with torch.no_grad():
            self.A_log.uniform_(1e-3, 16.0).log_()
            self.dt_bias.fill_(1.0)

    # -- shared pieces ----------------------------------------------------

    def _mix_qkv(self, x: Tensor) -> Tensor:
        """Concatenated q/k/v projections feeding the local convolution."""
        return torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)

    def _share_key_heads(self, q: Tensor, k: Tensor):
        """Repeat key heads across value heads (Qwen uses 32 value : 16 key)."""
        repeat = self.n_value_heads // self.n_key_heads
        if repeat == 1:
            return q, k
        return q.repeat_interleave(repeat, dim=2), k.repeat_interleave(repeat, dim=2)

    def _split_qkv(self, mixed: Tensor, batch: int, seq_len: int):
        q_size = self.n_key_heads * self.key_dim
        v_size = self.n_value_heads * self.value_dim
        q, k, v = torch.split(mixed, (q_size, q_size, v_size), dim=-1)
        q = q.reshape(batch, seq_len, self.n_key_heads, self.key_dim)
        k = k.reshape(batch, seq_len, self.n_key_heads, self.key_dim)
        v = v.reshape(batch, seq_len, self.n_value_heads, self.value_dim)
        return (*self._share_key_heads(q, k), v)

    def _conv_input(self, mixed: Tensor, previous: Optional[Tensor]) -> Tensor:
        """Prefix the mixed q/k/v with the carried convolution context.

        ``previous`` always holds exactly ``conv_width`` rows, so the result is
        long enough for the convolution to return one output per input token.
        """
        if self.conv_width == 0:
            return mixed
        context = (
            mixed.new_zeros(mixed.shape[0], self.conv_width, mixed.shape[-1])
            if previous is None
            else previous
        )
        return torch.cat((context, mixed), dim=1)

    def _project_qkv(self, x: Tensor, previous: Optional[Tensor] = None):
        """Causal depthwise convolution over the mixed q/k/v, then the split.

        The convolution sees the three projections as one tensor, so it runs as
        a single depthwise pass rather than one per projection. Only the mixed
        tensor is convolved; q/k/v are already separate by the time the delta
        rule runs.
        """
        context = self._conv_input(self._mix_qkv(x), previous)
        convolved = F.silu(self.conv(context.transpose(1, 2))).transpose(1, 2)
        return self._split_qkv(convolved, x.shape[0], x.shape[1])

    def _gates(self, x: Tensor):
        """Log-space decay ``g`` and write gate ``beta``, per value head, fp32."""
        log_decay = -self.A_log.float().exp() * F.softplus(
            self.gate_proj(x).float() + self.dt_bias.float()
        )
        return log_decay, torch.sigmoid(self.beta_proj(x).float())

    def _finalize(self, core: Tensor, z: Tensor) -> Tensor:
        """Per-head gated RMSNorm (silu gate), then the output projection."""
        core = core.reshape(*core.shape[:2], self.n_value_heads, self.value_dim)
        core = self.g_norm(core) * F.silu(z.reshape_as(core))
        return self.o_proj(core.flatten(start_dim=-2))

    def _next_conv_window(self, mixed: Tensor, previous: Optional[Tensor]) -> Tensor:
        """Trailing ``conv_width`` mixed inputs, left-padded if still too short."""
        batch = mixed.shape[0]
        history = mixed if previous is None else torch.cat((previous, mixed), dim=1)
        if history.shape[1] < self.conv_width:
            short = self.conv_width - history.shape[1]
            history = torch.cat(
                (history.new_zeros(batch, short, history.shape[-1]), history), dim=1
            )
        if self.conv_width == 0:
            return history[:, :0]
        return history[:, -self.conv_width :].contiguous()

    def _padding_mask(self, x: Tensor, attn_mask: Optional[Tensor]):
        """Unwrap the 2-D padding mask to ``[B, T]`` bool, or None."""
        if attn_mask is None:
            return None
        if attn_mask.ndim != 4 or attn_mask.shape[1:3] != (1, 1):
            raise ValueError("Gated DeltaNet supports only a 2D padding mask")
        valid = attn_mask[:, 0, 0, :].to(dtype=torch.bool)
        if valid.shape != x.shape[:2]:
            raise ValueError(
                "Gated DeltaNet padding mask must match batch and sequence dimensions"
            )
        if valid.shape[1] > 1 and (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Gated DeltaNet supports right padding only")
        return valid

    # -- training ---------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        rotary_emb: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        is_causal: bool = True,
        fwd: Optional[str] = None,
    ) -> Tensor:
        if kv_cache is not None or fwd is not None:
            raise NotImplementedError(
                "Gated DeltaNet inference state is not wired into the paged cache; call "
                "prefill()/decode_step() with a GDNState instead"
            )
        if x.ndim != 3:
            raise ValueError(
                "Gated DeltaNet expects training inputs with shape [batch, seq, hidden]"
            )
        valid = self._padding_mask(x, attn_mask)
        q, k, v = self._project_qkv(x)
        log_decay, beta = self._gates(x)
        # Right padding is safe without touching the state: padded rows carry
        # beta = 0 and g = 0, so they write nothing and never decay, and
        # causality keeps them out of the preceding outputs.
        core, _ = chunk_gated_delta_rule(q, k, v, log_decay, beta)
        if valid is not None:
            core = core * valid[:, :, None, None].to(core.dtype)
        return self._finalize(core, self.z_proj(x))

    # -- inference --------------------------------------------------------

    def init_state(
        self,
        batch: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> GDNState:
        """Zero state for ``batch`` sequences.

        ``dtype`` is the activation dtype: the convolution window follows it so
        decode reproduces the batch path's convolution, while the recurrent
        matrix is always float32.
        """
        return GDNState(
            conv=torch.zeros(
                batch, self.conv_width, self.conv_channels, dtype=dtype, device=device
            ),
            recurrent=torch.zeros(
                batch,
                self.n_value_heads,
                self.key_dim,
                self.value_dim,
                dtype=torch.float32,
                device=device,
            ),
        )

    def prefill(self, x: Tensor, state: Optional[GDNState] = None):
        """Chunked evaluation of a prompt. Returns ``(output, GDNState)``.

        Padding is rejected rather than ignored: pad rows would be folded into
        the returned state. Pack documents and reset state at their boundaries
        instead (boundaries are not interpreted by these operators yet).
        """
        if x.ndim != 3:
            raise ValueError("Gated DeltaNet prefill expects [batch, seq, hidden]")
        previous_conv = None if state is None else state.conv
        q, k, v = self._project_qkv(x, previous_conv)
        log_decay, beta = self._gates(x)
        core, recurrent = chunk_gated_delta_rule(
            q,
            k,
            v,
            log_decay,
            beta,
            initial_state=None if state is None else state.recurrent,
            output_final_state=True,
        )
        return self._finalize(core, self.z_proj(x)), GDNState(
            conv=self._next_conv_window(
                self._mix_qkv(x), None if state is None else state.conv
            ),
            recurrent=recurrent,
        )

    def decode_step(self, x: Tensor, state: Optional[GDNState] = None):
        """One token per sequence. ``x`` is ``[B, 1, hidden]``.

        Returns ``(output [B, 1, hidden], GDNState)``. ``state=None`` starts from
        zeros, so decoding from scratch matches ``prefill`` on the same tokens.
        """
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(
                "Gated DeltaNet decode_step expects exactly one token per sequence"
            )
        if state is None:
            state = self.init_state(x.shape[0], dtype=x.dtype, device=x.device)
        mixed = self._mix_qkv(x)
        window = self._conv_input(mixed, state.conv)
        convolved = F.silu(self.conv(window.transpose(1, 2))).transpose(1, 2)
        q, k, v = self._split_qkv(convolved, x.shape[0], 1)
        log_decay, beta = self._gates(x)
        core, recurrent = recurrent_gated_delta_rule_step(
            q[:, 0], k[:, 0], v[:, 0], log_decay[:, 0], beta[:, 0], state.recurrent
        )
        return (
            self._finalize(core.unsqueeze(1), self.z_proj(x)),
            GDNState(
                conv=self._next_conv_window(mixed, state.conv), recurrent=recurrent
            ),
        )


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
