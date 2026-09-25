"""Differentiable Gated DeltaNet (GDN) reference operators.

Per (batch, head) the recurrent state is ``S`` in ``R^{K x V}`` and the rule is
the delta rule with a per-head scalar decay gate:

    S_t = diag(exp(g_t)) S_(t-1) + beta_t k_t (v_t - diag(exp(g_t)) S_(t-1) k_t)^T
    o_t = q_t^T S_t

The gate acts *before* the state's prediction of the current key is computed:
decay the state, measure the prediction error against ``v_t``, then write that
error back scaled by the write gate ``beta_t``. ``q``/``k`` are L2-normalized
and ``q`` is additionally scaled by ``1/sqrt(K)``.

Three interfaces expose the same rule:

- :func:`chunk_gated_delta_rule` — O(T/chunk_size) sequential steps with
  matmuls inside each chunk. Training and prefill use this.
- :func:`recurrent_gated_delta_rule` — one step per token, the full-sequence
  reference the chunked path is validated against.
- :func:`recurrent_gated_delta_rule_step` — a single token with the caller
  holding the state. This is the decode operator; its state has a fixed shape
  independent of the number of steps taken.

Layout conventions (matching the rest of the attention components):

- ``chunk``/``recurrent`` take ``query``/``key``/``value`` as ``[B, T, H, D]``
  and ``g``/``beta`` as ``[B, T, H]``, and return ``[B, T, H, V]``.
- ``_step`` takes a single token with the sequence axis already removed:
  ``[B, H, D]`` and ``[B, H]``.
- The state is always ``[B, H, K, V]`` in float32, initialized to zeros.

Precision: every operator casts its inputs to float32, accumulates in float32,
and casts the output back to the input dtype. The recurrent state stays float32
regardless of the activation dtype, matching the FLA/HF reference implementations
and the kernels' requirement that state snapshots be fp32.

These are reference operators: pure PyTorch, no fused kernels, no backward
kernels of their own (autograd differentiates the ops directly). Padding is
handled by the caller's mask; document boundaries are not interpreted here.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

DEFAULT_CHUNK_SIZE = 64


def l2norm(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """L2 normalization with eps inside the sqrt, matching FLA's ``l2norm``."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _to_heads_fp32(x: Tensor) -> Tensor:
    """[B, T, H, ...] -> contiguous float32 [B, H, T, ...]."""
    return x.transpose(1, 2).contiguous().float()


def _from_heads(x: Tensor, dtype: torch.dtype) -> Tensor:
    """[B, H, T, ...] -> contiguous ``dtype`` [B, T, H, ...]."""
    return x.transpose(1, 2).contiguous().to(dtype)


def _normalize_and_scale(query: Tensor, key: Tensor) -> Tuple[Tensor, Tensor]:
    """L2-normalize both, then scale q by 1/sqrt(K). Order matches the reference:
    the scale is applied to the normalized query, not before."""
    query, key = l2norm(query), l2norm(key)
    return query * (key.shape[-1] ** -0.5), key


def chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Chunked (WY/UT-transform) evaluation of the delta rule.

    Each chunk of ``chunk_size`` tokens is handled with dense matmuls; only the
    per-chunk state recurrence is sequential, so the number of sequential steps
    is O(T/chunk_size) instead of O(T). This is the training/prefill operator.

    Causality comes from the lower-triangular decay mask applied to the
    intra-chunk scores, so no ``T x T`` matrix is ever materialized: the
    intra-chunk tensors are ``[B, H, T/chunk_size, chunk_size, chunk_size]``,
    linear in ``T``.

    Args:
        chunk_size: tokens per chunk. The kernels use 64; any positive divisor
            works here and must give the same result.
        initial_state: ``[B, H, K, V]`` float32 state to continue from.
        output_final_state: also return the state after the last token.

    Returns:
        ``(output [B, T, H, V], final_state [B, H, K, V] float32 or None)``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    dtype = query.dtype
    query, key, value = (_to_heads_fp32(x) for x in (query, key, value))
    g, beta = (_to_heads_fp32(x) for x in (g, beta))
    query, key = _normalize_and_scale(query, key)

    batch, heads, seq_len, key_dim = key.shape
    value_dim = value.shape[-1]

    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
        beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    total = seq_len + pad
    num_chunks = total // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = (
        x.reshape(batch, heads, num_chunks, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(batch, heads, num_chunks, chunk_size)

    # Padded rows carry g = 0 and beta = 0, so they neither decay the state nor
    # contribute a write: the final state is identical to the unpadded run.
    g = g.cumsum(dim=-1)
    # e^(G_i - G_j) for i >= j, zero above the diagonal.
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # UT transform: A = (I + tril(diag(beta) K K^T * decay))^-1, solved by
    # forward substitution. The loop is inherently sequential in the row index
    # but runs over every chunk at once; the clones keep autograd's saved
    # tensors unmodified while rows are still being read.
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    state = (
        torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    output = torch.zeros_like(value)
    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_new = v_i - k_cumdecay[:, :, i] @ state
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        output[:, :, i] = attn_inter + attn_i @ v_new
        state = (
            state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    output = output.reshape(batch, heads, total, value_dim)[:, :, :seq_len]
    return _from_heads(output, dtype), (state if output_final_state else None)


def recurrent_gated_delta_rule_step(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    state: Tensor,
) -> Tuple[Tensor, Tensor]:
    """One token of the delta rule: the decode operator.

    Args (single token, sequence axis absent):
        query: ``[B, H, K]``, key: ``[B, H, K]``, value: ``[B, H, V]``,
        g: ``[B, H]`` (log decay), beta: ``[B, H]`` (write gate),
        state: ``[B, H, K, V]`` float32.

    Returns:
        ``(output [B, H, V] in the input dtype, new_state [B, H, K, V] fp32)``.

    The state shape does not depend on how many steps have been taken, so decode
    memory is constant in the history length.
    """
    dtype = query.dtype
    query, key = _normalize_and_scale(query.float(), key.float())
    output, new_state = _delta_rule_update(
        query, key, value.float(), g.float(), beta.float(), state
    )
    return output.to(dtype), new_state


def _delta_rule_update(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    state: Tensor,
) -> Tuple[Tensor, Tensor]:
    """One step shared by both recurrent operators.

    All arguments are float32 and ``query``/``key`` are already normalized and
    scaled; the caller owns that work so the full-sequence path does it once
    instead of once per token. einsum keeps the ``[B, H, K, V]`` products out of
    memory, which matters because this runs once per token.
    """
    decayed = state * g.exp().unsqueeze(-1).unsqueeze(-1)
    prediction = torch.einsum("bhkv,bhk->bhv", decayed, key)
    delta = (value - prediction) * beta.unsqueeze(-1)
    new_state = decayed + torch.einsum("bhk,bhv->bhkv", key, delta)
    return torch.einsum("bhkv,bhk->bhv", new_state, query), new_state


def recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Full-sequence per-token recurrence: the reference the chunked operator and
    the decode operator are each checked against.

    Same layout and return contract as :func:`chunk_gated_delta_rule`. Normalizes
    q/k once for the whole sequence, as the FLA/HF reference does, rather than
    once per token. This is the slowest of the three operators on every device:
    it is the definition, not an implementation to ship.
    """
    dtype = query.dtype
    query, key, value = (_to_heads_fp32(x) for x in (query, key, value))
    g, beta = (_to_heads_fp32(x) for x in (g, beta))
    query, key = _normalize_and_scale(query, key)

    state = (
        torch.zeros(
            query.shape[0],
            query.shape[1],
            query.shape[-1],
            value.shape[-1],
            dtype=torch.float32,
            device=query.device,
        )
        if initial_state is None
        else initial_state.float()
    )
    outputs = []
    for index in range(query.shape[2]):
        output, state = _delta_rule_update(
            query[:, :, index],
            key[:, :, index],
            value[:, :, index],
            g[:, :, index],
            beta[:, :, index],
            state,
        )
        outputs.append(output)
    stacked = torch.stack(outputs, dim=2)
    return _from_heads(stacked, dtype), (state if output_final_state else None)


def chunk_gated_delta_rule_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    do: Tensor,
    dh: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Reference gradients for :func:`chunk_gated_delta_rule`.

    This is autograd through the reference forward, not a hand-derived reverse
    pass: the reference is already differentiable and is what every fused
    kernel in this file is checked against, so it is the natural ground truth
    for the backward kernels too. Hand-deriving it would add a second thing to
    get wrong without adding an independent opinion.

    Returns ``(dq, dk, dv, dg, dbeta)`` in the caller's layout and dtypes.
    """
    inputs = [
        t.detach().clone().requires_grad_(True) for t in (query, key, value, g, beta)
    ]
    with torch.enable_grad():
        output, final_state = chunk_gated_delta_rule(
            *inputs, output_final_state=dh is not None
        )
        grads = [do]
        if dh is not None:
            grads.append(dh)
        torch.autograd.backward(
            [output] + ([final_state] if dh is not None else []), grads
        )
    return tuple(t.grad for t in inputs)
