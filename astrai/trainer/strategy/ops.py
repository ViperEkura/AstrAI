"""Training strategy implementations with factory pattern."""

from dataclasses import dataclass
from typing import (
    Dict,
    List,
    NotRequired,
    Optional,
    Tuple,
    TypedDict,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from astrai.model.components.mlp import RouterStats
from astrai.model.masking import PackedDocumentMask


class LossOutput(TypedDict):
    loss: Tensor
    metrics: Dict[str, float]
    normalizer: NotRequired[float]


class LogprobsOutput(TypedDict):
    logprobs: Tensor
    aux_loss: Optional[Tensor]
    router_stats: Optional[List[RouterStats]]


@dataclass
class ForwardResult:
    """Model forward over the (possibly sequence-sharded) batch.

    ``logits`` keeps the model dtype. The training strategy's fused linear CE
    supplies ``loss_sum`` and leaves logits unset. MoE extras ride along so
    loss assembly can attach aux-loss and router diagnostics.
    """

    logits: Optional[Tensor]
    aux_loss: Optional[Tensor] = None
    router_stats: Optional[List[RouterStats]] = None
    loss_sum: Optional[Tensor] = None


def move_to_device(batch: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
    """Move batch tensors to specified device with non-blocking transfer.

    Non-tensor values (e.g. the rollout's per-response ``finish_reasons``)
    pass through untouched — only tensors can carry a device.
    """
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


#: Model-dtype byte budget for one row-chunk of the deferred lm_head matmul
#: in :func:`_chunked_token_logprobs`.  64 MB of bf16 logits per chunk keeps
#: the fp32 upcast + logsumexp workspace well under a quarter gigabyte while
#: amortizing the GEMM over hundreds of rows.
_CHUNK_LOGIT_BYTES = 64 * 1024 * 1024


def _chunked_token_logprobs(hidden_states: Tensor, weight: Tensor, targets: Tensor):
    """Per-token log-probs without materializing the full ``[N, S, V]`` tensor.

    The model forward is taken with ``skip_lm_head=True`` so only the
    post-norm hidden states ``[N, S, H]`` exist; rows are then pushed
    through ``lm_head`` in chunks sized by :data:`_CHUNK_LOGIT_BYTES`.
    Each chunk computes the same expression as the full-tensor path —
    ``gather(log_softmax(logits.float()))[target] == logits[target].float()
    - logsumexp(logits.float())`` — so results agree up to bf16 GEMM
    tiling noise.  No-grad callers only (autograd would retain every
    chunk's logits, defeating the point).
    """
    n, s, hidden = hidden_states.shape
    flat_hidden = hidden_states.reshape(n * s, hidden)
    flat_targets = targets.reshape(n * s)
    vocab, dtype_bytes = weight.shape[0], weight.element_size()
    rows_per_chunk = max(1, _CHUNK_LOGIT_BYTES // (vocab * dtype_bytes))
    weight_t = weight.t()
    out = torch.empty(n * s, dtype=torch.float32, device=hidden_states.device)
    for start in range(0, n * s, rows_per_chunk):
        end = min(start + rows_per_chunk, n * s)
        logits = flat_hidden[start:end] @ weight_t
        logits = logits.float()
        picked = logits.gather(-1, flat_targets[start:end].unsqueeze(-1)).squeeze(-1)
        out[start:end] = picked - torch.logsumexp(logits, dim=-1)
    return out.view(n, s)


def _chunked_token_logprobs_grad(
    hidden_states: Tensor, weight: Tensor, targets: Tensor
) -> Tensor:
    """Gradient-enabled twin of :func:`_chunked_token_logprobs`.

    Each row-chunk's lm_head matmul + logsumexp runs under non-reentrant
    activation checkpointing: the forward frees the chunk's logits right
    after producing its per-token log-probs, and the backward recomputes
    one chunk at a time.  Autograd therefore never retains the full
    ``[N, S, V]`` logits tensor (nor one copy per chunk), while the
    hidden-state and lm_head gradients match the full-tensor path up to
    GEMM tiling noise.  Gradients reach ``weight`` through the closure —
    non-reentrant checkpoint re-runs the body with grad enabled during
    backward, so closed-over parameters receive their grads naturally.
    """
    n, s, hidden = hidden_states.shape
    flat_hidden = hidden_states.reshape(n * s, hidden)
    flat_targets = targets.reshape(n * s)
    vocab, dtype_bytes = weight.shape[0], weight.element_size()
    rows_per_chunk = max(1, _CHUNK_LOGIT_BYTES // (vocab * dtype_bytes))

    def chunk_logprobs(chunk_hidden: Tensor, chunk_targets: Tensor) -> Tensor:
        logits = (chunk_hidden @ weight.t()).float()
        picked = logits.gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1)
        return picked - torch.logsumexp(logits, dim=-1)

    pieces = []
    for start in range(0, n * s, rows_per_chunk):
        end = min(start + rows_per_chunk, n * s)
        pieces.append(
            torch_checkpoint(
                chunk_logprobs,
                flat_hidden[start:end],
                flat_targets[start:end],
                use_reentrant=False,
            )
        )
    return torch.cat(pieces).view(n, s)


def _truncation_metric(finish_reasons: List[List[str]]) -> Dict[str, Tensor]:
    """Response-termination observability from rollout finish reasons.

    Distinguishes natural stops from length-truncated generations: a high
    truncation rate means the overlong penalty (and PPO's no-bootstrap-at-
    truncation convention) dominates the objective rather than the request
    reward.  Empty when the rollout carried no finish reasons (offline
    batches, or rollouts produced before the field existed).
    """
    flat = [reason for group in finish_reasons for reason in group]
    if not flat:
        return {}
    return {
        "truncation_rate": torch.tensor(
            sum(1 for reason in flat if reason == "length") / len(flat)
        ),
        "stop_rate": torch.tensor(
            sum(1 for reason in flat if reason == "stop") / len(flat)
        ),
    }


def _importance_ratio_metrics(
    ratio: Tensor, token_masks: Tensor, clip_low: float, clip_high: float
) -> Dict[str, Tensor]:
    """Drift observability for the importance ratio ``exp(logπ - logπ_old)``.

    Reports the ratio distribution over valid tokens plus the fraction
    touching the clip band — a cheap canary for replayed rollouts going
    stale (``max_policy_lag`` too loose) or behaviour log-probs that no
    longer match the sampling policy.
    """
    with torch.no_grad():
        valid = token_masks.bool()
        zero = ratio.sum() * 0.0
        if not bool(valid.any()):
            return {
                "ratio_mean": zero,
                "ratio_min": zero,
                "ratio_max": zero,
                "clip_fraction": zero,
            }
        ratios = ratio[valid]
        clipped = ((ratios < 1 - clip_low) | (ratios > 1 + clip_high)).float().mean()
        return {
            "ratio_mean": ratios.mean(),
            "ratio_min": ratios.min(),
            "ratio_max": ratios.max(),
            "clip_fraction": clipped,
        }


def get_logprobs(
    model: nn.Module,
    input_ids: Tensor,
    attn_mask: Tensor,
    loss_mask: Tensor,
    reduction: str,
    grad_chunked: bool = False,
) -> LogprobsOutput:
    """Compute token-wise log probabilities from model outputs.

    Args:
        model: The language model
        input_ids: Input token IDs of shape [batch_size, seq_len]
        attn_mask: Attention mask passed to the model (may include causal).
        loss_mask: Per-token mask for loss reduction.
        reduction: How to reduce over sequence dimension ("mean", "sum", "none")
        grad_chunked: Opt in to the checkpointed chunked lm_head even with
            gradients enabled (see :func:`_chunked_token_logprobs_grad`).

    Returns:
        Log probabilities with reduction applied over sequence dimension

    Under ``torch.no_grad`` the forward runs with ``skip_lm_head=True`` and
    log-probs are computed in row chunks from the hidden states (see
    :func:`_chunked_token_logprobs`) — the reference/old-policy passes never
    materialize the full ``[N, S, V]`` fp32 log-softmax.  With gradients
    enabled the full-tensor path runs unless ``grad_chunked`` is set (or the
    model cannot skip its lm_head), in which case the checkpointed chunked
    path computes the same log-probs without retaining the logits.
    """
    allowed_reductions = ["mean", "sum", "none"]
    if reduction not in allowed_reductions:
        raise ValueError(
            f"reduction must be one of {allowed_reductions}, got '{reduction}'"
        )

    shifted_input_ids = input_ids[:, 1:]
    shifted_loss_mask = loss_mask[:, 1:]

    sliced_mask = (
        attn_mask[:, :, :-1, :-1] if attn_mask.dim() == 4 else attn_mask[:, :-1]
    )
    use_chunked = (
        isinstance(model, nn.Module)
        and getattr(model, "lm_head", None) is not None
        and (grad_chunked or not torch.is_grad_enabled())
    )
    if use_chunked:
        try:
            outputs = model(input_ids[:, :-1], sliced_mask, skip_lm_head=True)
        except TypeError:
            # Model or wrapper does not accept the kwarg; full path below.
            outputs = None
        if outputs is not None and outputs.get("logits") is not None:
            # A wrapper silently ignored the flag; the chunked contract
            # (logits is None) did not hold.
            outputs = None
    else:
        outputs = None
    if outputs is None:
        outputs = model(input_ids[:, :-1], sliced_mask)

    if outputs["logits"] is None:
        chunk_fn = (
            _chunked_token_logprobs_grad
            if torch.is_grad_enabled()
            else _chunked_token_logprobs
        )
        token_logprobs = chunk_fn(
            outputs["hidden_states"], model.lm_head.weight, shifted_input_ids
        )
    else:
        log_probs = torch.log_softmax(outputs["logits"].float(), dim=-1)
        token_logprobs = torch.gather(
            log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)
        ).squeeze(-1)

    if reduction == "mean":
        logprobs = (token_logprobs * shifted_loss_mask).sum(
            dim=-1
        ) / shifted_loss_mask.sum(dim=-1).clamp(min=1.0)
    elif reduction == "sum":
        logprobs = (token_logprobs * shifted_loss_mask).sum(dim=-1)
    else:
        logprobs = token_logprobs * shifted_loss_mask
    return {
        "logprobs": logprobs,
        "aux_loss": outputs.get("aux_loss"),
        "router_stats": outputs.get("router_stats"),
    }


def rollout_sequences(
    prompts: Tensor,
    prompt_mask: Tensor,
    responses: Tensor,
    response_masks: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Concatenate grouped prompts with responses for sequence scoring.

    Expands ``prompts`` [B, P] across the group dimension of ``responses``
    [B, G, R] and builds the combined key-padding + causal attention mask.

    Returns:
        ``(full_sequences, attn_mask)`` each shaped [B*G, P + R]; the
        attention mask is 4-D boolean.
    """
    group_size = responses.size(1)
    responses_flat = responses.view(-1, responses.size(-1))
    masks_flat = response_masks.view(-1, responses.size(-1)).bool()
    prompt_expanded = prompts.unsqueeze(1).repeat(1, group_size, 1).flatten(0, 1)
    prompt_mask_expanded = (
        prompt_mask.unsqueeze(1).expand(-1, group_size, -1).flatten(0, 1).bool()
    )

    full_sequences = torch.cat([prompt_expanded, responses_flat], dim=-1)
    # Build full attention mask: key-padding + causal
    key_pad = torch.cat([prompt_mask_expanded, masks_flat], dim=-1)[:, None, None, :]
    S = key_pad.shape[-1]
    causal = torch.tril(
        torch.ones(S, S, dtype=torch.bool, device=full_sequences.device)
    )[None, None, :, :]
    attn_mask = key_pad & causal
    return full_sequences, attn_mask


def rollout_token_logprobs(
    model: nn.Module,
    prompts: Tensor,
    prompt_mask: Tensor,
    responses: Tensor,
    response_masks: Tensor,
    grad_chunked: bool = False,
) -> LogprobsOutput:
    """Per-response-token log probabilities for a grouped rollout batch.

    Prompt tokens are masked out (0) so logprobs are computed only for
    response tokens.  ``get_logprobs`` shifts the mask by one position, so
    the first response token's logprob (predicted from the last prompt
    token) is correctly included.

    Returns:
        ``logprobs`` reshaped to [B, G, R]: position j is the log-probability
        of response token j under ``model``.
    """
    batch_size, group_size, response_len = responses.shape
    prompt_len = prompts.size(1)
    full_sequences, attn_mask = rollout_sequences(
        prompts, prompt_mask, responses, response_masks
    )
    masks_flat = response_masks.view(-1, response_len)
    full_masks = torch.cat(
        [
            torch.zeros(
                batch_size * group_size,
                prompt_len,
                dtype=torch.bool,
                device=full_sequences.device,
            ),
            masks_flat,
        ],
        dim=-1,
    )

    # get_logprobs returns [B*G, S-1] (S = prompt_len + response_len).
    # Response token logprobs occupy the last ``response_len`` positions.
    output = get_logprobs(
        model,
        full_sequences,
        attn_mask,
        full_masks,
        "none",
        grad_chunked=grad_chunked,
    )
    output["logprobs"] = output["logprobs"][:, prompt_len - 1 :].view(
        batch_size, group_size, response_len
    )
    return output


def rollout_token_values(
    model: nn.Module,
    prompts: Tensor,
    prompt_mask: Tensor,
    responses: Tensor,
    response_masks: Tensor,
) -> Tensor:
    """Critic values [B, G, R] aligned with response token positions.

    Position j holds V(s_j) — the value of the state right before response
    token j is emitted — matching the logprob alignment of
    :func:`rollout_token_logprobs`.
    """
    prompt_len = prompts.size(1)
    full_sequences, attn_mask = rollout_sequences(
        prompts, prompt_mask, responses, response_masks
    )
    output = model(full_sequences, input_mask=attn_mask)
    values = output["values"].float()[
        :, prompt_len - 1 : prompt_len - 1 + responses.size(-1)
    ]
    return values.view(responses.shape)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Tensor, Tensor]:
    """Generalized advantage estimation over padded response tokens.

    Args:
        rewards: [B, G, R] per-token rewards; the terminal reward must sit
            at each response's last valid position, padded positions 0.
        values: [B, G, R] rollout-time critic values V(s_t) (see
            :func:`rollout_token_values`).
        mask: [B, G, R] valid-token mask; padded positions are excluded
            and cannot leak into valid advantages.
        gamma: Discount factor.
        gae_lambda: GAE bias/variance trade-off.

    Returns:
        ``(advantages, returns)`` shaped [B, G, R].  The episode ends at the
        last valid token (no bootstrap value beyond truncation).
    """
    response_len = rewards.size(-1)
    flat_rewards = rewards.reshape(-1, response_len)
    flat_mask = mask.reshape(-1, response_len).to(values.dtype)
    # Padded values must be zero or the backward scan would leak them.
    flat_values = values.reshape(-1, response_len) * flat_mask

    advantages = torch.zeros_like(flat_rewards)
    gae = torch.zeros_like(flat_values[:, 0])
    for t in range(response_len - 1, -1, -1):
        next_values = (
            flat_values[:, t + 1]
            if t + 1 < response_len
            else torch.zeros_like(flat_values[:, t])
        )
        delta = flat_rewards[:, t] + gamma * next_values - flat_values[:, t]
        gae = flat_mask[:, t] * (delta + gamma * gae_lambda * gae)
        advantages[:, t] = gae
    returns = advantages + flat_values
    return advantages.view_as(rewards), returns.view_as(rewards)


def _validate_behavior_logprobs(behavior_logprobs: Tensor, responses: Tensor) -> None:
    """Reject behaviour-policy logprobs that do not match the responses."""
    if behavior_logprobs.shape != responses.shape:
        raise ValueError(
            "logprobs_old shape must match responses: "
            f"got {tuple(behavior_logprobs.shape)}, "
            f"expected {tuple(responses.shape)}"
        )
    if not torch.isfinite(behavior_logprobs).all():
        raise ValueError("logprobs_old must contain only finite values")


def _is_packed(position_ids: Tensor) -> bool:
    """Whether rows pack multiple documents (positions reset mid-row)."""
    return bool((position_ids[:, 1:] <= position_ids[:, :-1]).any())


def make_doc_boundary_mask(position_ids: Tensor) -> Tensor:
    S = position_ids.size(1)
    device = position_ids.device
    boundaries = position_ids[:, 1:] <= position_ids[:, :-1]
    doc_ids = torch.cat(
        [
            torch.zeros(position_ids.size(0), 1, dtype=torch.long, device=device),
            boundaries.long().cumsum(dim=1),
        ],
        dim=1,
    )
    same_doc = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
    return (same_doc & causal).unsqueeze(1)


def make_doc_packed_mask(position_ids: Tensor) -> PackedDocumentMask:
    """Describe packed documents for one varlen causal attention call."""
    boundaries = position_ids[:, 1:] <= position_ids[:, :-1]
    doc_ids = torch.cat(
        [
            torch.zeros_like(position_ids[:, :1], dtype=torch.long),
            boundaries.long().cumsum(dim=1),
        ],
        dim=1,
    )
    starts = torch.cat(
        [torch.ones_like(position_ids[:, :1], dtype=torch.bool), boundaries],
        dim=1,
    ).flatten()
    offsets = starts.nonzero().flatten()
    offsets = torch.cat([offsets, offsets.new_tensor([position_ids.numel()])]).to(
        torch.int32
    )
    max_seqlen = int((offsets[1:] - offsets[:-1]).max().item())
    return PackedDocumentMask(doc_ids, offsets, max_seqlen)


def _collect_moe_diagnostics(
    router_stats_list: List[RouterStats],
) -> Dict[str, float]:
    """Collect MoE routing diagnostic metrics from per-layer router stats.

    Args:
        router_stats_list: One :class:`RouterStats` dict per MoE layer with
            keys ``probs`` (N, E) and ``topk_indices`` (N, K), both detached.

    Returns:
        Dict with keys: router_entropy, dead_expert_fraction,
        load_imbalance_mean, load_imbalance_max.  Values are averaged
        across layers.
    """
    layer_entropies: List[Tensor] = []
    layer_dead_fractions: List[Tensor] = []
    layer_imbalance_means: List[Tensor] = []
    layer_imbalance_maxs: List[Tensor] = []

    for stats in router_stats_list:
        probs = stats["probs"].float()
        topk_indices = stats["topk_indices"]
        num_experts = probs.shape[-1]
        if num_experts == 0:
            continue
        probs = probs.reshape(-1, num_experts)
        if probs.numel() == 0:
            continue

        # Router entropy
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()

        # Load from the actual dispatch without a [tokens, top_k, experts] tensor.
        flat_experts = topk_indices.reshape(-1)
        expert_counts = probs.new_zeros(num_experts).scatter_add_(
            0, flat_experts, probs.new_ones(flat_experts.numel())
        )
        ideal_load = expert_counts.mean()  # N*K / E
        load_ratios = expert_counts / max(float(ideal_load), 1.0)
        imbalance_mean = (load_ratios - 1.0).abs().mean()
        imbalance_max = load_ratios.max()
        dead_fraction = (expert_counts == 0).float().mean()

        layer_entropies.append(entropy)
        layer_dead_fractions.append(dead_fraction)
        layer_imbalance_means.append(imbalance_mean)
        layer_imbalance_maxs.append(imbalance_max)

    if not layer_entropies:
        return {}

    return {
        "router_entropy": float(torch.stack(layer_entropies).mean().cpu().item()),
        "dead_expert_fraction": float(
            torch.stack(layer_dead_fractions).mean().cpu().item()
        ),
        "load_imbalance_mean": float(
            torch.stack(layer_imbalance_means).mean().cpu().item()
        ),
        "load_imbalance_max": float(
            torch.stack(layer_imbalance_maxs).mean().cpu().item()
        ),
    }
