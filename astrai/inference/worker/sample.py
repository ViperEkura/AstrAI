"""Sampling with cached host metadata and raw-model log probabilities."""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingMeta:
    """Batch facts resolved on the host to avoid decode-time GPU synchronization."""

    greedy: bool
    any_temp_not_one: bool
    max_top_k: int
    any_topp_lt1: bool
    has_freq: bool


@torch.inference_mode()
def sample(
    logits: Tensor,
    temperature: Union[float, Tensor] = 1.0,
    top_k: Union[int, Tensor] = 0,
    top_p: Union[float, Tensor] = 1.0,
    frequency_penalty: Union[float, Tensor] = 0.0,
    input_ids: Optional[Tensor] = None,
    input_mask: Optional[Tensor] = None,
    filter_value: float = -float("inf"),
    return_logprobs: bool = False,
    meta: Optional[SamplingMeta] = None,
    seeds: Optional[List[Optional[int]]] = None,
):
    """Apply frequency penalty, temperature, top-k and top-p, then sample.

    Returned log probabilities use the original model distribution, before
    penalties or filtering. Per-request seeds do not advance the global RNG.
    The worker supplies cached metadata; standalone callers may use scalars
    or per-row tensors without it.
    """
    if meta is None:
        meta = SamplingMeta(
            greedy=bool((temperature == 0).all())
            if isinstance(temperature, Tensor)
            else temperature == 0,
            any_temp_not_one=bool((temperature != 1).any())
            if isinstance(temperature, Tensor)
            else temperature != 1,
            max_top_k=max(0, int(top_k.max()))
            if isinstance(top_k, Tensor)
            else max(0, top_k),
            any_topp_lt1=bool((top_p < 1).any())
            if isinstance(top_p, Tensor)
            else top_p < 1,
            has_freq=bool((frequency_penalty != 0).any())
            if isinstance(frequency_penalty, Tensor)
            else frequency_penalty != 0,
        )
    raw_log_probs = (
        torch.log_softmax(logits.float(), dim=-1) if return_logprobs else None
    )
    if meta.greedy and not meta.has_freq:
        tokens = logits.argmax(dim=-1)
    else:
        if meta.has_freq and input_ids is not None:
            ids = input_ids.to(logits.device, non_blocking=True)
            batch, vocab = logits.shape
            offsets = torch.arange(batch, device=logits.device).unsqueeze(1) * vocab
            flat = offsets + ids
            if input_mask is not None:
                flat = torch.where(
                    input_mask.to(logits.device, non_blocking=True),
                    flat,
                    torch.full_like(ids, batch * vocab),
                )
            flat = flat.reshape(-1)
            counts = torch.zeros(
                batch * vocab + 1, device=logits.device, dtype=torch.float32
            )
            counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
            counts = counts[: batch * vocab].view(batch, vocab)
            penalty = (
                frequency_penalty.to(device=logits.device, dtype=torch.float32).view(
                    -1, 1
                )
                if isinstance(frequency_penalty, Tensor)
                else float(frequency_penalty)
            )
            logits = logits - (counts * penalty).to(logits.dtype)
        if meta.any_temp_not_one:
            temp = (
                temperature.to(logits.device, non_blocking=True)
                .view(-1, 1)
                .clamp(min=1e-8)
                if isinstance(temperature, Tensor)
                else max(temperature, 1e-8)
            )
            logits = logits / temp
        k = min(meta.max_top_k, logits.size(-1))
        if k > 0:
            values = torch.topk(logits, k, dim=-1).values
            if isinstance(top_k, Tensor):
                ks = top_k.to(logits.device, non_blocking=True).clamp(min=0, max=k)
                thresholds = values.gather(
                    -1, (ks - 1).clamp(min=0).long().unsqueeze(-1)
                )
                thresholds = torch.where(
                    ks.unsqueeze(-1) > 0, thresholds, -float("inf")
                )
            else:
                thresholds = values[..., -1:]
            logits[logits < thresholds] = filter_value
        if meta.any_topp_lt1:
            ps = (
                top_p.to(logits.device, non_blocking=True).view(-1, 1)
                if isinstance(top_p, Tensor)
                else top_p
            )
            sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)
            remove = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > ps
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                1, indices, remove
            )
            logits[mask] = filter_value
        probabilities = torch.softmax(logits, dim=-1)
        if seeds is None:
            tokens = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
        else:
            if len(seeds) != logits.shape[0]:
                raise ValueError("sampling seeds must match the batch size")
            tokens = torch.cat(
                [
                    torch.multinomial(
                        probabilities[i],
                        num_samples=1,
                        generator=None
                        if seed is None
                        else torch.Generator(device=logits.device).manual_seed(seed),
                    )
                    for i, seed in enumerate(seeds)
                ]
            )
    if return_logprobs:
        return tokens, raw_log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens
