"""Sampling with cached host metadata and optional token log probabilities."""

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
    has_greedy: bool


_MASK64 = (1 << 64) - 1


def _seed_uniforms(seeds: List[Optional[int]]) -> Tensor:
    """Map each seed to a uniform in [0, 1) with splitmix64; None rows use the global RNG."""
    values = []
    for seed in seeds:
        if seed is None:
            values.append(float("nan"))
            continue
        z = (seed + 0x9E3779B97F4A7C15) & _MASK64
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        values.append(((z ^ (z >> 31)) >> 11) * 2.0**-53)
    uniforms = torch.tensor(values, dtype=torch.float64)
    missing = torch.isnan(uniforms)
    if missing.any():
        uniforms[missing] = torch.rand(int(missing.sum()), dtype=torch.float64)
    return uniforms


@torch.inference_mode()
def sample(
    logits: Tensor,
    temperature: Union[float, Tensor] = 1.0,
    top_k: Union[int, Tensor] = 0,
    top_p: Union[float, Tensor] = 1.0,
    frequency_penalty: Union[float, Tensor] = 0.0,
    frequency_counts: Optional[Tensor] = None,
    filter_value: float = -float("inf"),
    return_logprobs: bool = False,
    meta: Optional[SamplingMeta] = None,
    seeds: Optional[List[Optional[int]]] = None,
):
    """Apply frequency penalty, temperature, top-k and top-p, then sample.

    Positive-temperature rows report scores after penalties and temperature,
    before top-k/top-p. Greedy rows report raw model log probabilities.
    The caller supplies frequency counts. Per-request seeds leave the global
    RNG unchanged.
    The worker supplies cached metadata; standalone callers may use scalars
    or per-row tensors without it.
    """
    if not isinstance(return_logprobs, bool):
        raise TypeError("return_logprobs must be a bool")
    if meta is None:
        meta = SamplingMeta(
            greedy=bool((temperature == 0).all())
            if isinstance(temperature, Tensor)
            else temperature == 0,
            has_greedy=bool((temperature == 0).any())
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
    raw_logits = logits
    if not meta.greedy or meta.has_freq:
        logits = logits.float()
    if meta.has_freq and frequency_counts is not None:
        penalty = (
            frequency_penalty.to(device=logits.device, dtype=torch.float32).view(-1, 1)
            if isinstance(frequency_penalty, Tensor)
            else float(frequency_penalty)
        )
        logits = logits - frequency_counts * penalty

    greedy_rows = None
    if not meta.greedy and meta.any_temp_not_one:
        if isinstance(temperature, Tensor):
            temp = temperature.to(device=logits.device).view(-1, 1)
            if meta.has_greedy:
                greedy_rows = temp == 0
                greedy_tokens = logits.argmax(dim=-1)
                temp = temp.masked_fill(greedy_rows, 1.0)
            logits = logits / temp.clamp(min=1e-8)
        else:
            logits = logits / max(temperature, 1e-8)

    if return_logprobs:
        scores = raw_logits if meta.greedy else logits
        if greedy_rows is not None:
            scores = torch.where(greedy_rows, raw_logits, scores)
        log_probs = scores.float().log_softmax(-1)

    if meta.greedy:
        tokens = logits.argmax(dim=-1)
    else:
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
            # One stateless uniform per row and a batched inverse-CDF draw, instead
            # of a Generator plus multinomial launch per row on every decode step.
            uniforms = _seed_uniforms(seeds).to(logits.device, non_blocking=True)
            cdf = probabilities.double().cumsum(dim=-1)
            target = (uniforms * cdf[:, -1]).unsqueeze(-1)
            tokens = torch.searchsorted(cdf, target, right=True).squeeze(-1)
            tokens = tokens.clamp_(max=probabilities.size(-1) - 1)
        if greedy_rows is not None:
            tokens = torch.where(greedy_rows.squeeze(-1), greedy_tokens, tokens)
    if return_logprobs:
        return tokens, log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens
