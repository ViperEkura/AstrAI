"""Composable sampling strategies for logit transformation.

Implements the Strategy pattern: each sampling technique
(temperature, top-k, top-p, frequency penalty) is a pluggable
strategy that can be composed into a pipeline.

All strategies accept both scalar and per-sample tensor
parameters, so a single pipeline works for any batch size.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
from torch import Tensor


class BaseSamplingStrategy(ABC):
    """Abstract base for a logit transformation strategy."""

    @abstractmethod
    def apply(
        self,
        logits: Tensor,
        filter_value: float = -float("inf"),
        input_ids: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Applies the strategy to logits.

        Args:
            logits: Raw logits tensor (batch, vocab_size).
            filter_value: Value assigned to filtered-out positions.
            input_ids: Previously generated token IDs ``[batch, seq_len]``,
                padded with 0. Used by frequency penalty.
            input_mask: Boolean mask ``[batch, seq_len]``, True for real
                tokens, False for padding. Used to exclude padding from
                penalty computation.

        Returns:
            Transformed logits tensor.
        """
        raise NotImplementedError


class TemperatureStrategy(BaseSamplingStrategy):
    """Divides logits by temperature to control randomness.

    Args:
        temperature: Scalar or ``[batch]`` tensor.
    """

    def __init__(self, temperature: Union[float, Tensor] = 1.0):
        self.temperature = temperature

    def apply(
        self,
        logits: Tensor,
        filter_value: float = -float("inf"),
        input_ids: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
    ) -> Tensor:
        t = self.temperature
        if isinstance(t, Tensor):
            t = t.to(logits.device, non_blocking=True).view(-1, 1)
            t = torch.clamp(t, min=1e-8)
            if (t != 1.0).any():
                logits = logits / t
        elif t != 1.0:
            logits = logits / max(t, 1e-8)
        return logits


class TopKStrategy(BaseSamplingStrategy):
    """Keeps only the top-k logits, setting the rest to filter_value.

    Args:
        top_k: Scalar or ``[batch]`` tensor (0 disables).
    """

    def __init__(self, top_k: Union[int, Tensor] = 0):
        self.top_k = top_k

    def apply(
        self,
        logits: Tensor,
        filter_value: float = -float("inf"),
        input_ids: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tk = self.top_k
        if isinstance(tk, Tensor):
            tk = tk.to(logits.device, non_blocking=True).long().clamp(min=0)
            max_k = int(tk.max().item())
            if max_k <= 0:
                return logits
            max_k = min(max_k, logits.size(-1))
            values, _ = torch.topk(logits, max_k, dim=-1)
            per_row_k = tk.clamp(max=max_k)
            thresholds = torch.full_like(logits[..., -1:], -float("inf"))
            positive = per_row_k > 0
            if positive.any():
                row_idx = torch.arange(logits.size(0), device=logits.device)[positive]
                thresholds[positive] = values[
                    row_idx, per_row_k[positive] - 1
                ].unsqueeze(-1)
            logits[logits < thresholds] = filter_value
            return logits
        if tk > 0:
            k = min(tk, logits.size(-1))
            thresholds = torch.topk(logits, k, dim=-1)[0][..., -1:]
            logits[logits < thresholds] = filter_value
        return logits


class TopPStrategy(BaseSamplingStrategy):
    """Nucleus (top-p) filtering: keeps the smallest set of tokens whose
    cumulative probability exceeds top_p.

    Args:
        top_p: Scalar or ``[batch]`` tensor (1.0 disables).
    """

    def __init__(self, top_p: Union[float, Tensor] = 1.0):
        self.top_p = top_p

    def _apply(
        self, logits: Tensor, top_p: Union[float, Tensor], filter_value: float
    ) -> Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, sorted_indices, remove)
        logits[mask] = filter_value
        return logits

    def apply(
        self,
        logits: Tensor,
        filter_value: float = -float("inf"),
        input_ids: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tp = self.top_p
        if isinstance(tp, Tensor):
            tp = tp.to(logits.device, non_blocking=True)
            if (tp < 1.0).any():
                logits = self._apply(logits, tp.view(-1, 1), filter_value)
        elif tp < 1.0:
            logits = self._apply(logits, tp, filter_value)
        return logits


class FrequencyPenaltyStrategy(BaseSamplingStrategy):
    """Penalizes tokens based on how many times they appeared in history.

    Subtracts ``penalty * count(token)`` from each token's logit, where
    ``count(token)`` is the number of occurrences in the generation history
    (prompt + output). A penalty of ``0.0`` disables the strategy.

    Unlike repetition penalty (which only checks *presence*), frequency
    penalty scales linearly with occurrence count: the first use is
    penalized once, the third use three times. This allows natural
    repetition of common words while suppressing degenerate loops.

    Reference: OpenAI API ``frequency_penalty`` parameter.

    Args:
        penalty: Scalar or ``[batch]`` tensor (0.0 disables, range -2.0~2.0).
    """

    def __init__(self, penalty: Union[float, Tensor] = 0.0):
        self.penalty = penalty

    def apply(
        self,
        logits: Tensor,
        filter_value: float = -float("inf"),
        input_ids: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if input_ids is None:
            return logits

        p = self.penalty
        if isinstance(p, Tensor):
            p = p.to(logits.device, non_blocking=True).view(-1, 1)
            if (p == 0.0).all():
                return logits
        elif p == 0.0:
            return logits

        input_ids = input_ids.to(logits.device, non_blocking=True)

        if input_mask is not None:
            input_mask = input_mask.to(logits.device, non_blocking=True)
            masked_ids = input_ids.clone()
            masked_ids[~input_mask] = -1
        else:
            masked_ids = input_ids

        batch_sz, seq_len = masked_ids.shape
        vocab_size = logits.size(-1)

        if isinstance(p, Tensor):
            penalty_per_row = p.expand(batch_sz, 1)
        else:
            penalty_per_row = torch.full(
                (batch_sz, 1), float(p), device=logits.device, dtype=logits.dtype
            )

        counts = torch.zeros(
            batch_sz, vocab_size, device=logits.device, dtype=logits.dtype
        )
        valid_mask = masked_ids >= 0
        if valid_mask.any():
            valid_ids = masked_ids[valid_mask]
            row_indices = (
                torch.arange(batch_sz, device=logits.device)
                .unsqueeze(1)
                .expand_as(masked_ids)[valid_mask]
            )
            counts.index_put_(
                (row_indices, valid_ids),
                torch.ones_like(valid_ids, dtype=logits.dtype),
                accumulate=True,
            )

        return logits - penalty_per_row * counts


class SamplingPipeline(BaseSamplingStrategy):
    """Composes multiple sampling strategies into a single transformation.

    Strategies are applied sequentially in the order they are provided,
    matching the original temperature -> top-k -> top-p ordering.

    Usage::

        pipeline = SamplingPipeline([
            TemperatureStrategy(0.8),
            TopKStrategy(50),
            TopPStrategy(0.95),
        ])
        logits = pipeline.apply(logits)
        token = pipeline.sample(logits)       # softmax + multinomial
    """

    def __init__(self, strategies: List[BaseSamplingStrategy]):
        self.strategies = strategies

    def apply(
        self,
        logits: Tensor,
        filter_value: float = -float("inf"),
        input_ids: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
    ) -> Tensor:
        for strategy in self.strategies:
            logits = strategy.apply(logits, filter_value, input_ids, input_mask)
        return logits

    @staticmethod
    def _is_greedy(temperature: Union[float, Tensor]) -> bool:
        if isinstance(temperature, Tensor):
            return temperature.numel() == 1 and temperature.item() == 0
        return temperature == 0

    @torch.inference_mode()
    def sample(
        self,
        logits: Tensor,
        filter_value: float = -float("inf"),
        input_ids: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply strategies then sample (softmax + multinomial).

        Short-circuits to ``argmax`` when temperature is exactly 0
        (deterministic / greedy decode).

        Args:
            logits: Raw logits ``[batch, vocab_size]``.
            input_ids: Previously generated token IDs ``[batch, seq_len]``.
            input_mask: Boolean mask for ``input_ids`` padding.

        Returns:
            Sampled token IDs ``[batch]``.
        """
        for s in self.strategies:
            if isinstance(s, TemperatureStrategy) and self._is_greedy(s.temperature):
                return logits.argmax(dim=-1)
            break

        return torch.multinomial(
            torch.softmax(
                self.apply(logits, filter_value, input_ids, input_mask), dim=-1
            ),
            num_samples=1,
        ).squeeze(-1)


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
) -> Tensor:
    """Apply sampling strategies then sample (softmax + multinomial).

    Shortcut for ``SamplingPipeline(...).sample(logits)``.

    When **temperature** is exactly 0 (scalar or single-element tensor)
    the function short-circuits to ``argmax`` for deterministic decode.

    Args:
        logits: Raw logits ``[batch, vocab_size]``.
        frequency_penalty: Penalty per occurrence for repeated tokens
            (0.0 disables, range -2.0~2.0).
        input_ids: Previously generated token IDs ``[batch, seq_len]``.
        input_mask: Boolean mask for ``input_ids`` padding.

    Returns:
        Sampled token IDs ``[batch]``.
    """
    if SamplingPipeline._is_greedy(temperature):
        return logits.argmax(dim=-1)
    return SamplingPipeline(
        [
            TemperatureStrategy(temperature),
            TopKStrategy(top_k),
            TopPStrategy(top_p),
            FrequencyPenaltyStrategy(frequency_penalty),
        ]
    ).sample(logits, filter_value, input_ids, input_mask)
