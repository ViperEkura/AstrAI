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

    @property
    def preserves_argmax(self) -> bool:
        """Whether ``apply`` never moves the argmax token.

        Conservative default: strategies must opt in. The greedy
        short-circuit in :class:`SamplingPipeline` asks this
        polymorphically, so a new strategy that can move the argmax
        automatically disables it — no isinstance bookkeeping.
        """
        return False

    @property
    def is_greedy(self) -> bool:
        """Whether this strategy collapses sampling onto the argmax token."""
        return False


class TemperatureStrategy(BaseSamplingStrategy):
    """Divides logits by temperature to control randomness.

    Args:
        temperature: Scalar or ``[batch]`` tensor.
    """

    def __init__(self, temperature: Union[float, Tensor] = 1.0):
        self.temperature = temperature

    @staticmethod
    def is_greedy_temperature(temperature: Union[float, Tensor]) -> bool:
        if isinstance(temperature, Tensor):
            return bool((temperature == 0).all())
        return temperature == 0

    @property
    def is_greedy(self) -> bool:
        return self.is_greedy_temperature(self.temperature)

    @property
    def preserves_argmax(self) -> bool:
        # Scaling by a positive constant (1/t, clamped away from zero)
        # preserves logit order; t=0 degenerates onto the argmax itself.
        return True

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

    @property
    def preserves_argmax(self) -> bool:
        # The argmax token always ranks first, so any k >= 1 keeps it.
        return True

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

    @property
    def preserves_argmax(self) -> bool:
        # Nucleus filtering always keeps the highest-probability token.
        return True

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
            p = p.to(logits.device, non_blocking=True).view(-1)
            if (p == 0.0).all():
                return logits
        elif p == 0.0:
            return logits

        input_ids = input_ids.to(logits.device, non_blocking=True)
        if input_mask is not None:
            input_mask = input_mask.to(logits.device, non_blocking=True)

        batch_sz = input_ids.shape[0]
        vocab_size = logits.size(-1)

        # Sync-free update: map each history token to a flat
        # ``row * vocab + token`` bucket (padding to one trailing sentinel
        # bucket), count with ``index_add_``, and subtract in one
        # elementwise pass. No nonzero/unique/boolean-mask indexing, so the
        # hot path never forces a device-host synchronization.
        row_offsets = (
            torch.arange(batch_sz, device=logits.device, dtype=torch.long).unsqueeze(1)
            * vocab_size
        )
        if input_mask is not None:
            flat = torch.where(
                input_mask,
                row_offsets + input_ids,
                torch.full_like(input_ids, batch_sz * vocab_size),
            )
        else:
            flat = row_offsets + input_ids
        flat = flat.reshape(-1)
        counts = torch.zeros(
            batch_sz * vocab_size + 1, device=logits.device, dtype=torch.float32
        )
        counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        counts = counts[: batch_sz * vocab_size].view(batch_sz, vocab_size)
        if isinstance(p, Tensor):
            deltas = counts * p.to(torch.float32).view(-1, 1)
        else:
            deltas = counts * float(p)
        return logits - deltas.to(logits.dtype)


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

    @property
    def preserves_argmax(self) -> bool:
        # A composite preserves the argmax iff every stage does.
        return all(s.preserves_argmax for s in self.strategies)

    @property
    def is_greedy(self) -> bool:
        """Whether sampling always yields the argmax of the raw logits.

        True iff some stage forces greedy and no stage can move the
        argmax before or after it. Both facts are declared
        polymorphically by each strategy, so composing in a new strategy
        type (or a nested pipeline) updates this automatically.
        """
        return any(s.is_greedy for s in self.strategies) and self.preserves_argmax

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

    @torch.inference_mode()
    def sample(
        self,
        logits: Tensor,
        filter_value: float = -float("inf"),
        input_ids: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
        return_logprobs: bool = False,
    ):
        """Apply strategies then sample (softmax + multinomial).

        Short-circuits to ``argmax`` when temperature is exactly 0
        (deterministic / greedy decode).

        Args:
            logits: Raw logits ``[batch, vocab_size]``.
            input_ids: Previously generated token IDs ``[batch, seq_len]``.
            input_mask: Boolean mask for ``input_ids`` padding.
            return_logprobs: If ``True``, return ``(tokens, logprobs)``
                where ``logprobs[i]`` is the log-probability of
                ``tokens[i]`` under the raw (pre-strategy) model
                distribution, matching training-side policy logprobs.

        Returns:
            Sampled token IDs ``[batch]``, or — when ``return_logprobs``
            is ``True`` — a ``(token_ids, chosen_logprobs)`` tuple.
        """
        if self.is_greedy:
            tokens = logits.argmax(dim=-1)
            if not return_logprobs:
                return tokens
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            chosen = torch.gather(log_probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
            return tokens, chosen

        # Capture the raw distribution before the strategy pipeline runs:
        # top-k/top-p mutate the logits tensor in place, so computing this
        # after ``apply`` would read the filtered distribution instead of
        # the raw model distribution the caller documented.
        if return_logprobs:
            raw_log_probs = torch.log_softmax(logits.float(), dim=-1)

        transformed = self.apply(logits, filter_value, input_ids, input_mask)
        tokens = torch.multinomial(
            torch.softmax(transformed, dim=-1), num_samples=1
        ).squeeze(-1)
        if not return_logprobs:
            return tokens
        # Log-probabilities of the raw (pre-strategy) model distribution,
        # matching the training-side policy logprobs exactly: the behaviour
        # logprobs recorded for online RL must live in the same
        # distribution the trainer differentiates, not the
        # temperature/top-p filtered one tokens were drawn from.
        chosen = torch.gather(raw_log_probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        return tokens, chosen


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
):
    """Apply sampling strategies then sample (softmax + multinomial).

    Shortcut for ``SamplingPipeline(...).sample(logits, return_logprobs=)``.

    When **temperature** is exactly 0 (scalar or single-element tensor)
    the function short-circuits to ``argmax`` for deterministic decode.

    When **frequency_penalty** is 0 (the common decode case), the entire
    frequency penalty computation — including the O(batch * vocab) count
    tensor allocation — is skipped.

    Args:
        logits: Raw logits ``[batch, vocab_size]``.
        frequency_penalty: Penalty per occurrence for repeated tokens
            (0.0 disables, range -2.0~2.0).
        input_ids: Previously generated token IDs ``[batch, seq_len]``.
        input_mask: Boolean mask for ``input_ids`` padding.
        return_logprobs: If ``True``, also return the log-probability
            of each sampled token under the raw (pre-strategy) model
            distribution — usable directly for RL rollout (PPO/GRPO
            importance ratios against the training-side policy logprobs).

    Returns:
        Sampled token IDs ``[batch]``, or — when ``return_logprobs`` is
        ``True`` — a ``(token_ids, chosen_logprobs)`` tuple where
        ``chosen_logprobs`` has shape ``[batch]``.
    """
    has_freq = (
        (isinstance(frequency_penalty, Tensor) and (frequency_penalty != 0).any())
        if isinstance(frequency_penalty, Tensor)
        else frequency_penalty != 0
    )

    strategies: List[BaseSamplingStrategy] = []
    if has_freq:
        # Penalty first, on the raw logits (OpenAI semantics): applying it
        # after a temperature scaling would shrink it by the temperature
        # and annihilate it entirely at temperature=0.
        strategies.append(FrequencyPenaltyStrategy(frequency_penalty))
    strategies.extend(
        [
            TemperatureStrategy(temperature),
            TopKStrategy(top_k),
            TopPStrategy(top_p),
        ]
    )

    return SamplingPipeline(strategies).sample(
        logits,
        filter_value=filter_value,
        input_ids=input_ids,
        input_mask=input_mask,
        return_logprobs=return_logprobs,
    )
