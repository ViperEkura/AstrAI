"""Data contracts and configuration for online training rollouts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TypeVar

from torch import Tensor


@dataclass(kw_only=True)
class RawRollout:
    """Generation output before reward scoring.

    Produced by :class:`RolloutGenerator`; consumed by :class:`RolloutRunner`
    to assemble a :class:`RolloutResult` once rewards are attached.

    Fields are designed to cover all common RL algorithms:
    GRPO, PPO, Online DPO, Rejection Sampling, etc.

    Fields:
        prompts: Tokenized prompts, shape ``[B, P_len]``.
        prompt_mask: Boolean mask for real prompt tokens, shape ``[B, P_len]``.
        responses: Generated response token IDs, shape ``[B, G, R_max]``.
        response_mask: Boolean mask for real (non-pad) response tokens,
            shape ``[B, G, R_max]``.
        logprobs_old: Per-token log-probs under the behaviour policy,
            shape ``[B, G, R_max]``.
        prompt_texts: Decoded prompt strings (for reward models that
            need text).
        response_texts: Decoded response strings, shape ``[B, G]``
            (for reward models).
        finish_reasons: Scheduler termination reason per response
            (``"stop"`` or ``"length"``), shape ``[B][G]``.  Truncated
            responses must not be mistaken for finished episodes: PPO
            treats the last valid token as terminal either way, but the
            truncation rate is the observable that says the max-token
            budget — not the policy — ended the episode.
    """

    prompts: Tensor
    prompt_mask: Tensor
    responses: Tensor
    response_mask: Tensor
    logprobs_old: Tensor
    policy_version: int = 0
    prompt_texts: List[str] = field(default_factory=list)
    response_texts: List[List[str]] = field(default_factory=list)
    finish_reasons: List[List[str]] = field(default_factory=list)


@dataclass(kw_only=True)
class RolloutResult(RawRollout):
    """A :class:`RawRollout` with reward scoring attached.

    Produced by :class:`RolloutRunner` once the :class:`BaseRewardModel`
    has scored the decoded responses.

    Fields:
        rewards: Reward per response, shape ``[B, G]``.
        advantages: Optional GAE advantages ``[B, G, R_max]`` pinned at
            rollout time by actor-critic strategies (PPO).  ``None`` until
            a strategy computes them.
        returns: Optional GAE value targets ``[B, G, R_max]`` matching
            ``advantages``.
    """

    rewards: Tensor
    advantages: Optional[Tensor] = None
    returns: Optional[Tensor] = None


class BaseRewardModel(ABC):
    """Pluggable reward model interface.

    Subclasses should implement ``score()`` to return a ``[B, G]`` float
    tensor of rewards.  Implementations can be:
    * A loaded reward model (e.g. ArmoRM, Skywork-Reward)
    * An external API call
    * A rule-based function (format, length, keyword matching)
    """

    @abstractmethod
    def score(self, prompts: List[str], responses: List[List[str]]) -> Tensor:
        """Score each generated response.

        Args:
            prompts: Raw prompt strings, length ``B``.
            responses: Generated response strings, shape ``[B, G]``.

        Returns:
            Float tensor of shape ``[B, G]``.
        """
        ...


_PAD = 0
T = TypeVar("T")


class RolloutVersionError(RuntimeError):
    """A rollout cannot be attributed to an acceptable policy version."""


@dataclass(frozen=True)
class SamplingParams:
    """Sampling configuration for one rollout call.

    A :class:`RolloutGenerator` holds the *training* defaults; validation
    (and any other caller) derives its own instance via
    :func:`dataclasses.replace` so only the fields that differ from the
    training rollout need to be stated — ``temperature=0.0`` selects the
    greedy decode path.

    ``seed`` binds independent sampling streams to tokenized prompt,
    policy version, response index and duplicate occurrence within the batch.
    For unique prompts the stream survives row reordering and batch splitting.
    ``None`` keeps the shared multinomial RNG path.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 1024
    group_size: int = 8
    frequency_penalty: float = 0.0
    rep_window: int = 64
    seed: Optional[int] = None
