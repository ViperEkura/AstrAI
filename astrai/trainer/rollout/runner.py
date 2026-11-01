"""Score, cache, and evaluate online training rollouts."""

from typing import TYPE_CHECKING, Callable, Dict, Iterator, Optional, Sequence, Tuple

import torch
from torch import Tensor

from astrai.trainer.backend import WeightPublisher
from astrai.trainer.rollout.generator import RolloutGenerator
from astrai.trainer.rollout.types import (
    BaseRewardModel,
    RawRollout,
    RolloutResult,
    RolloutVersionError,
    SamplingParams,
    T,
)

if TYPE_CHECKING:
    from astrai.trainer.rollout.async_round import AsyncRoundCoordinator


class RolloutRunner:
    """Produces :class:`RolloutResult` from a prompt batch.

    Composes a :class:`RolloutGenerator` (generation + decoding) with a
    :class:`BaseRewardModel` (scoring).  Maintains an internal cache so
    the same batch prompt can be replayed for multiple gradient steps.
    A new rollout is triggered every ``rollout_interval`` calls to
    :meth:`step` (or after :meth:`clear_cache`).

    The ``__call__`` contract returns a ``(RolloutResult, is_fresh)``
    tuple — callers must use the boolean to detect a refreshed rollout
    rather than relying on object identity.

    Usage::

        generator = RolloutGenerator(scheduler, tokenizer, SamplingParams(...))
        runner = RolloutRunner(generator, reward_model, rollout_interval=512)
        result, is_fresh = runner(prompt_batch)
        if is_fresh:
            ...  # e.g. sync behaviour policy
    """

    def __init__(
        self,
        generator: Optional[RolloutGenerator] = None,
        reward_model: Optional[BaseRewardModel] = None,
        rollout_interval: int = 512,
        max_policy_lag: Optional[int] = None,
        coordinator: Optional["AsyncRoundCoordinator"] = None,
        publishers: Sequence[WeightPublisher] = (),
    ):
        if (coordinator is None and (generator is None or reward_model is None)) or (
            coordinator is not None
            and (generator is not None or reward_model is not None)
        ):
            raise ValueError("provide either generator and reward_model or coordinator")
        self.coordinator = coordinator
        self._publishers = tuple(publishers)
        if rollout_interval <= 0:
            raise ValueError("rollout_interval must be positive")
        if max_policy_lag is not None and max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative or None")
        self.generator = generator
        self.reward_model = reward_model
        self.rollout_interval = rollout_interval
        self.max_policy_lag = (
            rollout_interval - 1 if max_policy_lag is None else max_policy_lag
        )

        self._cache: Optional[RolloutResult] = None
        self._cache_key = None
        self._steps_since_rollout: int = 0

    @property
    def policy_version(self) -> int:
        return (
            self.coordinator.policy_version
            if self.coordinator is not None
            else self.generator.policy_version
        )

    def update_weights(self, policy_version: int) -> int:
        """Publish the shared policy's new version to the rollout backend."""
        if self.coordinator is not None:
            raise RuntimeError("async rollout publishes weights during optimizer step")
        return self.generator.update_weights(policy_version)

    def apply_weight_update(
        self, policy_version: Optional[int], update: Callable[[int], T]
    ) -> T:
        """Apply a model update and publish its version as one operation."""
        if self.coordinator is not None:
            return self.coordinator.apply_weight_update(policy_version, update)
        return self.generator.apply_weight_update(policy_version, update)

    def publish_weights(self, version: int, model) -> None:
        """Copy replicas inside the caller's apply_weight_update lock."""
        for publisher in self._publishers:
            publisher.publish(version, model)

    def step(self):
        """Advance the internal counter (call once per optimizer step)."""
        if self.coordinator is None:
            self._steps_since_rollout += 1

    def iter_rounds(
        self, batches, stop_requested: Callable[[], bool]
    ) -> Iterator[RolloutResult]:
        if self.coordinator is None:
            raise RuntimeError("iter_rounds requires an async rollout coordinator")
        return self.coordinator.iter_rounds(batches, stop_requested)

    def close(self) -> None:
        if self.coordinator is not None:
            self.coordinator.close()

    def clear_cache(self):
        """Force next call to re-run rollout."""
        self._cache = None
        self._cache_key = None

    @staticmethod
    def _batch_key(batch: Dict):
        """Build a stable key for the prompt fields accepted by the generator."""

        def freeze(value):
            if isinstance(value, dict):
                return tuple(sorted((key, freeze(val)) for key, val in value.items()))
            if isinstance(value, (list, tuple)):
                return tuple(freeze(item) for item in value)
            return value

        fields = ("messages", "instruction", "input", "output")
        return tuple(
            (field, freeze(batch[field])) for field in fields if field in batch
        )

    def _score(self, raw: RawRollout) -> RolloutResult:
        rewards = _score_rewards(self.reward_model, raw)
        return RolloutResult.from_raw(raw, rewards.to(device=raw.prompts.device))

    def _validate_policy_version(
        self, result: RawRollout, *, live_version: Optional[int] = None
    ) -> None:
        version = result.policy_version
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise RolloutVersionError(f"rollout has invalid policy version {version!r}")
        if live_version is None:
            live_version = self.policy_version
        if version > live_version:
            raise RolloutVersionError(
                f"rollout has future policy version {version}; "
                f"live policy version is {live_version}"
            )
        lag = live_version - version
        if lag > self.max_policy_lag:
            raise RolloutVersionError(
                f"rollout policy lag {lag} exceeds max_policy_lag="
                f"{self.max_policy_lag} (rollout={version}, live={live_version})"
            )

    def __call__(self, batch: Dict[str, Tensor]) -> Tuple[RolloutResult, bool]:
        """Return ``(cached or fresh) RolloutResult`` plus an ``is_fresh`` flag.

        Triggers a new rollout when ``_steps_since_rollout >= rollout_interval``
        or when the cache is empty. The reuse decision, its version
        validation, and the returned object are all captured inside one
        policy snapshot, so a concurrent commit, refresh, or cache clear
        can never hand out an object the snapshot has already invalidated.
        """
        if self.coordinator is not None:
            raise RuntimeError("async rollouts use iter_rounds")
        cache_key = self._batch_key(batch)

        def reuse(live_version: int) -> Optional[Tuple[RolloutResult, bool]]:
            cached = self._cache
            if (
                cached is None
                or self._cache_key != cache_key
                or self._steps_since_rollout >= self.rollout_interval
            ):
                return None
            self._validate_policy_version(cached, live_version=live_version)
            return cached, False

        outcome = self.generator.with_policy_snapshot(reuse)
        if outcome is not None:
            return outcome

        raw = self.generator.generate(batch)
        self._validate_policy_version(raw)
        scored = self._score(raw)
        # Post-scoring check: reward scoring may call slow external services;
        # surface an over-lag policy move before the commit critical section.
        self._validate_policy_version(scored)

        def commit(live_version: int) -> Tuple[RolloutResult, bool]:
            self._validate_policy_version(scored, live_version=live_version)
            self._cache = scored
            self._cache_key = cache_key
            self._steps_since_rollout = 0
            return scored, True

        # A weight update cannot land between the final version check and
        # cache publication. Reward scoring itself intentionally remains
        # outside the policy lock because it may call an external service.
        return self.generator.with_policy_snapshot(commit)

    def evaluate(
        self, batch: Dict, params: Optional[SamplingParams] = None
    ) -> RolloutResult:
        """One-off rollout + scoring that leaves the replay cache untouched.

        Used by validation on online strategies: the training cache, its
        cadence counter, and the cache key stay intact, so evaluation
        prompts never disturb the rollout replay schedule.  ``params``
        overrides the generator's training sampling defaults for this
        call only.
        """
        if self.coordinator is not None:
            raise RuntimeError("async validation is unsupported")
        raw = self.generator.generate(batch, params)
        self._validate_policy_version(raw)
        scored = self._score(raw)
        self._validate_policy_version(scored)
        return scored


class RolloutEvaluator:
    """Validation-time rollout scoring with its own sampling configuration.

    Unlike :meth:`RolloutRunner.evaluate` — a one-off rollout on the
    *training* runner, still scored as an RL loss — the evaluator owns
    its :class:`SamplingParams` outright (typically greedy, with a
    val-specific group size) and reports reward statistics instead.  The
    RL loss is degenerate as a validation signal under greedy decoding
    or ``group_size == 1`` (zero group advantage), so it is not computed.
    """

    def __init__(
        self,
        generator: RolloutGenerator,
        reward_model: BaseRewardModel,
        params: SamplingParams,
    ):
        self.generator = generator
        self.reward_model = reward_model
        self.params = params

    def evaluate(self, batch: Dict) -> Dict[str, float]:
        """Generate + score one batch; return scalar validation metrics."""
        raw = self.generator.generate(batch, self.params)
        rewards = _score_rewards(self.reward_model, raw)
        lengths = raw.response_mask.sum(dim=-1).to(torch.float32)
        std = rewards.std(unbiased=False).item() if rewards.numel() > 1 else 0.0
        metrics = {
            "reward_mean": rewards.mean().item(),
            "reward_std": std,
            "reward_max": rewards.max().item(),
            "response_len_mean": lengths.mean().item(),
            "num_responses": float(rewards.numel()),
        }
        flat_reasons = [reason for group in raw.finish_reasons for reason in group]
        if flat_reasons:
            metrics["truncation_rate"] = sum(
                1 for reason in flat_reasons if reason == "length"
            ) / len(flat_reasons)
        return metrics


def _score_rewards(reward_model: BaseRewardModel, raw: RawRollout) -> Tensor:
    """Score a rollout's decoded responses, validating shape and finiteness."""
    rewards = reward_model.score(raw.prompt_texts, raw.response_texts)
    if not isinstance(rewards, Tensor):
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
    expected_shape = raw.responses.shape[:2]
    if rewards.shape != expected_shape:
        raise ValueError(
            f"Reward model returned shape {tuple(rewards.shape)}, "
            f"expected {tuple(expected_shape)}"
        )
    if not torch.isfinite(rewards).all():
        raise ValueError("Reward model returned non-finite values")
    return rewards
