"""Online rollout runner for RL training.

Provides:
- :class:`RawRollout` — generation output container (no reward yet)
- :class:`RolloutResult` — a :class:`RawRollout` with rewards attached
- :class:`BaseRewardModel` — pluggable reward interface
- :class:`RolloutGenerator` — KV-cache-backed generation of grouped
  responses + decoding (no reward); delegates the generation loop to
  :class:`~astrai.inference.scheduler.InferenceScheduler.run_batch`
  so rollout and the production inference server share one code path
- :class:`RolloutRunner` — orchestrates generation + scoring with a
  step-driven cache; its ``__call__`` returns ``(RolloutResult, is_fresh)``
  so callers do not need to rely on object identity to detect refreshes.
"""

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import torch
import torch.distributed as dist
from torch import Tensor

from astrai.inference.scheduler import InferenceScheduler
from astrai.inference.task import GenerationResult


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
    """

    prompts: Tensor
    prompt_mask: Tensor
    responses: Tensor
    response_mask: Tensor
    logprobs_old: Tensor
    policy_version: int = 0
    prompt_texts: List[str] = field(default_factory=list)
    response_texts: List[List[str]] = field(default_factory=list)
    sampling_groups: List["DynamicSamplingGroup"] = field(default_factory=list)


@dataclass(kw_only=True)
class RolloutResult(RawRollout):
    """A :class:`RawRollout` with reward scoring attached.

    Produced by :class:`RolloutRunner` once the :class:`BaseRewardModel`
    has scored the decoded responses.

    Fields:
        rewards: Reward per response, shape ``[B, G]``.
    """

    rewards: Tensor


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


class DynamicSamplingBudgetError(RuntimeError):
    """A dynamic-sampling step cannot produce a complete, safe batch."""


class DynamicSamplingState(str, Enum):
    """Lifecycle of one prompt group's generation attempt."""

    PENDING = "pending"
    GENERATING = "generating"
    SCORING = "scoring"
    ACCEPTED = "accepted"
    REFILL = "refill"
    INVALIDATED = "invalidated"
    DROPPED = "dropped"


_DYNAMIC_SAMPLING_TRANSITIONS = {
    DynamicSamplingState.PENDING: {DynamicSamplingState.GENERATING},
    DynamicSamplingState.GENERATING: {
        DynamicSamplingState.SCORING,
        DynamicSamplingState.INVALIDATED,
        DynamicSamplingState.DROPPED,
    },
    DynamicSamplingState.SCORING: {
        DynamicSamplingState.ACCEPTED,
        DynamicSamplingState.REFILL,
        DynamicSamplingState.INVALIDATED,
        DynamicSamplingState.DROPPED,
    },
    # Acceptance is provisional until the whole step is committed. A policy
    # change while another group is refilled invalidates every old-version row.
    DynamicSamplingState.ACCEPTED: {DynamicSamplingState.INVALIDATED},
    DynamicSamplingState.REFILL: set(),
    DynamicSamplingState.INVALIDATED: set(),
    DynamicSamplingState.DROPPED: set(),
}


@dataclass
class DynamicSamplingGroup:
    """Auditable metadata for one prompt-group generation attempt."""

    prompt_uid: str
    attempt_id: int
    generation_seed: int
    behavior_policy_version: Optional[int] = None
    reward_vector: List[float] = field(default_factory=list)
    reward_variance: Optional[float] = None
    accepted: bool = False
    discard_reason: Optional[str] = None
    refill_round: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    generated_tokens: int = 0
    state: DynamicSamplingState = DynamicSamplingState.PENDING

    def transition(
        self,
        state: DynamicSamplingState,
        *,
        discard_reason: Optional[str] = None,
    ) -> None:
        """Move to ``state`` while enforcing the lifecycle graph."""
        if state not in _DYNAMIC_SAMPLING_TRANSITIONS[self.state]:
            raise RuntimeError(
                f"invalid dynamic-sampling transition {self.state.value} -> "
                f"{state.value}"
            )
        self.state = state
        self.discard_reason = discard_reason
        self.accepted = state is DynamicSamplingState.ACCEPTED
        if state in {
            DynamicSamplingState.ACCEPTED,
            DynamicSamplingState.REFILL,
            DynamicSamplingState.INVALIDATED,
            DynamicSamplingState.DROPPED,
        }:
            self.completed_at = time.time()


@dataclass(frozen=True)
class DynamicSamplingConfig:
    """Budgets and acceptance policy for versioned rollout refill."""

    enabled: bool = False
    variance_threshold: float = 0.0
    max_refill_rounds: int = 2
    max_generated_tokens_per_group: int = 32_768
    max_wall_time_per_group: float = 300.0
    max_total_rollout_tokens_per_step: int = 262_144
    max_pending_groups: int = 128
    base_seed: int = 3407

    def __post_init__(self) -> None:
        if self.variance_threshold < 0:
            raise ValueError("variance_threshold must be non-negative")
        if self.max_refill_rounds < 0:
            raise ValueError("max_refill_rounds must be non-negative")
        for name in (
            "max_generated_tokens_per_group",
            "max_total_rollout_tokens_per_step",
            "max_pending_groups",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_wall_time_per_group <= 0:
            raise ValueError("max_wall_time_per_group must be positive")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")


@dataclass
class DynamicSamplingMetrics:
    """Per-refresh dynamic-sampling counters and efficiency metrics."""

    groups_total: int = 0
    groups_accepted: int = 0
    groups_zero_variance: int = 0
    refill_rounds: int = 0
    refill_tokens: int = 0
    groups_dropped: int = 0
    groups_version_invalidated: int = 0
    groups_budget_exhausted: int = 0
    total_generated_tokens: int = 0
    accepted_generated_tokens: int = 0

    def as_dict(self) -> Dict[str, float]:
        total = self.total_generated_tokens
        effective = 0.0 if total == 0 else self.groups_accepted * 1_000_000 / total
        waste = 0.0 if total == 0 else (total - self.accepted_generated_tokens) / total
        return {
            "groups_total": float(self.groups_total),
            "groups_accepted": float(self.groups_accepted),
            "zero_variance_groups": float(self.groups_zero_variance),
            "refill_rounds": float(self.refill_rounds),
            "refill_tokens": float(self.refill_tokens),
            "dropped_groups": float(self.groups_dropped),
            "version_invalidated_groups": float(self.groups_version_invalidated),
            "budget_exhausted_groups": float(self.groups_budget_exhausted),
            "total_generated_tokens": float(total),
            "effective_groups_per_million_tokens": effective,
            "rollout_waste_ratio": waste,
        }


class RolloutGenerator:
    """Pure generation + decoding for a group of responses per prompt.

    Delegates the prefill/decode loop to
    :meth:`~astrai.inference.scheduler.InferenceScheduler.run_batch`,
    which uses a real KV cache (no O(n²) recompute).  Has no dependency
    on any reward model; can be reused in isolation for offline
    generation, qualitative sampling, or eval pipelines.
    """

    def __init__(
        self,
        scheduler: InferenceScheduler,
        tokenizer,
        max_tokens: int = 1024,
        group_size: int = 8,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        rep_window: int = 64,
    ):
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.group_size = group_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.rep_window = rep_window
        self._weight_lock = threading.RLock()

    @property
    def policy_version(self) -> int:
        return self.scheduler.policy_version

    def update_weights(self, policy_version: int) -> int:
        """Acknowledge shared-model weights and invalidate older scheduler KV."""
        with self._weight_lock:
            return self.scheduler.update_weights(policy_version)

    def apply_weight_update(self, policy_version: int, update: Callable[[], T]) -> T:
        """Apply a shared-model mutation at an atomic generation boundary."""
        with self._weight_lock:
            return self.scheduler.apply_weight_update(policy_version, update)

    def with_policy_snapshot(self, inspect: Callable[[int], T]) -> T:
        """Inspect a version stable against generator and scheduler updates."""
        if not callable(inspect):
            raise TypeError("inspect must be callable")
        with self._weight_lock:
            return self.scheduler.with_policy_snapshot(inspect)

    @torch.no_grad()
    def generate(
        self, batch: Dict, *, generation_seed: Optional[int] = None
    ) -> RawRollout:
        """Expand prompts by ``group_size`` and generate one response each.

        Accepted batch formats (per sample, repeated B times):

        - **messages**: ``{"messages": [{"role": "user", "content": "..."}, ...]}``
        - **instruction + input + output**: ``{"instruction": "...",
          "input": "...", "output": "..."}`` — mapped to ``system`` /
          ``user`` / ``assistant`` messages; ``input`` and ``output``
          are optional and skipped when empty.

        Both are rendered through the tokenizer's chat template with
        ``add_generation_prompt=True`` so rollout prompts match the
        format the policy was SFT-trained on.
        """
        with self._weight_lock:

            def generate_snapshot(generation_version: int) -> RawRollout:
                model = self.scheduler._executor.model
                was_training = model.training
                model.eval()
                try:
                    if generation_seed is None:
                        return self._generate_eval(batch, generation_version)
                    if generation_seed < 0:
                        raise ValueError("generation_seed must be non-negative")
                    device = torch.device(self.scheduler.device)
                    cuda_devices = [device] if device.type == "cuda" else []
                    # Restore caller RNG state after deterministic generation;
                    # retries must not perturb training-side randomness.
                    with torch.random.fork_rng(devices=cuda_devices):
                        torch.manual_seed(generation_seed)
                        return self._generate_eval(batch, generation_version)
                finally:
                    model.train(was_training)

            # Capture the version under the scheduler lock as well as the
            # generator lock. This also serializes callers that update the
            # scheduler directly instead of going through this wrapper.
            return self.scheduler.with_policy_snapshot(generate_snapshot)

    def _generate_eval(self, batch: Dict, generation_version: int) -> RawRollout:
        prompt_texts, flat_prompt_ids = self._prepare_prompts(batch)
        B = len(prompt_texts)
        G = self.group_size
        # Re-expand flat list to G copies per prompt for run_batch.
        expanded_prompt_ids: List[List[int]] = []
        for ids in flat_prompt_ids:
            expanded_prompt_ids.extend([list(ids)] * G)

        results = self.scheduler.run_batch(
            expanded_prompt_ids,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            rep_window=self.rep_window,
            return_logprobs=True,
            return_details=True,
        )
        if len(results) != B * G:
            raise RuntimeError(
                f"Rollout scheduler returned {len(results)} results, expected {B * G}"
            )
        for result in results:
            if not isinstance(result, GenerationResult):
                raise RuntimeError("Rollout scheduler returned an invalid result type")

        failures = [
            (index, result)
            for index, result in enumerate(results)
            if result.error_reason is not None
            or result.finish_reason in ("cancelled", "rejected")
        ]
        if failures:
            reasons = ", ".join(
                f"request {index}: {result.error_reason or result.finish_reason}"
                for index, result in failures
            )
            raise RuntimeError(f"Rollout generation failed: {reasons}")

        for result in results:
            if len(result.token_ids) != len(result.logprobs):
                raise RuntimeError(
                    "Rollout scheduler returned misaligned token IDs and logprobs"
                )

        # Pad successful structured results to a uniform response length.
        max_len = max((len(result.token_ids) for result in results), default=0)
        max_len = max(max_len, 1)

        device = self.scheduler.device
        P_len = max(len(ids) for ids in flat_prompt_ids)
        prompts_tensor = torch.zeros(B, P_len, dtype=torch.long, device=device)
        prompt_mask = torch.zeros(B, P_len, dtype=torch.bool, device=device)
        for i, ids in enumerate(flat_prompt_ids):
            prompts_tensor[i, -len(ids) :] = torch.tensor(
                ids, dtype=torch.long, device=device
            )
            prompt_mask[i, -len(ids) :] = True

        responses = torch.full((B, G, max_len), _PAD, dtype=torch.long, device=device)
        response_mask = torch.zeros((B, G, max_len), dtype=torch.bool, device=device)
        logprobs_old = torch.zeros((B, G, max_len), dtype=torch.float, device=device)

        flat_idx = 0
        response_texts: List[List[str]] = [[] for _ in range(B)]
        for i in range(B):
            for g in range(G):
                result = results[flat_idx]
                token_ids, lps = result.token_ids, result.logprobs
                flat_idx += 1
                n = len(token_ids)
                if n:
                    responses[i, g, :n] = torch.tensor(
                        token_ids, dtype=torch.long, device=device
                    )
                    response_mask[i, g, :n] = True
                    logprobs_old[i, g, :n] = torch.tensor(
                        lps, dtype=torch.float, device=device
                    )
                response_texts[i].append(
                    self.tokenizer.decode(token_ids, skip_special_tokens=True)
                )

        return RawRollout(
            prompts=prompts_tensor,
            prompt_mask=prompt_mask,
            responses=responses,
            response_mask=response_mask,
            logprobs_old=logprobs_old,
            policy_version=generation_version,
            prompt_texts=prompt_texts,
            response_texts=response_texts,
        )

    def _prepare_prompts(self, batch: Dict) -> Tuple[List[str], List[List[int]]]:
        """Render batch prompts to ``(texts, token_id_lists)``.

        Returns two parallel lists of length B (number of prompts in
        the batch).  Dispatches by batch keys:

        - ``"messages"``: treated as a pre-built message list per sample.
        - ``"instruction"`` (optionally ``"input"`` and ``"output"``): mapped
          to ``system`` / ``user`` / ``assistant`` messages respectively.

        Both paths go through the tokenizer's chat template with
        ``add_generation_prompt=True``.
        """
        if "messages" in batch:
            messages_list = batch["messages"]
        elif "instruction" in batch:
            instructions = batch["instruction"]
            B = len(instructions)
            inputs = batch.get("input") or [""] * B
            outputs = batch.get("output") or [""] * B
            messages_list = [
                self._instruction_to_messages(i, u, o)
                for i, u, o in zip(instructions, inputs, outputs)
            ]
        else:
            raise ValueError(
                "Rollout batch must contain either 'messages' or "
                "'instruction' (optionally 'input'/'output'); got keys: "
                f"{list(batch.keys())}"
            )

        try:
            prompt_texts = self.tokenizer.apply_chat_template(
                messages_list, tokenize=False, add_generation_prompt=True
            )
            if (
                not isinstance(prompt_texts, list)
                or len(prompt_texts) != len(messages_list)
                or not all(isinstance(text, str) for text in prompt_texts)
            ):
                raise TypeError("Tokenizer does not support batched chat templates")
            flat_prompt_ids = self.tokenizer.encode(prompt_texts)
            if len(flat_prompt_ids) != len(messages_list) or not all(
                isinstance(ids, list) for ids in flat_prompt_ids
            ):
                raise TypeError("Tokenizer does not support batched encoding")
        except (TypeError, IndexError, KeyError):
            # Keep compatibility with lightweight tokenizer adapters that only
            # implement the single-conversation template API.
            prompt_texts = []
            flat_prompt_ids = []
            for messages in messages_list:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                prompt_texts.append(text)
                flat_prompt_ids.append(list(ids))
        return prompt_texts, flat_prompt_ids

    @staticmethod
    def _instruction_to_messages(
        instruction: str, inp: str = "", output: str = ""
    ) -> List[Dict[str, str]]:
        """Map instruction/input/output to chat messages.

        Role mapping follows the convention used throughout the
        preprocessing pipeline: ``instruction`` → system, ``input`` →
        user, ``output`` → assistant.  Empty fields are skipped so a
        bare instruction produces a ``[system]`` list and the chat
        template's ``add_generation_prompt`` adds the assistant header
        for sampling.
        """
        messages: List[Dict[str, str]] = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        if inp:
            messages.append({"role": "user", "content": inp})
        if output:
            messages.append({"role": "assistant", "content": output})
        return messages


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

        generator = RolloutGenerator(policy, tokenizer, pipeline, ...)
        runner = RolloutRunner(generator, reward_model, rollout_interval=512)
        result, is_fresh = runner(prompt_batch)
        if is_fresh:
            ...  # e.g. sync behaviour policy
    """

    def __init__(
        self,
        generator: RolloutGenerator,
        reward_model: BaseRewardModel,
        rollout_interval: int = 512,
        max_policy_lag: Optional[int] = None,
        dynamic_sampling: Optional[DynamicSamplingConfig] = None,
    ):
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
        self.dynamic_sampling = dynamic_sampling or DynamicSamplingConfig()

        self._cache: Optional[RolloutResult] = None
        self._cache_key = None
        self._steps_since_rollout: int = 0
        self._dynamic_refresh_id: int = 0
        self._dynamic_attempt_id: int = 0
        self._last_sampling_metrics: Dict[str, float] = {}
        self._last_sampling_history: List[DynamicSamplingGroup] = []

    @property
    def policy_version(self) -> int:
        return self.generator.policy_version

    def update_weights(self, policy_version: int) -> int:
        """Publish the shared policy's new version to the rollout backend."""
        return self.generator.update_weights(policy_version)

    def apply_weight_update(self, policy_version: int, update: Callable[[], T]) -> T:
        """Apply a model update and publish its version as one operation."""
        return self.generator.apply_weight_update(policy_version, update)

    def step(self):
        """Advance the internal counter (call once per optimizer step)."""
        self._steps_since_rollout += 1

    def clear_cache(self):
        """Force next call to re-run rollout."""
        self._cache = None
        self._cache_key = None

    @property
    def last_sampling_metrics(self) -> Dict[str, float]:
        """Metrics from the most recent dynamic-sampling refresh."""
        return dict(self._last_sampling_metrics)

    @property
    def last_sampling_history(self) -> List[DynamicSamplingGroup]:
        """Attempt records from the most recent dynamic-sampling refresh."""
        return list(self._last_sampling_history)

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
        rewards = self.reward_model.score(raw.prompt_texts, raw.response_texts)
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
        device = raw.prompts.device
        return RolloutResult(
            prompts=raw.prompts,
            prompt_mask=raw.prompt_mask,
            responses=raw.responses,
            response_mask=raw.response_mask,
            rewards=rewards.to(device=device),
            logprobs_old=raw.logprobs_old,
            policy_version=raw.policy_version,
            prompt_texts=raw.prompt_texts,
            response_texts=raw.response_texts,
            sampling_groups=raw.sampling_groups,
        )

    @staticmethod
    def _batch_size(batch: Dict) -> int:
        if "messages" in batch:
            return len(batch["messages"])
        if "instruction" in batch:
            return len(batch["instruction"])
        raise ValueError("dynamic sampling requires messages or instruction prompts")

    @staticmethod
    def _select_batch(batch: Dict, indices: List[int]) -> Dict:
        """Select prompt rows without retaining unrelated training tensors."""
        selected = {}
        for key in ("messages", "instruction", "input", "output"):
            if key not in batch:
                continue
            value = batch[key]
            if isinstance(value, Tensor):
                index = torch.tensor(indices, dtype=torch.long, device=value.device)
                selected[key] = value.index_select(0, index)
            elif isinstance(value, tuple):
                selected[key] = tuple(value[index] for index in indices)
            else:
                selected[key] = [value[index] for index in indices]
        return selected

    def _prompt_uids(self, batch: Dict, batch_size: int) -> List[str]:
        batch_digest = hashlib.sha256(repr(self._batch_key(batch)).encode()).hexdigest()
        return [f"{batch_digest[:20]}:{index}" for index in range(batch_size)]

    @staticmethod
    def _collective_device(device: torch.device) -> torch.device:
        if dist.is_available() and dist.is_initialized():
            return device if dist.get_backend() == "nccl" else torch.device("cpu")
        return device

    @classmethod
    def _synchronize_version(cls, version: int, device: torch.device) -> int:
        if not (dist.is_available() and dist.is_initialized()):
            return version
        collective_device = cls._collective_device(device)
        minimum = torch.tensor(version, dtype=torch.long, device=collective_device)
        maximum = minimum.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        min_version, max_version = int(minimum.item()), int(maximum.item())
        if min_version != max_version:
            raise RolloutVersionError(
                "rollout policy version differs across ranks: "
                f"min={min_version}, max={max_version}"
            )
        return min_version

    @classmethod
    def _synchronize_acceptance(
        cls, accepted: List[bool], device: torch.device
    ) -> List[bool]:
        if not (dist.is_available() and dist.is_initialized()):
            return accepted
        collective_device = cls._collective_device(device)
        flags = torch.tensor(accepted, dtype=torch.int32, device=collective_device)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        return [bool(value) for value in flags.cpu().tolist()]

    @classmethod
    def _synchronize_failure(cls, failed: bool, device: torch.device) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return failed
        collective_device = cls._collective_device(device)
        flag = torch.tensor(int(failed), dtype=torch.int32, device=collective_device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    @staticmethod
    def _combine_dynamic_rows(
        accepted: Dict[int, Tuple[RolloutResult, int, DynamicSamplingGroup]],
        *,
        batch_size: int,
        policy_version: int,
    ) -> RolloutResult:
        """Reassemble accepted rows, padding attempts to common dimensions."""
        ordered = [accepted[index] for index in range(batch_size)]
        first = ordered[0][0]
        group_size = first.responses.size(1)
        max_prompt_len = max(result.prompts.size(1) for result, _, _ in ordered)
        max_response_len = max(result.responses.size(2) for result, _, _ in ordered)
        device = first.prompts.device

        prompts = torch.zeros(
            batch_size, max_prompt_len, dtype=first.prompts.dtype, device=device
        )
        prompt_mask = torch.zeros(
            batch_size, max_prompt_len, dtype=torch.bool, device=device
        )
        responses = torch.full(
            (batch_size, group_size, max_response_len),
            _PAD,
            dtype=first.responses.dtype,
            device=device,
        )
        response_mask = torch.zeros_like(responses, dtype=torch.bool)
        logprobs_old = torch.zeros(
            batch_size,
            group_size,
            max_response_len,
            dtype=first.logprobs_old.dtype,
            device=device,
        )
        rewards = torch.zeros(
            batch_size, group_size, dtype=first.rewards.dtype, device=device
        )
        prompt_texts: List[str] = []
        response_texts: List[List[str]] = []
        sampling_groups: List[DynamicSamplingGroup] = []

        for destination, (result, source, record) in enumerate(ordered):
            if result.policy_version != policy_version:
                raise RolloutVersionError(
                    "dynamic sampling attempted to mix behavior policy versions"
                )
            prompt_len = result.prompts.size(1)
            response_len = result.responses.size(2)
            prompts[destination, -prompt_len:] = result.prompts[source]
            prompt_mask[destination, -prompt_len:] = result.prompt_mask[source]
            responses[destination, :, :response_len] = result.responses[source]
            response_mask[destination, :, :response_len] = result.response_mask[source]
            logprobs_old[destination, :, :response_len] = result.logprobs_old[source]
            rewards[destination] = result.rewards[source]
            prompt_texts.append(result.prompt_texts[source])
            response_texts.append(result.response_texts[source])
            sampling_groups.append(record)

        return RolloutResult(
            prompts=prompts,
            prompt_mask=prompt_mask,
            responses=responses,
            response_mask=response_mask,
            logprobs_old=logprobs_old,
            rewards=rewards,
            policy_version=policy_version,
            prompt_texts=prompt_texts,
            response_texts=response_texts,
            sampling_groups=sampling_groups,
        )

    def _generate_dynamic(self, batch: Dict) -> RolloutResult:
        """Generate a full batch without mixing behavior-policy versions."""
        config = self.dynamic_sampling
        batch_size = self._batch_size(batch)
        if batch_size == 0:
            raise ValueError("dynamic sampling requires at least one prompt group")
        if batch_size > config.max_pending_groups:
            raise DynamicSamplingBudgetError(
                f"pending groups {batch_size} exceed max_pending_groups="
                f"{config.max_pending_groups}"
            )

        metrics = DynamicSamplingMetrics(groups_total=batch_size)
        history: List[DynamicSamplingGroup] = []
        accepted: Dict[int, Tuple[RolloutResult, int, DynamicSamplingGroup]] = {}
        pending = list(range(batch_size))
        prompt_uids = self._prompt_uids(batch, batch_size)
        refill_rounds = [0] * batch_size
        generated_per_group = [0] * batch_size
        group_started = [time.monotonic()] * batch_size
        target_version: Optional[int] = None
        max_attempt_tokens = self.generator.group_size * self.generator.max_tokens
        self._dynamic_refresh_id += 1

        try:
            while pending:
                seed = (
                    config.base_seed
                    + self._dynamic_refresh_id * 1_000_003
                    + self._dynamic_attempt_id
                )
                records: List[DynamicSamplingGroup] = []
                for index in pending:
                    self._dynamic_attempt_id += 1
                    record = DynamicSamplingGroup(
                        prompt_uid=prompt_uids[index],
                        attempt_id=self._dynamic_attempt_id,
                        generation_seed=seed,
                        refill_round=refill_rounds[index],
                    )
                    record.transition(DynamicSamplingState.GENERATING)
                    records.append(record)

                preflight_reason = None
                if any(
                    generated_per_group[index] + max_attempt_tokens
                    > config.max_generated_tokens_per_group
                    for index in pending
                ):
                    preflight_reason = "max_generated_tokens_per_group"
                elif (
                    metrics.total_generated_tokens + len(pending) * max_attempt_tokens
                    > config.max_total_rollout_tokens_per_step
                ):
                    preflight_reason = "max_total_rollout_tokens_per_step"
                elif any(
                    time.monotonic() - group_started[index]
                    >= config.max_wall_time_per_group
                    for index in pending
                ):
                    preflight_reason = "max_wall_time_per_group"

                preflight_failed = self._synchronize_failure(
                    preflight_reason is not None,
                    self._collective_device(
                        torch.device(self.generator.scheduler.device)
                    ),
                )
                if preflight_failed:
                    reason = preflight_reason or "peer_rank_budget_exhausted"
                    for record in records:
                        record.transition(
                            DynamicSamplingState.DROPPED,
                            discard_reason=reason,
                        )
                    for _, _, record in accepted.values():
                        record.transition(
                            DynamicSamplingState.INVALIDATED,
                            discard_reason="peer_group_budget_exhausted",
                        )
                    history.extend(records)
                    history.extend(record for _, _, record in accepted.values())
                    metrics.groups_dropped += len(pending)
                    metrics.groups_budget_exhausted += len(pending)
                    raise DynamicSamplingBudgetError(
                        "dynamic sampling cannot start another generation "
                        f"attempt within {reason}"
                    )

                subset = self._select_batch(batch, pending)
                raw = None
                generation_error = None
                try:
                    raw = self.generator.generate(subset, generation_seed=seed)
                except Exception as error:
                    generation_error = error
                generation_failed = self._synchronize_failure(
                    generation_error is not None,
                    torch.device(self.generator.scheduler.device),
                )
                if generation_failed:
                    for record in records:
                        record.transition(
                            DynamicSamplingState.DROPPED,
                            discard_reason="generation_failed",
                        )
                    for _, _, record in accepted.values():
                        record.transition(
                            DynamicSamplingState.INVALIDATED,
                            discard_reason="peer_group_generation_failed",
                        )
                    history.extend(records)
                    history.extend(record for _, _, record in accepted.values())
                    metrics.groups_dropped += len(pending)
                    if generation_error is not None:
                        raise generation_error
                    raise RuntimeError(
                        "dynamic sampling generation failed on a peer rank"
                    )
                assert raw is not None
                version_error = None
                try:
                    self._validate_policy_version(raw)
                except RolloutVersionError as error:
                    version_error = error
                version_failed = self._synchronize_failure(
                    version_error is not None, raw.prompts.device
                )
                if version_failed:
                    for record in records:
                        record.transition(
                            DynamicSamplingState.INVALIDATED,
                            discard_reason="policy_version_rejected",
                        )
                    for _, _, record in accepted.values():
                        record.transition(
                            DynamicSamplingState.INVALIDATED,
                            discard_reason="peer_policy_version_rejected",
                        )
                    history.extend(records)
                    history.extend(record for _, _, record in accepted.values())
                    metrics.groups_version_invalidated += len(pending) + len(accepted)
                    if version_error is not None:
                        raise version_error
                    raise RolloutVersionError(
                        "dynamic sampling policy version was rejected on a peer rank"
                    )
                version = self._synchronize_version(
                    raw.policy_version, raw.prompts.device
                )

                token_counts = raw.response_mask.sum(dim=(1, 2)).cpu().tolist()
                for record, token_count, index in zip(records, token_counts, pending):
                    record.behavior_policy_version = version
                    record.generated_tokens = int(token_count)
                    generated_per_group[index] += int(token_count)
                    metrics.total_generated_tokens += int(token_count)
                    if record.refill_round > 0:
                        metrics.refill_rounds += 1
                        metrics.refill_tokens += int(token_count)

                if target_version is not None and version != target_version:
                    for record in records:
                        record.transition(
                            DynamicSamplingState.INVALIDATED,
                            discard_reason="policy_version_changed",
                        )
                    for _, _, record in accepted.values():
                        record.transition(
                            DynamicSamplingState.INVALIDATED,
                            discard_reason="policy_version_changed_during_refill",
                        )
                    invalidated = len(records) + len(accepted)
                    metrics.groups_version_invalidated += invalidated
                    history.extend(records)
                    history.extend(record for _, _, record in accepted.values())
                    accepted.clear()
                    pending = list(range(batch_size))
                    refill_rounds = [0] * batch_size
                    target_version = version
                    continue

                if target_version is None:
                    target_version = version

                for record in records:
                    record.transition(DynamicSamplingState.SCORING)
                scored = None
                scoring_error = None
                try:
                    scored = self._score(raw)
                except Exception as error:
                    scoring_error = error
                scoring_failed = self._synchronize_failure(
                    scoring_error is not None, raw.prompts.device
                )
                if scoring_failed:
                    for record in records:
                        record.transition(
                            DynamicSamplingState.DROPPED,
                            discard_reason="scoring_failed",
                        )
                    for _, _, record in accepted.values():
                        record.transition(
                            DynamicSamplingState.INVALIDATED,
                            discard_reason="peer_group_scoring_failed",
                        )
                    history.extend(records)
                    history.extend(record for _, _, record in accepted.values())
                    metrics.groups_dropped += len(pending)
                    if scoring_error is not None:
                        raise scoring_error
                    raise RuntimeError("dynamic sampling scoring failed on a peer rank")
                assert scored is not None
                variances = scored.rewards.float().var(dim=1, unbiased=False)
                locally_accepted = [
                    bool(value > config.variance_threshold) for value in variances
                ]
                jointly_accepted = self._synchronize_acceptance(
                    locally_accepted, scored.prompts.device
                )

                next_pending: List[int] = []
                local_budget_exhausted = False
                remaining_step_tokens = (
                    config.max_total_rollout_tokens_per_step
                    - metrics.total_generated_tokens
                )
                for row, (index, record, accept) in enumerate(
                    zip(pending, records, jointly_accepted)
                ):
                    rewards = scored.rewards[row].detach().float().cpu()
                    record.reward_vector = rewards.tolist()
                    record.reward_variance = float(variances[row].item())
                    if accept:
                        record.transition(DynamicSamplingState.ACCEPTED)
                        accepted[index] = (scored, row, record)
                        continue

                    metrics.groups_zero_variance += 1
                    elapsed = time.monotonic() - group_started[index]
                    reason = None
                    if refill_rounds[index] >= config.max_refill_rounds:
                        reason = "max_refill_rounds"
                    elif (
                        generated_per_group[index] + max_attempt_tokens
                        > config.max_generated_tokens_per_group
                    ):
                        reason = "max_generated_tokens_per_group"
                    elif elapsed >= config.max_wall_time_per_group:
                        reason = "max_wall_time_per_group"
                    elif remaining_step_tokens < max_attempt_tokens:
                        reason = "max_total_rollout_tokens_per_step"

                    if reason is not None:
                        record.transition(
                            DynamicSamplingState.DROPPED,
                            discard_reason=reason,
                        )
                        metrics.groups_dropped += 1
                        metrics.groups_budget_exhausted += 1
                        local_budget_exhausted = True
                    else:
                        record.transition(
                            DynamicSamplingState.REFILL,
                            discard_reason="reward_variance_below_threshold",
                        )
                        refill_rounds[index] += 1
                        remaining_step_tokens -= max_attempt_tokens
                        next_pending.append(index)
                    history.append(record)

                failed = self._synchronize_failure(
                    local_budget_exhausted, scored.prompts.device
                )
                if failed:
                    raise DynamicSamplingBudgetError(
                        "dynamic sampling exhausted a group budget; refusing "
                        "a partial or cross-rank-inconsistent training batch"
                    )
                pending = next_pending

            assert target_version is not None
            final = self._combine_dynamic_rows(
                accepted, batch_size=batch_size, policy_version=target_version
            )
            metrics.groups_accepted = batch_size
            metrics.accepted_generated_tokens = int(final.response_mask.sum().item())
            history.extend(record for _, _, record in accepted.values())
            self._last_sampling_metrics = metrics.as_dict()
            self._last_sampling_history = history
            return final
        except BaseException:
            self._last_sampling_metrics = metrics.as_dict()
            self._last_sampling_history = history
            raise

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
        or when the cache is empty.
        """
        cache_key = self._batch_key(batch)
        if (
            self._cache is None
            or cache_key != self._cache_key
            or self._steps_since_rollout >= self.rollout_interval
        ):
            if self.dynamic_sampling.enabled:
                scored = self._generate_dynamic(batch)
            else:
                raw = self.generator.generate(batch)
                self._validate_policy_version(raw)
                scored = self._score(raw)

            def commit(live_version: int) -> Tuple[RolloutResult, bool]:
                if self.dynamic_sampling.enabled:
                    live_version = self._synchronize_version(
                        live_version, scored.prompts.device
                    )
                self._validate_policy_version(scored, live_version=live_version)
                self._cache = scored
                self._cache_key = cache_key
                self._steps_since_rollout = 0
                return scored, True

            # A weight update cannot land between the final version check and
            # cache publication. Reward scoring itself intentionally remains
            # outside the policy lock because it may call an external service.
            return self.generator.with_policy_snapshot(commit)

        cached = self._cache
        assert cached is not None

        def reuse(live_version: int) -> Tuple[RolloutResult, bool]:
            if self.dynamic_sampling.enabled:
                live_version = self._synchronize_version(
                    live_version, cached.prompts.device
                )
            self._validate_policy_version(cached, live_version=live_version)
            return cached, False

        return self.generator.with_policy_snapshot(reuse)
