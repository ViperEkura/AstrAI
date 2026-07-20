"""Online rollout runner for RL training.

Provides:
- :class:`RawRollout` — generation output container (no reward yet)
- :class:`RolloutResult` — a :class:`RawRollout` with rewards attached
- :class:`BaseRewardModel` — pluggable reward interface
- :class:`RolloutGenerator` — KV-cache-backed generation of grouped
  responses + decoding (no reward); delegates the generation loop to
  :class:`~astrai.inference.core.scheduler.InferenceScheduler.run_batch`
  so rollout and the production inference server share one code path
- :class:`RolloutRunner` — orchestrates generation + scoring with a
  step-driven cache; its ``__call__`` returns ``(RolloutResult, is_fresh)``
  so callers do not need to rely on object identity to detect refreshes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from astrai.inference.core.scheduler import InferenceScheduler


@dataclass(kw_only=True)
class RawRollout:
    """Generation output before reward scoring.

    Produced by :class:`RolloutGenerator`; consumed by :class:`RolloutRunner`
    to assemble a :class:`RolloutResult` once rewards are attached.

    Fields are designed to cover all common RL algorithms:
    GRPO, PPO, Online DPO, Rejection Sampling, etc.

    Fields:
        prompts: Tokenized prompts, shape ``[B, P_len]``.
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
    responses: Tensor
    response_mask: Tensor
    logprobs_old: Tensor
    prompt_texts: List[str] = field(default_factory=list)
    response_texts: List[List[str]] = field(default_factory=list)


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


class RolloutGenerator:
    """Pure generation + decoding for a group of responses per prompt.

    Delegates the prefill/decode loop to
    :meth:`~astrai.inference.core.scheduler.InferenceScheduler.run_batch`,
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

    @torch.no_grad()
    def generate(self, batch: Dict) -> RawRollout:
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
        )

        # Each element is (token_ids, logprobs); pad to max length.
        max_len = 0
        for token_ids, _lp in results:
            max_len = max(max_len, len(token_ids))
        max_len = max(max_len, 1)

        device = self.scheduler.device
        P_len = max(len(ids) for ids in flat_prompt_ids)
        prompts_tensor = torch.zeros(B, P_len, dtype=torch.long, device=device)
        for i, ids in enumerate(flat_prompt_ids):
            prompts_tensor[i, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=device
            )

        responses = torch.full((B, G, max_len), _PAD, dtype=torch.long, device=device)
        response_mask = torch.zeros((B, G, max_len), dtype=torch.bool, device=device)
        logprobs_old = torch.zeros((B, G, max_len), dtype=torch.float, device=device)

        flat_idx = 0
        response_texts: List[List[str]] = [[] for _ in range(B)]
        for i in range(B):
            for g in range(G):
                token_ids, lps = results[flat_idx]
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
            responses=responses,
            response_mask=response_mask,
            logprobs_old=logprobs_old,
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

        prompt_texts: List[str] = []
        flat_prompt_ids: List[List[int]] = []
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
    ):
        self.generator = generator
        self.reward_model = reward_model
        self.rollout_interval = rollout_interval

        self._cache: Optional[RolloutResult] = None
        self._steps_since_rollout: int = 0

    def step(self):
        """Advance the internal counter (call once per optimizer step)."""
        self._steps_since_rollout += 1

    def clear_cache(self):
        """Force next call to re-run rollout."""
        self._cache = None

    def _score(self, raw: RawRollout) -> RolloutResult:
        rewards = self.reward_model.score(raw.prompt_texts, raw.response_texts)
        device = raw.prompts.device
        return RolloutResult(
            prompts=raw.prompts,
            responses=raw.responses,
            response_mask=raw.response_mask,
            rewards=rewards.to(device=device),
            logprobs_old=raw.logprobs_old,
            prompt_texts=raw.prompt_texts,
            response_texts=raw.response_texts,
        )

    def __call__(self, batch: Dict[str, Tensor]) -> Tuple[RolloutResult, bool]:
        """Return ``(cached or fresh) RolloutResult`` plus an ``is_fresh`` flag.

        Triggers a new rollout when ``_steps_since_rollout >= rollout_interval``
        or when the cache is empty.
        """
        if self._cache is None or self._steps_since_rollout >= self.rollout_interval:
            raw = self.generator.generate(batch)
            self._cache = self._score(raw)
            self._steps_since_rollout = 0
            return self._cache, True
        return self._cache, False
