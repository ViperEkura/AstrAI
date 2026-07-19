"""Online rollout runner for RL training.

Provides:
- :class:`RolloutResult` — universal data container for online sampling
- :class:`BaseRewardModel` — pluggable reward interface
- :class:`RolloutRunner` — generates + scores batches for any RL strategy
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.inference.sample import SamplingPipeline


@dataclass
class RolloutResult:
    """Universal container produced by :class:`RolloutRunner`.

    Fields are designed to cover all common RL algorithms:
    GRPO, PPO, Online DPO, Rejection Sampling, etc.
    """

    prompts: Tensor
    """Tokenized prompts, shape ``[B, P_len]``."""

    responses: Tensor
    """Generated response token IDs, shape ``[B, G, R_max]``."""

    response_mask: Tensor
    """Boolean mask for real (non-pad) response tokens, shape ``[B, G, R_max]``."""

    rewards: Tensor
    """Reward per response, shape ``[B, G]``."""

    logprobs_old: Tensor
    """Per-token log-probs under the behaviour policy, shape ``[B, G, R_max]``."""

    prompt_texts: List[str] = field(default_factory=list)
    """Decoded prompt strings (for reward models that need text)."""

    response_texts: List[List[str]] = field(default_factory=list)
    """Decoded response strings, shape ``[B, G]`` (for reward models)."""


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


def generate_responses(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    max_new_tokens: int,
    sampling_pipeline: SamplingPipeline,
    stop_ids: List[int],
) -> Dict[str, Tensor]:
    """Autoregressive generation with log-prob tracking.

    Args:
        model: Policy model (``forward`` returns ``{"logits": ...}``).
        input_ids: ``[B, P_len]`` prompt token IDs.
        attention_mask: ``[B, P_len]`` boolean mask.
        max_new_tokens: Maximum tokens to generate.
        sampling_pipeline: Composed sampling strategies.
        stop_ids: Token IDs that stop generation (eos, etc.).

    Returns:
        ``dict`` with keys:
        - ``generated_ids``: ``[B, max_new_tokens]`` (padded to same length)
        - ``generated_mask``: ``[B, max_new_tokens]``
        - ``logprobs``: ``[B, max_new_tokens]`` per-token log-probs
    """
    _PAD = 0
    B, P_len = input_ids.shape
    device = input_ids.device
    stop_ids_set = set(stop_ids)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    all_ids = input_ids.clone()
    all_mask = attention_mask.clone()
    logprob_list: List[Tensor] = []

    for _ in range(max_new_tokens):
        outputs = model(input_ids=all_ids, input_mask=all_mask)
        logits = outputs["logits"][:, -1, :].float()
        log_probs = F.log_softmax(logits, dim=-1)

        logits = sampling_pipeline.apply(logits, input_ids=all_ids, input_mask=all_mask)
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        next_tokens[done] = _PAD
        chosen_logprobs = torch.gather(log_probs, -1, next_tokens.unsqueeze(-1))
        logprob_list.append(chosen_logprobs)

        all_ids = torch.cat([all_ids, next_tokens.unsqueeze(1)], dim=-1)
        all_mask = torch.cat([all_mask, (~done).unsqueeze(1)], dim=-1)

        done = done | torch.tensor(
            [t.item() in stop_ids_set for t in next_tokens],
            device=device,
        )
        if done.all():
            break

    logprobs = torch.cat(logprob_list, dim=-1)
    if logprobs.size(1) < max_new_tokens:
        pad_len = max_new_tokens - logprobs.size(1)
        logprobs = F.pad(logprobs, (0, pad_len), value=0.0)

    generated_ids = all_ids[:, P_len:]
    if generated_ids.size(1) < max_new_tokens:
        pad_len = max_new_tokens - generated_ids.size(1)
        generated_ids = F.pad(generated_ids, (0, pad_len), value=_PAD)

    generated_mask = generated_ids != _PAD

    return {
        "generated_ids": generated_ids,
        "generated_mask": generated_mask,
        "logprobs": logprobs,
    }


class RolloutRunner:
    """Produces :class:`RolloutResult` from a prompt batch.

    Maintains an internal cache so the same batch prompt can be replayed
    for multiple gradient steps.  A new rollout is triggered every
    ``rollout_interval`` calls to :meth:`step`.

    Usage::

        runner = RolloutRunner(policy, old_policy, tokenizer,
                               reward_model, sampling_pipeline, config)
        result = runner(prompt_batch)
    """

    def __init__(
        self,
        policy_model: nn.Module,
        old_model: Optional[nn.Module],
        tokenizer,
        reward_model: BaseRewardModel,
        sampling_pipeline: SamplingPipeline,
        max_tokens: int = 1024,
        group_size: int = 8,
        rollout_interval: int = 512,
    ):
        self.policy_model = policy_model
        self.old_model = old_model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.sampling_pipeline = sampling_pipeline
        self.max_tokens = max_tokens
        self.group_size = group_size
        self.rollout_interval = rollout_interval
        self.stop_ids = getattr(tokenizer, "stop_ids", []) or []

        self._cache: Optional[RolloutResult] = None
        self._steps_since_rollout: int = 0

    def step(self):
        """Advance the internal counter (call once per optimizer step)."""
        self._steps_since_rollout += 1

    def clear_cache(self):
        """Force next call to re-run rollout."""
        self._cache = None

    def _tokenize_prompts(self, raw_texts: List[str]) -> Dict[str, Tensor]:
        ids_list = self.tokenizer.encode(raw_texts, out_ids=True)
        B = len(ids_list)
        P_max = max(len(ids) for ids in ids_list) if ids_list else 0
        input_ids = torch.zeros(B, P_max, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            input_ids[i, : len(ids)] = torch.tensor(ids[:P_max], dtype=torch.long)
        attention_mask = input_ids != 0
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _decode(self, token_ids: Tensor, mask: Tensor) -> List[List[str]]:
        B, G, _ = token_ids.shape
        texts = []
        for i in range(B):
            group_texts = []
            for g in range(G):
                ids = token_ids[i, g, mask[i, g]].tolist()
                group_texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
            texts.append(group_texts)
        return texts

    @torch.no_grad()
    def _run(self, batch: Dict[str, Tensor]) -> RolloutResult:
        """Execute the actual generation + reward scoring."""
        prompt_ids = batch["input_ids"] if "input_ids" in batch else batch["prompts"]
        prompt_mask = (
            batch["attention_mask"] if "attention_mask" in batch else (prompt_ids != 0)
        )
        B, P_len = prompt_ids.shape
        G = self.group_size
        device = prompt_ids.device

        prompt_texts: List[str] = []
        for i in range(B):
            ids = prompt_ids[i, prompt_mask[i]].tolist()
            prompt_texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))

        expanded_ids = prompt_ids.unsqueeze(1).expand(-1, G, -1).reshape(B * G, P_len)
        expanded_mask = prompt_mask.unsqueeze(1).expand(-1, G, -1).reshape(B * G, P_len)

        gen_out = generate_responses(
            model=self.policy_model,
            input_ids=expanded_ids,
            attention_mask=expanded_mask,
            max_new_tokens=self.max_tokens,
            sampling_pipeline=self.sampling_pipeline,
            stop_ids=self.stop_ids,
        )

        gen_ids = gen_out["generated_ids"].reshape(B, G, -1)
        gen_mask = gen_out["generated_mask"].reshape(B, G, -1)
        gen_logprobs = gen_out["logprobs"].reshape(B, G, -1)

        response_texts = self._decode(gen_ids, gen_mask)
        reward_tensor = self.reward_model.score(prompt_texts, response_texts)
        rewards = reward_tensor.to(device=device)

        return RolloutResult(
            prompts=prompt_ids,
            responses=gen_ids,
            response_mask=gen_mask,
            rewards=rewards,
            logprobs_old=gen_logprobs,
            prompt_texts=prompt_texts,
            response_texts=response_texts,
        )

    def __call__(self, batch: Dict[str, Tensor]) -> RolloutResult:
        """Return cached or fresh :class:`RolloutResult`.

        Triggers a new rollout when ``_steps_since_rollout >= rollout_interval``
        or when the cache is empty.
        """
        if self._cache is None or self._steps_since_rollout >= self.rollout_interval:
            self._cache = self._run(batch)
            self._steps_since_rollout = 0
        return self._cache
