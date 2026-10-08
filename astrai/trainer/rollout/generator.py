"""Generate grouped responses through the shared inference backend."""

import threading
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import torch

from astrai.inference.core.request import GenerationResult
from astrai.inference.sampling_rng import request_seed
from astrai.trainer.backend import RolloutBackend
from astrai.trainer.rollout.types import _PAD, RawRollout, SamplingParams, T


class RolloutGenerator:
    """Pure generation + decoding for a group of responses per prompt.

    Delegates the prefill/decode loop to the injected
    :class:`~astrai.trainer.backend.RolloutBackend` — colocated with the
    training model or a replica on another device; the generator is
    location-agnostic.  Has no dependency on any reward model; can be
    reused in isolation for offline generation, qualitative sampling, or
    eval pipelines.
    """

    def __init__(
        self,
        backend: RolloutBackend,
        tokenizer,
        params: SamplingParams,
        output_device=None,
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.params = params
        # Rollout tensors are handed to the training-side strategies on
        # this device; defaults to the backend's own (a no-op for a
        # colocated backend, a cross-device hop for a replica).
        self._output_device = (
            output_device
            if output_device is not None
            else getattr(backend, "device", None)
        )
        self._weight_lock = threading.RLock()

    @property
    def policy_version(self) -> int:
        return self.backend.policy_version

    def update_weights(self, policy_version: int) -> int:
        """Acknowledge shared-model weights and invalidate older scheduler KV."""
        with self._weight_lock:
            return self.backend.update_weights(policy_version)

    def apply_weight_update(
        self, policy_version: Optional[int], update: Callable[[int], T]
    ) -> T:
        """Apply a shared-model mutation at an atomic generation boundary.

        ``policy_version=None`` lets the scheduler derive ``live + 1`` under
        the policy lock, closing the read-compute-write race for callers
        that only need to advance by one.  The derived target version is
        handed to ``update`` so weight publishers can fan it out while
        still inside the lock.
        """
        with self._weight_lock:
            return self.backend.apply_weight_update(policy_version, update)

    def with_policy_snapshot(self, inspect: Callable[[int], T]) -> T:
        """Inspect a version stable against generator and scheduler updates."""
        if not callable(inspect):
            raise TypeError("inspect must be callable")
        with self._weight_lock:
            return self.backend.with_policy_snapshot(inspect)

    @torch.no_grad()
    def generate(
        self, batch: Dict, params: Optional[SamplingParams] = None
    ) -> RawRollout:
        """Expand prompts by ``group_size`` and generate one response each.

        ``params=None`` uses the generator's training defaults; passing a
        :class:`SamplingParams` instance overrides sampling for this call
        only (validation uses this to decode greedily, for example).

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
        effective = self.params if params is None else params
        with self._weight_lock:
            return self.backend.with_policy_snapshot(
                lambda generation_version: self._generate_eval(
                    batch, generation_version, effective
                )
            )

    def _generate_eval(
        self, batch: Dict, generation_version: int, params: SamplingParams
    ) -> RawRollout:
        prompt_texts, flat_prompt_ids = self._prepare_prompts(batch)
        B = len(prompt_texts)
        G = params.group_size
        # Re-expand flat list to G copies per prompt for run_batch.
        expanded_prompt_ids: List[List[int]] = []
        for ids in flat_prompt_ids:
            expanded_prompt_ids.extend([list(ids)] * G)

        sampling_options = {}
        if params.seed is not None:
            occurrences = Counter()
            seeds = []
            for ids in flat_prompt_ids:
                key = tuple(ids)
                occurrence = occurrences[key]
                occurrences[key] += 1
                seeds.extend(
                    request_seed(
                        params.seed, generation_version, ids, index, occurrence
                    )
                    for index in range(G)
                )
            sampling_options["request_seeds"] = seeds

        results = self.backend.generate(
            expanded_prompt_ids,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            frequency_penalty=params.frequency_penalty,
            rep_window=params.rep_window,
            return_logprobs=True,
            return_details=True,
            **sampling_options,
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

        device = self._output_device
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
        finish_reasons: List[List[str]] = [[] for _ in range(B)]
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
                finish_reasons[i].append(result.finish_reason)

        return RawRollout(
            prompts=prompts_tensor,
            prompt_mask=prompt_mask,
            responses=responses,
            response_mask=response_mask,
            logprobs_old=logprobs_old,
            policy_version=generation_version,
            prompt_texts=prompt_texts,
            response_texts=response_texts,
            finish_reasons=finish_reasons,
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
