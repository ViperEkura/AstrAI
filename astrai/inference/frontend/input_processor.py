"""Frontend-side input processing (vLLM: input_processor.py).

Tokenizes and validates generation inputs on the caller's thread, minting
the request id before anything is handed to the scheduler.  Building the
id on the frontend is what makes the "events for a request the frontend
has not bound yet" window impossible — the core can emit events from the
moment the request exists, and the consumer is already registered.
"""

import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

from astrai.tokenize.tokenizer import AutoTokenizer


@dataclass(frozen=True)
class ProcessedInput:
    """Normalized input for one generation request."""

    request_id: str
    prompt_ids: List[int]
    # Text form kept for callbacks that render the prompt side.
    prompt_text: Optional[str] = None


class InputProcessor:
    """Tokenize prompts and mint request ids (frontend side)."""

    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int):
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len

    def new_request_id(self) -> str:
        return f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    def process(
        self,
        prompt: str,
        *,
        add_special_tokens: bool = True,
    ) -> ProcessedInput:
        """Tokenize one prompt into a :class:`ProcessedInput`.

        Empty encodes
        are refused (a zero-token prompt never completes prefill), and
        over-length prompts are truncated to the model window from the
        left (keep the tail).
        """
        ids = self._tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        if ids and isinstance(ids[0], list):  # batched tokenizer shape
            ids = ids[0]
        return self._make_processed_input(prompt, ids)

    def _make_processed_input(self, prompt: str, ids: List[int]) -> ProcessedInput:
        if not ids:
            raise ValueError("prompt encoded to zero tokens; refusing to schedule")
        if len(ids) > self._max_seq_len:
            ids = ids[-self._max_seq_len :]
        return ProcessedInput(
            request_id=self.new_request_id(),
            prompt_ids=list(ids),
            prompt_text=prompt,
        )

    def process_batch(self, prompts: List[str]) -> List[ProcessedInput]:
        """Batched variant: one ``encode`` call for the whole list."""
        if not prompts:
            return []
        encoded = self._tokenizer.encode(list(prompts))
        if not isinstance(encoded, list):
            raise ValueError("batch tokenizer returned unexpected shape")
        # Tokenizers return either [[ids], ...] (batched) or a flat [ids]
        # when they do not distinguish single from batch input; accept both
        # by re-encoding per-prompt in the flat case.
        if len(encoded) != len(prompts) or not all(
            isinstance(ids, list) for ids in encoded
        ):
            encoded = [self._coerce_single(self._tokenizer.encode(p)) for p in prompts]
        return [
            self._make_processed_input(prompt, ids)
            for prompt, ids in zip(prompts, encoded)
        ]

    @staticmethod
    def _coerce_single(ids) -> List[int]:
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
