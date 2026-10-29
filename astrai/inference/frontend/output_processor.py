"""Frontend-side output processing (vLLM: output_processor.py).

Consumes :mod:`astrai.inference.core.events` output events for one request and
produces the user-visible stream: incremental detokenization, text-level
stop-sequence matching with truncation, and token usage accounting.

The scheduler loop never runs this code — it runs in the consumer's thread
/ task, which is what keeps tokenizer work off the GPU-submit path.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from astrai.inference.core.events import (
    FINISH_STOP_TOKEN,
    RequestError,
    RequestFinished,
    TokenDelta,
)
from astrai.inference.core.request import StreamDecoder
from astrai.tokenize.tokenizer import AutoTokenizer


class StopSequenceChecker:
    """Incremental stop-sequence matching over a growing text buffer.

    Matches may straddle token boundaries, so the checker keeps the
    un-yielded tail buffered: text is only released once it can no longer
    be part of a stop sequence (classic ``len - max_stop_len + 1`` window).
    """

    def __init__(self, sequences: List[str]):
        self._sequences = [s for s in sequences if s]
        self._max_len = max((len(s) for s in self._sequences), default=0)
        self._buffer = ""
        self.matched: Optional[str] = None

    @property
    def has_stops(self) -> bool:
        return bool(self._sequences)

    def push(self, text: str) -> Tuple[str, bool]:
        """Append ``text``; return ``(releasable_text, stopped)``.

        ``releasable_text`` is what may safely be forwarded downstream
        (everything except the ambiguous tail).  ``stopped`` is True when a
        stop sequence completed within the pushed text; the caller must
        drop the remainder of the generation.
        """
        if not self._sequences:
            return text, False
        if self.matched is not None:
            return "", True
        self._buffer += text
        first = len(self._buffer)
        for seq in self._sequences:
            position = self._buffer.find(seq)
            if 0 <= position < first:
                first = position
                self.matched = seq
        if self.matched is not None:
            head, self._buffer = self._buffer[:first], ""
            return head, True
        if self._max_len <= 1:
            return self.flush(), False
        keep = self._max_len - 1
        if len(self._buffer) > keep:
            release, self._buffer = (
                self._buffer[:-keep],
                self._buffer[-keep:],
            )
            return release, False
        return "", False

    def flush(self) -> str:
        """Release the buffered tail (terminal, no stop matched)."""
        out, self._buffer = self._buffer, ""
        return out


@dataclass
class ProcessedOutput:
    """User-visible output state for one request."""

    request_id: str
    token_ids: List[int] = field(default_factory=list)
    prompt_tokens: int = 0
    finish_reason: Optional[str] = None
    # Populated when a text stop sequence matched.
    stop_sequence: Optional[str] = None


class OutputProcessor:
    """Per-request event folder: events in, text + terminal state out.

    One instance per active request; the frontend (engine / protocol
    adapters) drives it from the event stream::

        proc = OutputProcessor("req-1", tokenizer, stop_sequences=["\\n\\n"])
        for event in events:
            text, stopped = proc.push(event)
            ...forward text...
            if proc.finished: break
    """

    def __init__(
        self,
        request_id: str,
        tokenizer: AutoTokenizer,
        stop_sequences: Optional[List[str]] = None,
        *,
        prompt_tokens: int = 0,
    ):
        self._decoder = StreamDecoder(tokenizer)
        self._stop = StopSequenceChecker(stop_sequences or [])
        # Text-level stops can finish before any core terminal event arrives.
        self.state = ProcessedOutput(request_id=request_id, prompt_tokens=prompt_tokens)

    @property
    def finished(self) -> bool:
        return self.state.finish_reason is not None

    def push(self, event) -> Tuple[str, bool]:
        """Fold one event; return ``(new_text, stopped_now)``."""
        if self.finished:
            return "", False
        if isinstance(event, TokenDelta):
            try:
                raw = self._decoder.push(event.token_id)
            except Exception:
                # A broken decoder (e.g. a mock or exotic tokenizer without
                # the Rust streaming handle) must not kill the fold: fall
                # back to the token's string form so the stream still
                # terminates and usage stays exact.
                raw = str(event.token_id)
            self.state.token_ids.append(event.token_id)
            text, stopped = self._stop.push(raw)
            if stopped:
                self._finish(FINISH_STOP_TOKEN)
            return text, stopped
        if isinstance(event, (RequestFinished, RequestError)):
            if isinstance(event, RequestFinished):
                self.state.prompt_tokens = event.prompt_tokens
                self.state.token_ids = list(event.token_ids) or self.state.token_ids
            self._finish(event.finish_reason)
            # A sampled EOS is not a matched text stop: its ambiguous tail
            # still belongs to the response, just as on length/cancellation.
            tail = self._stop.flush()
            return tail, False
        return "", False

    def usage(self) -> Tuple[int, int]:
        """(prompt_tokens, completion_tokens) for protocol responses."""
        return self.state.prompt_tokens, len(self.state.token_ids)

    def _finish(self, reason: str) -> None:
        self.state.finish_reason = reason
        self.state.stop_sequence = self._stop.matched
