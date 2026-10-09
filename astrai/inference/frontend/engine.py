"""Unified inference engine for continuous batching (vLLM-style frontend).

The engine is the frontend layer: it owns input processing (tokenization,
request-id minting), output processing (detokenization, stop handling,
usage) and the request tracker that bridges the scheduler's output events
to consumers.  The scheduler loop itself only emits token ids and terminal
facts — it never renders text and never runs consumer code.
"""

import asyncio
import gc
import logging
import time
import warnings
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Union,
)

import torch
import torch.nn as nn

from astrai.extension import ATTN_BACKEND, AttentionBackend, get_backend
from astrai.inference.core.cache.pool import BlockPool
from astrai.inference.core.events import RequestError, TokenDelta
from astrai.inference.core.scheduler import Scheduler
from astrai.inference.frontend.core_client import EngineCoreClient, InprocClient
from astrai.inference.frontend.input_processor import InputProcessor, ProcessedInput
from astrai.inference.frontend.output_processor import OutputProcessor
from astrai.inference.frontend.tracking import (
    RequestTracker,
    StreamChunk,
    map_finish_reason,
)
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer

logger = logging.getLogger(__name__)

# Whole-batch budget of a blocking generate().
_GENERATE_TIMEOUT_S = 300.0


class InferenceEngine:
    """Unified inference engine backed by continuous-batching scheduler."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 1,
        max_seq_len: Optional[int] = None,
        cache: Optional[BlockPool] = None,
        enable_cuda_graph: bool = True,
        backend: Optional[Union[str, ATTN_BACKEND, AttentionBackend, type]] = None,
        enable_overlap: bool = False,
        core_client: Optional[EngineCoreClient] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if core_client is None:
            scheduler = Scheduler(
                model=self.model,
                tokenizer=self.tokenizer,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                cache=cache,
                enable_cuda_graph=enable_cuda_graph,
                backend=backend,
                enable_overlap=enable_overlap,
            )
            core_client = InprocClient(scheduler)
        self._core = core_client
        self._tracker = RequestTracker()
        self._core.set_event_sink(self._tracker.sink)
        resolved_len = max_seq_len
        if resolved_len is None:
            cfg_len = getattr(model.config, "max_position_embeddings", None)
            resolved_len = cfg_len if cfg_len is not None else 4096
        self._max_seq_len = int(resolved_len)
        self._input_processor = InputProcessor(tokenizer, int(resolved_len))

        self._core.start()

    @property
    def scheduler(self) -> Scheduler:
        warnings.warn(
            "InferenceEngine.scheduler is deprecated; use EngineCoreClient "
            "for request, scoring, status and lifecycle operations",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(self._core, InprocClient):
            raise AttributeError("The injected core client has no local scheduler")
        return self._core.scheduler

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    # ---- frontend internals ----

    def _submit_prompt(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        top_k: int,
        frequency_penalty: float,
        rep_window: int,
        backend: Optional[AttentionBackend],
    ) -> ProcessedInput:
        """Tokenize + submit one prompt; the id exists before the core does."""
        processed = self._input_processor.process(prompt)
        self._tracker.register(processed.request_id, maxlen=self._max_seq_len + 1)
        try:
            self._core.send_request(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                rep_window=rep_window,
                request_id=processed.request_id,
                prompt_ids=processed.prompt_ids,
                backend=backend,
            )
        except BaseException:
            self._release_requests([processed.request_id])
            raise
        return processed

    def _release_requests(self, request_ids: List[str]) -> None:
        """Cancel unfinished core work and always discard frontend bookkeeping."""
        for request_id in request_ids:
            try:
                # Frontend text stops are not core terminal events. Conversely,
                # a terminal already queued by the core needs no extra abort,
                # even if the consumer has not folded that event yet.
                if not self._tracker.is_finished(request_id):
                    self._core.abort_request(request_id)
            except Exception:
                logger.exception("request cancellation failed for %s", request_id)
            finally:
                self._tracker.unregister(request_id)

    # ---- public API (signatures unchanged) ----

    def generate(
        self,
        prompt: Union[str, List[str]],
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        frequency_penalty: float = 0.0,
        rep_window: int = 64,
    ) -> Union[Generator, str, List[str]]:
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        if max_tokens is not None and max_tokens <= 0:
            if stream:
                return iter(())
            results = [""] * len(prompts)
            return results if is_batch else results[0]

        return self._generate(
            prompts,
            is_batch,
            stream,
            max_tokens,
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            rep_window,
        )

    def score(
        self,
        prompt: Union[str, List[int], List[Union[str, List[int]]]],
        continuation: Union[str, List[int], List[Union[str, List[int]]]],
        per_token: bool = False,
    ) -> Union[float, None, List[Any]]:
        """Teacher-forced log-probability of ``continuation`` given ``prompt``.

        The engine's entry point for log-likelihood metrics.  Nothing is
        sampled: the sequence is prefilled as ``prompt + continuation`` and
        only the continuation's own tokens are scored, projected at exactly
        the positions that predict them.  Callers therefore never build an
        attention mask -- a 2-D one used to switch causality off here.

        Strings are tokenized the way the eval scripts tokenize a scored pair:
        the prompt keeps its special tokens, the continuation does not.  Pass
        token ids to control the boundary exactly (tokenizing a pair jointly
        can differ from tokenizing the two sides).

        Args:
            prompt: one context, or a list of contexts.
            continuation: the matching continuation(s).
            per_token: return per-token log-probabilities instead of the sum.

        Returns:
            A ``float`` for a single pair, or a list for a batch.  ``None``
            marks a pair that cannot be scored (empty side, or the sequence
            reaching the engine's ``max_seq_len``).

        Note:
            Synchronous and not re-entrant: like ``run_batch`` it drives the
            executor on the calling thread, so do not overlap it with
            generation on the same engine.
        """

        # A list of ints is one prompt given as token ids; a list of strings or
        # of id-lists is a batch.
        def _is_batch(side) -> bool:
            return isinstance(side, list) and (
                not side or isinstance(side[0], (str, list))
            )

        is_batch = _is_batch(prompt)
        if is_batch != _is_batch(continuation):
            raise ValueError(
                "prompt and continuation must both be single or both batches"
            )
        if is_batch and len(prompt) != len(continuation):
            raise ValueError("prompt and continuation batches must have equal length")
        if not is_batch:
            prompt, continuation = [prompt], [continuation]

        def _encode(side: str, **kwargs) -> List[int]:
            out = self.tokenizer.encode(side, **kwargs)
            # Tokenizers in this repo return a flat list for a bare string;
            # accept the batched shape too rather than depending on that.
            if out and isinstance(out[0], list):
                out = out[0]
            return list(out)

        def _ids(side) -> List[int]:
            return _encode(side) if isinstance(side, str) else list(side)

        def _cont_ids(side) -> List[int]:
            return (
                _encode(side, add_special_tokens=False)
                if isinstance(side, str)
                else list(side)
            )

        prompts = [_ids(p) for p in prompt]
        conts = [_cont_ids(c) for c in continuation]

        # The executor holds max_batch_size worth of fixed-shape buffers, so
        # split a long batch here rather than failing deep in the executor.
        chunk = max(1, self._core.max_batch_size)
        results: List[Any] = []
        for start in range(0, len(prompts), chunk):
            results.extend(
                self._core.score_ids(
                    prompts[start : start + chunk],
                    conts[start : start + chunk],
                    per_token=per_token,
                )
            )
        return results if is_batch else results[0]

    def generate_async(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        frequency_penalty: float = 0.0,
        rep_window: int = 64,
    ) -> AsyncGenerator[str, None]:
        chunks = self.generate_events(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            rep_window=rep_window,
        )

        async def _agen():
            try:
                async for chunk in chunks:
                    if chunk.text:
                        yield chunk.text
            finally:
                await chunks.aclose()

        return _agen()

    @staticmethod
    def _set_terminal_chunk(chunk: StreamChunk, processor: OutputProcessor) -> None:
        chunk.finish_reason = map_finish_reason(processor.state.finish_reason)
        chunk.prompt_tokens = processor.state.prompt_tokens
        chunk.completion_tokens = len(processor.state.token_ids)
        chunk.stop_sequence = processor.state.stop_sequence

    def generate_events(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        frequency_penalty: float = 0.0,
        rep_window: int = 64,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator["StreamChunk", None]:
        """Async generation yielding structured chunks (vLLM OutputProcessor).

        Each chunk carries the incremental ``text`` plus the running token
        id delta and, on the final chunk, the request usage and mapped
        ``finish_reason``.  Protocol adapters consume this instead of
        re-tokenizing text to count tokens.
        """
        # Capture the caller's backend, but defer submission until consumption:
        # closing a never-started generator cannot run its finally block.
        backend = get_backend(use_default=False)

        async def _agen():
            processed = self._submit_prompt(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                rep_window=rep_window,
                backend=backend,
            )
            request_id = processed.request_id
            tracker = self._tracker
            terminal_chunk = None
            try:
                processor = OutputProcessor(
                    request_id,
                    self.tokenizer,
                    stop_sequences=stop_sequences,
                    prompt_tokens=len(processed.prompt_ids),
                )
                queue, pending = tracker.subscribe_async(
                    request_id, asyncio.get_running_loop()
                )
                while not processor.finished:
                    events = pending if pending else await queue.get()
                    pending = []
                    for event in events:
                        text, stopped = processor.push(event)
                        delta_ids = (
                            [event.token_id] if isinstance(event, TokenDelta) else []
                        )
                        if text or stopped or processor.finished:
                            chunk = StreamChunk(
                                text=text,
                                delta_token_ids=delta_ids,
                                current_token_ids=list(processor.state.token_ids),
                                stopped=stopped,
                            )
                            if processor.finished:
                                self._set_terminal_chunk(chunk, processor)
                                terminal_chunk = chunk
                                break
                            yield chunk
            finally:
                self._release_requests([request_id])
            # Release before yielding the terminal chunk: callers may pause or
            # stop iteration here without resuming the generator again.
            if terminal_chunk is not None:
                yield terminal_chunk

        return _agen()

    def _collect_blocking(
        self, request_ids: List[str], is_batch: bool
    ) -> Union[str, List[str]]:
        """Wait for terminal events, then decode each request's tokens once."""
        bulk_decode = getattr(self.tokenizer, "_tokenizer", None) is not None
        return self._collect_blocking_events(request_ids, is_batch, bulk_decode)

    def _collect_blocking_events(
        self, request_ids: List[str], is_batch: bool, bulk_decode: bool
    ) -> Union[str, List[str]]:
        deadline = time.monotonic() + _GENERATE_TIMEOUT_S
        try:
            for rid in request_ids:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._tracker.wait(rid, timeout=remaining):
                    completed = sum(
                        1 for r in request_ids if self._tracker.is_finished(r)
                    )
                    raise TimeoutError(
                        f"Generation timeout after {_GENERATE_TIMEOUT_S}s "
                        f"({completed}/{len(request_ids)} completed)"
                    )

            results = [""] * len(request_ids)
            for idx, rid in enumerate(request_ids):
                try:
                    events = self._tracker.drain(rid)
                    if not bulk_decode:
                        results[idx] = self._fold_events_incrementally(rid, events)
                        continue

                    token_ids = []
                    for event in events:
                        if isinstance(event, RequestError):
                            results[idx] = self._fold_events_incrementally(rid, events)
                            break
                        if isinstance(event, TokenDelta):
                            token_ids.append(event.token_id)
                    else:
                        try:
                            results[idx] = self.tokenizer.decode(
                                token_ids, skip_special_tokens=True
                            )
                        except Exception:
                            logger.exception("bulk output decode failed for %s", rid)
                            results[idx] = self._fold_events_incrementally(rid, events)
                except Exception:
                    logger.exception("output fold failed for %s", rid)
            return results if is_batch else results[0]
        finally:
            self._release_requests(request_ids)

    def _fold_events_incrementally(self, request_id: str, events: List[Any]) -> str:
        proc = OutputProcessor(request_id, self.tokenizer)
        for event in events:
            if isinstance(event, RequestError):
                logger.error(
                    "request %s failed (%s): %s",
                    request_id,
                    event.error_code,
                    event.message,
                )
                break
            proc.push(event)
        return proc.state.text

    def _generate(
        self,
        prompts: List[str],
        is_batch: bool,
        stream: bool,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        top_k: int,
        frequency_penalty: float,
        rep_window: int,
    ) -> Union[Generator, str, List[str]]:
        n = len(prompts)
        # One batched tokenize on the caller's thread, ids minted up front;
        # the scheduler receives the same ids so events can never precede
        # their consumer.  Queues are sized for the longest possible
        # generation: the non-streaming fold runs once, at completion, so
        # each request's events accumulate until then.
        processed = self._input_processor.process_batch(prompts)
        request_ids = [item.request_id for item in processed]
        backend = get_backend(use_default=False)

        def submit():
            for rid in request_ids:
                self._tracker.register(rid, maxlen=self._max_seq_len + 1)
            try:
                self._core.send_requests(
                    prompts=prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    rep_window=rep_window,
                    request_ids=request_ids,
                    prompts_ids=[item.prompt_ids for item in processed],
                    backend=backend,
                )
            except BaseException:
                self._release_requests(request_ids)
                raise

        if not stream:
            submit()
            return self._collect_blocking(request_ids, is_batch)

        def gen():
            submit()
            try:
                remaining = n
                finished = [False] * n
                idx_of = {rid: i for i, rid in enumerate(request_ids)}
                processors = {
                    rid: OutputProcessor(rid, self.tokenizer) for rid in request_ids
                }
                while remaining > 0:
                    progressed = False
                    for rid in request_ids:
                        if finished[idx_of[rid]]:
                            continue
                        proc = processors[rid]
                        for event in self._tracker.drain(rid):
                            progressed = True
                            text, _ = proc.push(event)
                            if text:
                                yield (idx_of[rid], text) if is_batch else text
                            if proc.finished:
                                finished[idx_of[rid]] = True
                                remaining -= 1
                                break
                    if remaining > 0 and not progressed:
                        waiting_id = next(
                            rid for rid in request_ids if not finished[idx_of[rid]]
                        )
                        self._tracker.wait(waiting_id, timeout=0.05)
            finally:
                self._release_requests(request_ids)

        return gen()

    def get_stats(self) -> Dict[str, Any]:
        return self._core.stats()

    @property
    def backend_name(self) -> str:
        return self._core.backend_name

    @property
    def cuda_graph_enabled(self) -> bool:
        return self._core.cuda_graph_enabled

    def shutdown(self):
        self._core.shutdown()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def build_engine(
    param_path: Optional[Union[str, Path]] = None,
    *,
    model: Optional[nn.Module] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    device: Optional[str] = "cuda",
    dtype: Optional[torch.dtype] = torch.bfloat16,
    max_batch_size: int = 16,
    max_seq_len: Optional[int] = None,
    **engine_kwargs: Any,
) -> InferenceEngine:
    """Composition root for inference assembly.

    Loads model and tokenizer from *param_path*, or accepts preloaded
    objects, places the model, and returns a started InferenceEngine.
    Extra *engine_kwargs* (cache, enable_cuda_graph, backend, enable_overlap)
    pass through to InferenceEngine. Placement parts left as None are skipped.
    """
    if param_path is not None:
        if model is not None or tokenizer is not None:
            raise ValueError("pass either param_path or model+tokenizer, not both")
        path = Path(param_path)
        if not path.exists():
            raise FileNotFoundError(f"Parameter directory not found: {path}")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path)
    elif model is None or tokenizer is None:
        raise ValueError("build_engine requires param_path or both model and tokenizer")

    placement: Dict[str, Any] = {}
    if device is not None:
        placement["device"] = device
    if dtype is not None:
        placement["dtype"] = dtype
    if placement:
        model.to(**placement)
        logger.info(
            f"Model placed on {placement.get('device')} "
            f"with dtype {placement.get('dtype')}"
        )

    return InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        **engine_kwargs,
    )
