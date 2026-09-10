"""Unified inference engine for continuous batching."""

import asyncio
import gc
import logging
import threading
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from astrai.extension import ATTN_BACKEND, AttentionBackend, get_backend
from astrai.inference.cache import PagePool
from astrai.inference.scheduler import InferenceScheduler
from astrai.inference.task import STOP, BatchedStreamCallback
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer

logger = logging.getLogger(__name__)


class GenerateResult:
    """Thread-safe token accumulator for streaming and non-streaming modes."""

    def __init__(self, count: int = 1):
        self._cond = threading.Condition()
        self._event = threading.Event()
        self.tokens: List[Tuple[int, str]] = []
        self.results: List[str] = [""] * count
        self._done: List[bool] = [False] * count
        self._completed = 0
        self._total = count

    def append(self, token: str, idx: int = 0):
        self.append_batch([(idx, token)])

    def append_batch(self, items: List[Tuple[int, Any]]) -> None:
        """Append multiple ``(idx, token)`` events under one lock/notify.

        Batched counterpart to :meth:`append` for per-step delivery: state
        updates for every event happen under a single condition hold and
        waiters are woken once per batch instead of once per token.
        """
        if not items:
            return
        with self._cond:
            for idx, token in items:
                self.tokens.append((idx, token))
                if token is STOP:
                    if not self._done[idx]:
                        self._done[idx] = True
                        self._completed += 1
                        self._cond.notify_all()
                else:
                    self.results[idx] += token
            self._event.set()

    def pop_all(self) -> List[Tuple[int, str]]:
        with self._cond:
            out = self.tokens.copy()
            self.tokens.clear()
            self._event.clear()
            return out

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._event.wait(timeout=timeout)

    def wait_completion(self, timeout: float = 300.0):
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._completed >= self._total, timeout=timeout
            ):
                raise TimeoutError(
                    f"Generation timeout after {timeout}s "
                    f"({self._completed}/{self._total} completed)"
                )

    def get_results(self) -> List[str]:
        with self._cond:
            return self.results.copy()


class _ResultSink(BatchedStreamCallback):
    """Batched stream channel from the scheduler into one GenerateResult.

    Registered as the ``stream_callback`` for every task of a single
    ``generate`` call, so the scheduler's one dispatch per decode step
    maps to one ``append_batch`` (one lock, one waiter wake). A task can
    start decoding the moment ``add_task`` returns — before the engine
    learns its id — so events for ids not yet bound are buffered and
    replayed on ``bind``.
    """

    def __init__(self, result: GenerateResult):
        self._result = result
        self._lock = threading.Lock()
        self._index_of: Dict[str, int] = {}
        self._pending: List[Tuple[str, Any]] = []

    def bind(self, task_id: str, idx: int) -> None:
        with self._lock:
            self._index_of[task_id] = idx
            replay = [(idx, token) for tid, token in self._pending if tid == task_id]
            if replay:
                self._pending = [
                    (tid, token) for tid, token in self._pending if tid != task_id
                ]
        if replay:
            self._result.append_batch(replay)

    def __call__(self, events: List[Tuple[str, Any]]) -> None:
        with self._lock:
            items: List[Tuple[int, Any]] = []
            for tid, token in events:
                idx = self._index_of.get(tid)
                if idx is None:
                    self._pending.append((tid, token))
                else:
                    items.append((idx, token))
        if items:
            self._result.append_batch(items)


class InferenceEngine:
    """Unified inference engine backed by continuous-batching scheduler."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 1,
        max_seq_len: Optional[int] = None,
        cache: Optional[PagePool] = None,
        enable_cuda_graph: bool = True,
        backend: Optional[Union[str, ATTN_BACKEND, AttentionBackend, type]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = InferenceScheduler(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            cache=cache,
            enable_cuda_graph=enable_cuda_graph,
            backend=backend,
        )

        self.scheduler.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

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
        request_backend = get_backend(use_default=False)
        result = GenerateResult()
        sink = _ResultSink(result)
        task_id = self.scheduler.add_task(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            rep_window=rep_window,
            backend=request_backend,
            stream_callback=sink,
        )
        sink.bind(task_id, 0)

        async def _agen():
            finished = False
            try:
                while not finished:
                    for _idx, token in result.pop_all():
                        if token is STOP:
                            finished = True
                            break
                        yield token
                    if not finished:
                        await asyncio.to_thread(result.wait, 0.05)
            finally:
                if not finished:
                    self.scheduler.cancel_task(task_id)

        return _agen()

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
        request_backend = get_backend(use_default=False)
        result = GenerateResult(count=n)
        sink = _ResultSink(result)
        task_ids = []
        for i, p in enumerate(prompts):
            task_id = self.scheduler.add_task(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                rep_window=rep_window,
                backend=request_backend,
                stream_callback=sink,
            )
            sink.bind(task_id, i)
            task_ids.append(task_id)

        if not stream:
            try:
                result.wait_completion()
            except TimeoutError:
                for tid in task_ids:
                    self.scheduler.cancel_task(tid)
                raise
            res = result.get_results()
            return res if is_batch else res[0]

        remaining = n
        finished = [False] * n

        def gen():
            nonlocal remaining
            try:
                while remaining > 0:
                    items = result.pop_all()
                    for idx, token in items:
                        if token is STOP:
                            if not finished[idx]:
                                finished[idx] = True
                                remaining -= 1
                        else:
                            yield (idx, token) if is_batch else token
                    if remaining > 0:
                        result.wait(timeout=0.05)
            finally:
                for idx, task_id in enumerate(task_ids):
                    if not finished[idx]:
                        self.scheduler.cancel_task(task_id)

        return gen()

    def get_stats(self) -> Dict[str, Any]:
        return self.scheduler.get_stats()

    @property
    def backend_name(self) -> str:
        return self.scheduler.backend_name

    @property
    def cuda_graph_enabled(self) -> bool:
        return self.scheduler.cuda_graph_enabled

    def shutdown(self):
        self.scheduler.stop()
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
    Extra *engine_kwargs* (cache, enable_cuda_graph, backend) pass
    through to InferenceEngine. Placement parts left as None are skipped.
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
