"""Unified inference engine for continuous batching."""

import asyncio
import gc
import threading
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from astrai.inference.core.cache import KVCache
from astrai.inference.core.scheduler import InferenceScheduler
from astrai.inference.core.task import STOP
from astrai.tokenize import AutoTokenizer


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
        with self._cond:
            self.tokens.append((idx, token))
            if token is not STOP:
                self.results[idx] += token
            else:
                if not self._done[idx]:
                    self._done[idx] = True
                    self._completed += 1
                    self._cond.notify_all()
            self._event.set()

    def pop_all(self) -> List[Tuple[int, str]]:
        with self._cond:
            out = self.tokens.copy()
            self.tokens.clear()
            if not out:
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


class GenerationRequest:
    """Request parameters for text generation."""

    def __init__(
        self,
        messages: List[Dict[str, str]],
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        frequency_penalty: float = 0.0,
        rep_window: int = 64,
        stream: bool = False,
    ):
        if not (isinstance(top_k, int) and top_k >= 0):
            raise ValueError("top_k must be a non-negative integer")
        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be a float between 0.0 and 1.0")
        if not (isinstance(temperature, (int, float)) and temperature > 0):
            raise ValueError("temperature must be a positive number")
        if not (
            isinstance(frequency_penalty, (int, float))
            and -2.0 <= frequency_penalty <= 2.0
        ):
            raise ValueError("frequency_penalty must be between -2.0 and 2.0")
        if not (isinstance(rep_window, int) and rep_window > 0):
            raise ValueError("rep_window must be a positive integer")

        self.messages = messages
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.rep_window = rep_window
        self.stream = stream


class InferenceEngine:
    """Unified inference engine backed by continuous-batching scheduler."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        max_batch_size: int = 1,
        max_seq_len: Optional[int] = None,
        max_prompt_len: int = 2048,
        page_size: int = 128,
        cache: Optional[KVCache] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = InferenceScheduler(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            max_prompt_len=max_prompt_len,
            cache=cache,
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

        if stream:
            return self._generate_streaming(
                prompts,
                is_batch,
                max_tokens,
                temperature,
                top_p,
                top_k,
                frequency_penalty,
                rep_window,
            )
        else:
            return self._generate_non_streaming(
                prompts,
                is_batch,
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
        sync_gen = self._generate_streaming(
            [prompt],
            False,
            max_tokens,
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            rep_window,
        )

        async def _agen():
            loop = asyncio.get_event_loop()
            while True:
                token = await loop.run_in_executor(None, self._next_token, sync_gen)
                if token is None:
                    break
                yield token

        return _agen()

    @staticmethod
    def _next_token(gen: Generator) -> Optional[str]:
        try:
            return next(gen)
        except StopIteration:
            return None

    def generate_with_request(
        self, request: GenerationRequest
    ) -> Union[Generator[str, None, None], str, List[str]]:
        prompt = self.tokenizer.apply_chat_template(request.messages, tokenize=False)
        return self.generate(
            prompt=prompt,
            stream=request.stream,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            frequency_penalty=request.frequency_penalty,
            rep_window=request.rep_window,
        )

    def _submit_tasks(
        self,
        prompts: List[str],
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        top_k: int,
        frequency_penalty: float,
        rep_window: int,
    ) -> Tuple[GenerateResult, List[str]]:
        n = len(prompts)
        result = GenerateResult(count=n)
        task_ids = []
        for i, p in enumerate(prompts):
            cb = self._make_callback(result, i)
            task_id = self.scheduler.add_task(
                prompt=p,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                rep_window=rep_window,
                stream_callback=cb,
            )
            task_ids.append(task_id)
        return result, task_ids

    @staticmethod
    def _make_callback(result: GenerateResult, idx: int):
        def cb(token):
            result.append(token, idx)

        return cb

    def _generate_streaming(
        self,
        prompts: List[str],
        is_batch: bool,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        top_k: int,
        frequency_penalty: float,
        rep_window: int,
    ) -> Generator:
        result, task_ids = self._submit_tasks(
            prompts,
            max_tokens,
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            rep_window,
        )
        n = len(prompts)
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
                for tid in task_ids:
                    self.scheduler.remove_task(tid)

        return gen()

    def _generate_non_streaming(
        self,
        prompts: List[str],
        is_batch: bool,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        top_k: int,
        frequency_penalty: float,
        rep_window: int,
    ) -> Union[str, List[str]]:
        result, task_ids = self._submit_tasks(
            prompts,
            max_tokens,
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            rep_window,
        )

        try:
            result.wait_completion()
        except TimeoutError:
            for tid in task_ids:
                self.scheduler.remove_task(tid)
            raise

        for tid in task_ids:
            self.scheduler.remove_task(tid)

        res = result.get_results()
        return res if is_batch else res[0]

    def get_stats(self) -> Dict[str, Any]:
        return self.scheduler.get_stats()

    def shutdown(self):
        self.scheduler.stop()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
