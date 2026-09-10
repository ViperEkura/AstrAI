import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor

from astrai.config.inference_config import InferenceConfig
from astrai.extension.backend.attention import (
    CudaBackend,
    get_backend,
)
from astrai.inference.cache import PagePool, TaskCacheManager
from astrai.inference.runtime.graph import CudaGraphContext
from astrai.inference.runtime.sample import sample
from astrai.inference.task import Task
from astrai.inference.workspace import InferenceWorkspace
from astrai.model.automodel import AutoModel

logger = logging.getLogger(__name__)
_config = InferenceConfig()


@contextmanager
def timed(label: str, log: Optional[logging.Logger] = None):
    """GPU-precise timer via CUDA events; falls back to perf_counter on CPU."""
    log = log or logger
    if not log.isEnabledFor(logging.DEBUG):
        yield
        return
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        tic = time.perf_counter()
    yield
    if use_cuda:
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        elapsed_ms = (time.perf_counter() - tic) * 1000
    log.debug("%s %.2fms", label, elapsed_ms)


@dataclass
class SamplingBatchInfo:
    """Per-batch sampling parameters, cached across decode steps.

    Sampling params are constant for a given ordered task set, so they are
    built once (pinned-memory async H2D) and reused until the task set
    changes.  ``top_ks`` is int32 to match the native consumers.
    """

    temperatures: Tensor  # float32 [B]
    top_ks: Tensor  # int32  [B]
    top_ps: Tensor  # float32 [B]
    freq_penalties: Tensor  # float32 [B]
    has_freq: bool  # any frequency_penalty != 0 (avoids per-step GPU .any())


@dataclass
class DecodeSteadyState:
    """Cached decode metadata for the steady-state case.

    When the same ordered task set decodes one token per step, sampling
    params and task signature are reused; only positions advance by 1.
    ``last_tokens`` keeps that step's sampled ids on-device so the next
    step with an unchanged signature can fill ``input_ids`` via a
    device-to-device copy.
    """

    task_sig: tuple
    positions: list[int]
    sampling_info: SamplingBatchInfo
    last_tokens: Optional[Tensor] = None


def _build_sampling_batch_info(tasks: List[Task], device) -> SamplingBatchInfo:
    pin = str(device).startswith("cuda")
    freq_penalties = torch.tensor(
        [t.frequency_penalty for t in tasks], dtype=torch.float32, pin_memory=pin
    ).to(device, non_blocking=True)
    return SamplingBatchInfo(
        temperatures=torch.tensor(
            [t.temperature for t in tasks], dtype=torch.float32, pin_memory=pin
        ).to(device, non_blocking=True),
        top_ks=torch.tensor(
            [t.top_k for t in tasks], dtype=torch.int32, pin_memory=pin
        ).to(device, non_blocking=True),
        top_ps=torch.tensor(
            [t.top_p for t in tasks], dtype=torch.float32, pin_memory=pin
        ).to(device, non_blocking=True),
        freq_penalties=freq_penalties,
        has_freq=bool((freq_penalties != 0).any()),
    )


def _warmup_cuda_graphs(
    model: AutoModel,
    pool: PagePool,
    task_cache: TaskCacheManager,
    ws: InferenceWorkspace,
    gctx: CudaGraphContext,
    max_batch_size: int,
    prompt_len: int = 1,
    device: Optional[str] = None,
):
    dev = device or next(model.parameters()).device

    # Prefill warmup: cuBLAS auto-tunes for the actual prompt-length tensor
    # shapes on first call (F.linear is the dominant cost).  This also warms
    # up the CUDA context (driver init) and compiles the graph-capture trace
    # that follows.  Custom .so kernels do NOT need this — they are pre-built.
    warmup_len = _config.prefill_warmup_len
    tid = "_warmup_prefill"
    if task_cache.task_alloc(tid, list(range(warmup_len))):
        with (
            torch.inference_mode(),
            timed("warmup prefill", logger),
        ):
            kv = task_cache.bind([tid], ws, start_pos=0)
            ids_in = torch.arange(warmup_len, device=dev)
            pos_in = ids_in
            model(
                ids_in,
                kv_cache=kv,
                position_ids=pos_in,
                fwd="prefill",
                logits_positions=torch.tensor(
                    [warmup_len - 1], dtype=torch.long, device=dev
                ),
            )
        task_cache.task_free(tid)

    batch_sizes = [1]
    n = 2
    while n <= max_batch_size:
        batch_sizes.append(n)
        n *= 2
    if max_batch_size not in batch_sizes:
        batch_sizes.append(max_batch_size)

    for b in batch_sizes:
        task_ids = [f"_warmup_decode_{b}_{i}" for i in range(b)]
        prompt_tokens = [list(range(prompt_len)) for _ in range(b)]
        alloc_ok = True
        for tid, pt in zip(task_ids, prompt_tokens):
            if not task_cache.task_alloc(tid, pt):
                alloc_ok = False
                break
        if not alloc_ok:
            for tid in task_ids:
                task_cache.task_free(tid)
            continue

        with (
            torch.inference_mode(),
            timed(f"warmup decode b={b}", logger),
        ):
            for step in range(2):
                seq_pos = step
                ws.position_ids[:b] = seq_pos
                for tid in task_ids:
                    task_cache.task_extend(tid, seq_pos)
                kv = task_cache.bind(task_ids, ws)
                ids_buf = ws.fill_input_ids([step] * b)
                gctx.forward(
                    model,
                    key=(b,),
                    input_ids=ids_buf,
                    kv_cache=kv,
                    position_ids=ws.position_ids[:b],
                    fwd="decode",
                )

        for tid in task_ids:
            task_cache.task_free(tid)
        torch.cuda.synchronize()


class Executor:
    """Model forward passes for prefill and decode phases."""

    def __init__(
        self,
        model: AutoModel,
        kv_cache: PagePool,
        task_cache: TaskCacheManager,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        enable_cuda_graph: bool = True,
    ):
        self.model = model
        self.kv_cache = kv_cache
        self.task_cache = task_cache
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        # Per-step decode cache for the steady-state case (same ordered
        # task set decodes one token per step).  Sampling params stay
        # constant; only positions advance.
        self._decode_cache: Optional[DecodeSteadyState] = None

        # Pre-allocated fixed-shape buffers for the decode hot path
        # (input_ids, decode mask, KV bind metadata).  Eagerly sized at init
        # so the workspace is CUDA-graph-capture friendly — no allocation
        # during capture.
        config = model.config
        max_q_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        backend = get_backend()
        self._graph_supported = backend.supports_graph() and (
            CudaBackend.available() and head_dim in CudaBackend.HEAD_DIMS
        )
        self._workspace = InferenceWorkspace(
            max_batch_size=kv_cache.max_batch_size,
            max_seq_len=kv_cache.max_seq_len,
            max_q_heads=max_q_heads,
            head_dim=head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        # CUDA-graph capture: one graph per (batch_size,) key.
        # Enabled at init-time via _warmup_cuda_graphs for CudaBackend
        # on supported head_dims; left disabled otherwise.
        self._graph_ctx = CudaGraphContext()
        if enable_cuda_graph:
            self._try_enable_cuda_graph()

    def _try_enable_cuda_graph(self):
        if not self._graph_supported:
            return

        self._graph_ctx.set_enabled(True)
        _warmup_cuda_graphs(
            self.model,
            self.kv_cache,
            self.task_cache,
            self._workspace,
            self._graph_ctx,
            max_batch_size=self.kv_cache.max_batch_size,
            device=self.device,
        )

    @property
    def cuda_graph_enabled(self) -> bool:
        return self._graph_ctx.enabled and self._graph_supported

    def _sample_logits(
        self,
        logits: Tensor,
        tasks: List[Task],
        return_logprobs: bool = False,
        info: Optional[SamplingBatchInfo] = None,
    ):
        """Sample from ``logits`` and return ``(host_payload, tokens)``.

        ``host_payload`` is the scheduler-facing list (token ids, or
        ``(token_id, logprob)`` tuples with ``return_logprobs``);
        ``tokens`` is the ``[B]`` device tensor that produced it, kept
        for the steady-state decode fast path.
        """
        info = info or _build_sampling_batch_info(tasks, self.device)
        if info.has_freq:
            history_lists = [
                t.prompt_ids[-t.rep_window :] + t.output_ids for t in tasks
            ]
            history_lens = [len(ids) for ids in history_lists]
            max_len = max(history_lens, default=0)
            padded_ids = torch.zeros(
                len(tasks), max_len, dtype=torch.long, device=self.device
            )
            padded_mask = torch.zeros(
                len(tasks), max_len, dtype=torch.bool, device=self.device
            )
            for i, ids in enumerate(history_lists):
                length = len(ids)
                padded_ids[i, :length] = torch.as_tensor(
                    ids, dtype=torch.long, device=self.device
                )
                padded_mask[i, :length] = True
        else:
            padded_ids = None
            padded_mask = None

        result = sample(
            logits,
            temperature=info.temperatures,
            top_k=info.top_ks,
            top_p=info.top_ps,
            frequency_penalty=info.freq_penalties,
            input_ids=padded_ids,
            input_mask=padded_mask,
            return_logprobs=return_logprobs,
        )
        if not return_logprobs:
            return result.tolist(), result

        tokens, logprobs = result
        tokens_list = tokens.tolist()
        logprobs_list = logprobs.tolist()
        for task, logprob in zip(tasks, logprobs_list):
            task.output_logprobs.append(float(logprob))
        return list(zip(tokens_list, logprobs_list)), tokens

    def execute_prefill(
        self,
        tasks: List[Task],
        start_pos: int = 0,
        return_logprobs: bool = False,
    ):
        tasks = sorted(tasks, key=lambda t: t.task_id)
        batch_sz = len(tasks)

        # Validate batch size bounds
        if batch_sz > self._workspace.max_batch_size:
            raise ValueError(
                f"Batch size {batch_sz} exceeds max_batch_size "
                f"{self._workspace.max_batch_size}"
            )

        prompt_lens = [len(t.prompt_ids) for t in tasks]

        # Validate inputs before any resource allocation
        if any(start_pos >= prompt_len for prompt_len in prompt_lens):
            raise ValueError("prefill start_pos must precede every prompt end")

        q_lens = [prompt_len - start_pos for prompt_len in prompt_lens]

        input_ids = torch.tensor(
            [token for t in tasks for token in t.prompt_ids[start_pos:]],
            dtype=torch.long,
            device=self.device,
        )

        task_ids = [t.task_id for t in tasks]
        position_ids = torch.cat(
            [
                torch.arange(
                    start_pos, prompt_len, dtype=torch.long, device=self.device
                )
                for prompt_len in prompt_lens
            ]
        )

        # Last packed position per request; the model projects only these rows.
        last_token_indices = (
            torch.tensor(q_lens, dtype=torch.long, device=self.device).cumsum(0) - 1
        )

        with (
            torch.inference_mode(),
            timed(
                f"execute_prefill b={batch_sz} tokens={sum(q_lens)} "
                f"q_len={min(q_lens)}..{max(q_lens)}",
                logger,
            ),
        ):
            outputs = self.model(
                input_ids,
                position_ids=position_ids,
                kv_cache=self.task_cache.bind(
                    task_ids,
                    self._workspace,
                    start_pos=start_pos,
                ),
                fwd="prefill",
                logits_positions=last_token_indices,
            )
            logits = outputs["logits"]

        step_out, _ = self._sample_logits(logits, tasks, return_logprobs)
        return tasks, step_out

    def execute_decode(
        self, tasks: List[Task], return_logprobs: bool = False
    ) -> List[int]:
        """Decode next token for each task.

        Args:
            return_logprobs: When ``True``, also record (and return)
                the log-probability of each sampled token under the
                post-strategy sampling distribution.  The logprob is
                appended to ``task.output_logprobs`` and the return
                list becomes ``List[Tuple[int, float]]``.

        Returns:
            ``List[int]`` of sampled token IDs, or
            ``List[Tuple[int, float]]`` of ``(token_id, logprob)`` when
            ``return_logprobs`` is ``True``.
        """
        if not tasks:
            return []

        b = len(tasks)

        # Validate batch size bounds
        if b > self._workspace.max_batch_size:
            raise ValueError(
                f"Batch size {b} exceeds max_batch_size "
                f"{self._workspace.max_batch_size}"
            )

        ws = self._workspace
        task_ids = [t.task_id for t in tasks]
        task_sig = tuple(task_ids)
        cur_positions = [t.next_pos for t in tasks]

        # ---- pre-replay: update input buffers in-place ----

        # When the previous decode step ran this same ordered task set, its
        # sampled tokens are still on-device and map 1:1 onto the current
        # slots — fill input ids device-to-device.  inference_mode guards
        # the read because the source was produced under sampling's
        # inference-mode context.
        #
        # ``cache_valid`` checks the decode cache's own task signature:
        # task ids are globally unique, so equality alone proves the cached
        # tokens were sampled for exactly this ordered batch.  Req-index
        # signatures in the cache manager are deliberately NOT consulted —
        # they are recycled when freed slots are reallocated, which once
        # let a fresh batch replay a previous generation's tokens.
        cached = self._decode_cache
        cache_valid = cached is not None and cached.task_sig == task_sig
        if cache_valid and cached.last_tokens is not None:
            with torch.inference_mode():
                input_ids = ws.fill_input_ids_from_device(cached.last_tokens)
        else:
            input_ids = ws.fill_input_ids(
                [t.output_ids[-1] if t.output_ids else t.prompt_ids[-1] for t in tasks]
            )

        kv_cache = self.task_cache.bind(task_ids, ws)

        # Reuse sampling state only if all conditions hold:
        # 1. The cached decode state belongs to THIS task set (task_sig)
        # 2. KV bind detected steady increment (same req_indices, seq_lens +1)
        reuse_decode_state = cache_valid and self.task_cache.bind_was_steady
        if reuse_decode_state:
            info = cached.sampling_info
            ws.position_ids[:b] += 1
        else:
            info = _build_sampling_batch_info(tasks, self.device)
            ws.position_ids[:b].copy_(
                torch.tensor(cur_positions, dtype=torch.long, device=self.device)
            )
        self._decode_cache = DecodeSteadyState(task_sig, cur_positions, info)

        # ---- forward (graph replay or live run + capture) ----

        use_graph = (
            self._graph_ctx.enabled
            and self._graph_supported
            and get_backend().supports_graph()
        )
        key = (b,)
        with (
            torch.inference_mode(),
            timed(f"execute_decode forward b={b}", logger),
        ):
            if use_graph:
                outputs = self._graph_ctx.forward(
                    self.model,
                    key=key,
                    input_ids=input_ids,
                    kv_cache=kv_cache,
                    position_ids=ws.position_ids[:b],
                    fwd="decode",
                )
            else:
                outputs = self.model(
                    input_ids,
                    kv_cache=kv_cache,
                    position_ids=ws.position_ids[:b],
                    fwd="decode",
                )
            logits = outputs["logits"]

        step_out, tokens_dev = self._sample_logits(
            logits, tasks, return_logprobs, info=info
        )
        self._decode_cache.last_tokens = tokens_dev
        return step_out
