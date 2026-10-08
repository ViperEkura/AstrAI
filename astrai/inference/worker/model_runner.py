import logging
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch
from torch import Tensor

from astrai.config.inference_config import InferenceConfig
from astrai.extension import attn_backend
from astrai.extension.backend.attention import (
    CudaBackend,
    get_backend,
)

if TYPE_CHECKING:
    # Type-only dependency: the worker consumes the KV manager's ``bind``
    # interface; importing it at runtime would create a core↔worker cycle.
    from astrai.inference.core.cache.pool import BlockPool, KVCacheManager
from astrai.inference.contracts import ExecutionRequest, SchedulerOutput
from astrai.inference.sampling_rng import sampling_uniform
from astrai.inference.worker.graph import CUDAGraphRunner
from astrai.inference.worker.pending import (
    PendingExecution,
    ResultRing,
)
from astrai.inference.worker.sample import (
    SamplingMeta,
    SamplingPipeline,
    build_sampling_pipeline,
    sample,
)
from astrai.inference.worker.workspace import InferenceWorkspace
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

    Sampling params are constant for a given ordered request set, so they are
    built once (pinned-memory async H2D) and reused until the request set
    changes.  ``top_ks`` is int32 to match the native consumers.

    ``meta`` holds the host-resolved facts (greedy, active filters) derived
    from the same request attributes, and ``pipeline`` the strategy chain
    built from them — both constant for the same ordered request set, so the
    steady-state decode path reuses the whole bundle instead of rebuilding
    strategy objects and re-probing device tensors every step.
    """

    temperatures: Tensor  # float32 [B]
    top_ks: Tensor  # int32  [B]
    top_ps: Tensor  # float32 [B]
    freq_penalties: Tensor  # float32 [B]
    has_freq: bool  # any frequency_penalty != 0 (avoids per-step GPU .any())
    meta: Optional[SamplingMeta] = None
    pipeline: Optional[SamplingPipeline] = None
    seeded: bool = False


@dataclass
class DecodeSteadyState:
    """Cached decode metadata for the steady-state case.

    When the same ordered request set decodes one token per step, sampling
    params and request signature are reused; only positions advance by 1.
    ``last_tokens`` keeps that step's sampled ids on-device so the next
    step with an unchanged signature can fill ``input_ids`` via a
    device-to-device copy.
    """

    task_sig: Any
    positions: List[int]
    sampling_info: SamplingBatchInfo
    last_tokens: Optional[Tensor] = None


def _build_sampling_batch_info(
    requests: List[ExecutionRequest], device
) -> SamplingBatchInfo:
    pin = str(device).startswith("cuda")
    freq_list = [t.frequency_penalty for t in requests]
    temps = [t.temperature for t in requests]
    top_ks = [t.top_k for t in requests]
    top_ps = [t.top_p for t in requests]
    seeded = [t.sampling.seed is not None for t in requests]
    if any(seeded) and not all(seeded):
        raise ValueError("seeded sampling requires a seed for every batch request")
    freq_penalties = torch.tensor(freq_list, dtype=torch.float32, pin_memory=pin).to(
        device, non_blocking=True
    )
    # Host-side any()/all(): the values came from the request list, so checking
    # them on device would force a synchronize right after the non-blocking
    # H2D copies — draining whatever prefill work is still queued
    # (measured 0.6 s stall at batch 128).
    has_freq = any(f != 0.0 for f in freq_list)
    meta = SamplingMeta(
        greedy=all(t == 0.0 for t in temps),
        any_temp_not_one=any(t != 1.0 for t in temps),
        max_top_k=max(top_ks, default=0),
        any_topp_lt1=any(tp < 1.0 for tp in top_ps),
        has_freq=has_freq,
    )
    temperatures = torch.tensor(temps, dtype=torch.float32, pin_memory=pin).to(
        device, non_blocking=True
    )
    top_ks_t = torch.tensor(top_ks, dtype=torch.int32, pin_memory=pin).to(
        device, non_blocking=True
    )
    top_ps_t = torch.tensor(top_ps, dtype=torch.float32, pin_memory=pin).to(
        device, non_blocking=True
    )
    info = SamplingBatchInfo(
        temperatures=temperatures,
        top_ks=top_ks_t,
        top_ps=top_ps_t,
        freq_penalties=freq_penalties,
        has_freq=has_freq,
        meta=meta,
        seeded=all(seeded),
    )
    info.pipeline = build_sampling_pipeline(
        temperatures,
        top_ks_t,
        top_ps_t,
        freq_penalties,
        meta=meta,
    )
    return info


def _warmup_cuda_graphs(
    model: AutoModel,
    pool: "BlockPool",
    cache_mgr: "KVCacheManager",
    ws: InferenceWorkspace,
    gctx: CUDAGraphRunner,
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
    if cache_mgr.alloc_slots(tid, list(range(warmup_len))):
        with (
            torch.inference_mode(),
            timed("warmup prefill", logger),
        ):
            kv = cache_mgr.bind([tid], ws, start_pos=0)
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
        cache_mgr.free_slots(tid)

    batch_sizes = [1]
    n = 2
    while n <= max_batch_size:
        batch_sizes.append(n)
        n *= 2
    if max_batch_size not in batch_sizes:
        batch_sizes.append(max_batch_size)

    for b in batch_sizes:
        request_ids = [f"_warmup_decode_{b}_{i}" for i in range(b)]
        prompt_tokens = [list(range(prompt_len)) for _ in range(b)]
        alloc_ok = True
        for tid, pt in zip(request_ids, prompt_tokens):
            if not cache_mgr.alloc_slots(tid, pt):
                alloc_ok = False
                break
        if not alloc_ok:
            for tid in request_ids:
                cache_mgr.free_slots(tid)
            continue

        with (
            torch.inference_mode(),
            timed(f"warmup decode b={b}", logger),
        ):
            for step in range(2):
                seq_pos = step
                ws.position_ids[:b] = seq_pos
                for tid in request_ids:
                    cache_mgr.extend_slots(tid, seq_pos)
                kv = cache_mgr.bind(request_ids, ws)
                ids_buf = ws.fill_input_ids([step] * b)
                gctx.forward(
                    model,
                    key=(b,),
                    input_ids=ids_buf,
                    kv_cache=kv,
                    position_ids=ws.position_ids[:b],
                    fwd="decode",
                )

        for tid in request_ids:
            cache_mgr.free_slots(tid)
        torch.cuda.synchronize()


class GPUModelRunner:
    """Model forward passes for prefill and decode phases (vLLM GPUModelRunner counterpart)."""

    def __init__(
        self,
        model: AutoModel,
        kv_cache: "BlockPool",
        cache_mgr: "KVCacheManager",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        enable_cuda_graph: bool = True,
    ):
        self.model = model
        self.kv_cache = kv_cache
        self.kv_manager = cache_mgr
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        # Per-step decode cache for the steady-state case (same ordered
        # request set decodes one token per step).  Sampling params stay
        # constant; only positions advance.
        self._decode_cache: Optional[DecodeSteadyState] = None

        # The most recent submitted-but-not-necessarily-committed decode
        # step.  ``submit_decode`` consults it to refuse the one unsafe
        # overlap (frequency penalty reading stale host histories); the
        # SchedulerStep's commit phase clears it.  Synchronous callers commit
        # immediately, so it is usually already consumed.
        self._pending: Optional[PendingExecution] = None

        # Async result relay: pinned depth-2 slots on a dedicated copy
        # stream.  A submitted step's tokens land here without a blocking
        # tolist; commit waits on the posted event instead.  Inert on CPU.
        self._result_ring = ResultRing(kv_cache.max_batch_size, self.device)

        # Pre-allocated fixed-shape buffers for the decode hot path
        # (input_ids, decode mask, KV bind metadata).  Eagerly sized at init
        # so the workspace is CUDA-graph-capture friendly — no allocation
        # during capture.
        config = model.config
        max_q_heads = config.num_attention_heads
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
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
        self._graph_ctx = CUDAGraphRunner()
        if enable_cuda_graph:
            self._try_enable_cuda_graph()

    def _try_enable_cuda_graph(self):
        if not self._graph_supported:
            return

        self._graph_ctx.set_enabled(True)
        _warmup_cuda_graphs(
            self.model,
            self.kv_cache,
            self.kv_manager,
            self._workspace,
            self._graph_ctx,
            max_batch_size=self.kv_cache.max_batch_size,
            device=self.device,
        )

    @property
    def cuda_graph_enabled(self) -> bool:
        return self._graph_ctx.enabled and self._graph_supported

    def execute_prefill(
        self,
        requests: List[ExecutionRequest],
        start_pos: int = 0,
        return_logprobs: bool = False,
        num_tokens: Optional[List[int]] = None,
        *,
        plan: SchedulerOutput,
    ):
        """Prefill (a chunk of) each request's prompt; sample only at chunk ends.

        With ``num_tokens=None`` this is the historical whole-remaining-prompt
        call.  The chunked-prefill path passes one window per request:
        ``(start_pos, num_tokens[i])`` covers the tokens this step computes.
        A request whose window does not reach its prompt end is a
        **continuation chunk** — its forward runs purely to materialize KV,
        ``logits_positions`` skips it, and no token is sampled for it (the
        caller advances ``num_computed_tokens`` instead); the final chunk of
        each request samples its next token from the window's last position.
        """
        # Sort for a deterministic packed order and carry the per-request
        # windows along: num_tokens arrives in the CALLER's request order.
        order = sorted(range(len(requests)), key=lambda i: requests[i].request_id)
        requests = [requests[i] for i in order]
        num_tokens = [num_tokens[i] for i in order] if num_tokens else None
        batch_sz = len(requests)

        # Validate batch size bounds
        if batch_sz > self._workspace.max_batch_size:
            raise ValueError(
                f"Batch size {batch_sz} exceeds max_batch_size "
                f"{self._workspace.max_batch_size}"
            )

        prompt_lens = [len(t.prompt_ids) for t in requests]
        if num_tokens is None:
            num_tokens = [prompt_len - start_pos for prompt_len in prompt_lens]
        if len(num_tokens) != batch_sz:
            raise ValueError("num_tokens must supply one window length per request")

        # Validate inputs before any resource allocation
        for prompt_len, n in zip(prompt_lens, num_tokens):
            if start_pos >= prompt_len or n <= 0 or start_pos + n > prompt_len:
                raise ValueError(
                    "prefill window must lie strictly inside the prompt: "
                    f"start_pos={start_pos} n={n} prompt_len={prompt_len}"
                )

        # Only the requests whose window reaches the prompt end sample a
        # token; continuation chunks run forward for KV materialization.
        sample_mask = [
            start_pos + n == prompt_len
            for prompt_len, n in zip(prompt_lens, num_tokens)
        ]

        input_ids = torch.tensor(
            [
                token
                for t, n in zip(requests, num_tokens)
                for token in t.prompt_ids[start_pos : start_pos + n]
            ],
            dtype=torch.long,
            device=self.device,
        )

        request_ids = [t.request_id for t in requests]
        position_ids = torch.cat(
            [
                torch.arange(
                    start_pos, start_pos + n, dtype=torch.long, device=self.device
                )
                for n in num_tokens
            ]
        )

        # Last packed position per SAMPLING request; the model projects
        # only these rows.  ``logits_positions`` indexes the PACKED rows,
        # so a sampling request's offset must include the windows of every
        # earlier request in the batch — sampling and continuation chunks
        # alike (continuation windows occupy packed rows too).
        if any(sample_mask):
            packed_ends = torch.tensor(num_tokens, dtype=torch.long).cumsum(0) - 1
            last_token_indices = packed_ends[
                torch.tensor(sample_mask, dtype=torch.bool)
            ].to(self.device)
        else:
            last_token_indices = torch.empty(0, dtype=torch.long, device=self.device)

        with (
            torch.inference_mode(),
            timed(
                f"execute_prefill b={batch_sz} tokens={sum(num_tokens)} "
                f"q_len={min(num_tokens)}..{max(num_tokens)}",
                logger,
            ),
        ):
            outputs = self.model(
                input_ids,
                position_ids=position_ids,
                kv_cache=self.kv_manager.bind(
                    request_ids,
                    self._workspace,
                    start_pos=start_pos,
                    seq_ends=[start_pos + n for n in num_tokens],
                ),
                fwd="prefill",
                logits_positions=last_token_indices,
            )
            logits = outputs["logits"]

        sampling_requests = [t for t, m in zip(requests, sample_mask) if m]
        if sampling_requests:
            pending = self._submit_sample(
                logits, sampling_requests, return_logprobs, plan=plan
            )
        else:
            pending = PendingExecution(snapshot=plan)
        # Even a KV-only chunk must complete before publishing pages/freeing KV.
        if torch.device(self.device).type == "cuda":
            pending.completion_event = torch.cuda.Event()
            pending.completion_event.record()
        return requests, pending

    def execute_score(
        self,
        requests: List[ExecutionRequest],
        per_token: bool = False,
    ) -> List[Any]:
        """Teacher-forced log-probabilities for a batch of prompts.

        Each request's ``prompt_ids`` is the *whole* scored sequence
        (context + continuation); ``request.cont_len`` says how many trailing
        tokens of it are the continuation to score.  The model is projected
        only at the positions that predict those tokens (``logits_positions``),
        so nothing is sampled and the softmax runs over ``sum(cont_len)`` rows
        rather than over the whole batch.

        Returns per request either the summed log-probability of its continuation
        or, with ``per_token=True``, the list of per-token log-probabilities.
        """
        if not requests:
            return []
        batch_sz = len(requests)
        if batch_sz > self._workspace.max_batch_size:
            raise ValueError(
                f"Batch size {batch_sz} exceeds max_batch_size "
                f"{self._workspace.max_batch_size}"
            )

        prompt_lens = [len(t.prompt_ids) for t in requests]
        cont_lens = [t.cont_len for t in requests]
        if any(c <= 0 for c in cont_lens):
            raise ValueError("every scored request needs a non-empty continuation")
        if any(c >= p for c, p in zip(cont_lens, prompt_lens)):
            raise ValueError("continuation must be shorter than the scored sequence")

        offsets = torch.tensor(prompt_lens, dtype=torch.long).cumsum(0) - torch.tensor(
            prompt_lens, dtype=torch.long
        )
        # Position p predicts the token at p+1, so a continuation occupying
        # logical positions [P-C, P) is read from logits at [P-C-1, P-1).
        positions: List[int] = []
        flat_targets: List[int] = []
        for offset, prompt_len, cont_len, request in zip(
            offsets.tolist(), prompt_lens, cont_lens, requests
        ):
            start = offset + prompt_len - cont_len - 1
            positions.extend(range(start, start + cont_len))
            flat_targets.extend(request.prompt_ids[-cont_len:])

        logits_positions = torch.tensor(positions, dtype=torch.long, device=self.device)
        input_ids = torch.tensor(
            [token for t in requests for token in t.prompt_ids],
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.cat(
            [
                torch.arange(prompt_len, dtype=torch.long, device=self.device)
                for prompt_len in prompt_lens
            ]
        )
        request_ids = [t.request_id for t in requests]

        with torch.inference_mode():
            outputs = self.model(
                input_ids,
                position_ids=position_ids,
                kv_cache=self.kv_manager.bind(
                    request_ids, self._workspace, start_pos=0
                ),
                fwd="prefill",
                logits_positions=logits_positions,
            )
            logits = outputs["logits"]

        tgt = torch.tensor(flat_targets, dtype=torch.long, device=self.device)
        logprobs = (
            torch.nn.functional.log_softmax(logits.float(), dim=-1)
            .gather(1, tgt.unsqueeze(-1))
            .squeeze(-1)
        )

        out: List[Any] = []
        cursor = 0
        for cont_len in cont_lens:
            chunk = logprobs[cursor : cursor + cont_len].tolist()
            cursor += cont_len
            out.append(chunk if per_token else sum(chunk))
        return out

    def submit_decode(
        self,
        requests: List[ExecutionRequest],
        return_logprobs: bool = False,
        *,
        plan: SchedulerOutput,
    ) -> Optional[PendingExecution]:
        """Launch one decode step without resolving any value on host.

        The submit half of the decode contract: fills input buffers, binds
        KV, replays the forward and launches sampling, then hands back a
        :class:`PendingExecution` carrying the device-resident tokens (and
        optional logprobs).  Nothing here touches host copies of the
        sampled values or mutates request output state — that is exclusively
        :meth:`PendingExecution.commit`'s job, invoked by the SchedulerStep's commit
        phase.

        Frequency penalty needs each request's host-side output history at
        submit time, so a pending (not yet committed) previous step for the
        same batch would be read stale; that combination is rejected here
        (``None``) and the caller falls back to commit-then-submit.
        """
        if not requests:
            return None

        b = len(requests)

        # Validate batch size bounds
        if b > self._workspace.max_batch_size:
            raise ValueError(
                f"Batch size {b} exceeds max_batch_size "
                f"{self._workspace.max_batch_size}"
            )

        ws = self._workspace
        request_ids = [t.request_id for t in requests]
        task_sig = tuple(t.identity for t in requests)
        cur_positions = [t.next_pos for t in requests]

        # ---- pre-replay: update input buffers in-place ----

        # When the previous decode step ran this same ordered request set, its
        # sampled tokens are still on-device and map 1:1 onto the current
        # slots — fill input ids device-to-device.  inference_mode guards
        # the read because the source was produced under sampling's
        # inference-mode context.
        #
        # ``cache_valid`` checks the decode cache's own request signature:
        # request ids are globally unique, so equality alone proves the cached
        # tokens were sampled for exactly this ordered batch.  Req-index
        # signatures in the cache manager are deliberately NOT consulted —
        # they are recycled when freed slots are reallocated, which once
        # let a fresh batch replay a previous generation's tokens.
        cached = self._decode_cache
        cache_valid = cached is not None and cached.task_sig == task_sig
        pending_same_batch = (
            self._pending is not None
            and not self._pending.committed
            and self._pending.snapshot.identities == task_sig
        )
        if cache_valid and cached.last_tokens is not None:
            with torch.inference_mode():
                input_ids = ws.fill_input_ids_from_device(cached.last_tokens)
        else:
            input_ids = ws.fill_input_ids([t.input_token_id for t in requests])

        kv_cache = self.kv_manager.bind(request_ids, ws)

        # Reuse sampling state only if all conditions hold:
        # 1. The cached decode state belongs to THIS request set (task_sig)
        # 2. KV bind detected steady increment (same req_indices, seq_lens +1)
        reuse_decode_state = cache_valid and self.kv_manager.bind_was_steady
        if reuse_decode_state:
            info = cached.sampling_info
            ws.position_ids[:b] += 1
        else:
            info = _build_sampling_batch_info(requests, self.device)
            ws.position_ids[:b].copy_(
                torch.tensor(cur_positions, dtype=torch.long, device=self.device)
            )
        if info.has_freq and pending_same_batch:
            # Frequency penalty subtracts per-token counts computed from the
            # host history (prompt tail + full output). With a pending step
            # in flight the histories are one step behind the device relay,
            # so the penalty would be computed against the wrong prefix.
            return None
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

        pending = self._submit_sample(
            logits, requests, return_logprobs, info=info, plan=plan
        )
        self._result_ring.post(pending)
        self._decode_cache.last_tokens = pending.tokens
        self._pending = pending
        return pending

    def _submit_sample(
        self,
        logits: Tensor,
        requests: List[ExecutionRequest],
        return_logprobs: bool = False,
        info: Optional[SamplingBatchInfo] = None,
        *,
        plan: SchedulerOutput,
    ) -> PendingExecution:
        """Sample from ``logits`` into a :class:`PendingExecution`.

        Same strategy pipeline and frequency-penalty history assembly as
        the pre-split host path, but the result stays on device inside a
        pending step instead of being resolved to host lists.
        """
        info = info or _build_sampling_batch_info(requests, self.device)
        if info.has_freq:
            history_lists = [
                t.prompt_ids[-t.rep_window :] + t.output_ids for t in requests
            ]
            history_lens = [len(ids) for ids in history_lists]
            max_len = max(history_lens, default=0)
            padded_ids = torch.zeros(
                len(requests), max_len, dtype=torch.long, device=self.device
            )
            padded_mask = torch.zeros(
                len(requests), max_len, dtype=torch.bool, device=self.device
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

        uniforms = None
        if info.seeded and not info.meta.greedy:
            uniforms = torch.tensor(
                [
                    sampling_uniform(t.sampling.seed, t.sampling_position)
                    for t in requests
                ],
                dtype=torch.float32,
                pin_memory=str(self.device).startswith("cuda"),
            ).to(self.device, non_blocking=True)

        result = (
            info.pipeline.sample(
                logits,
                input_ids=padded_ids,
                input_mask=padded_mask,
                return_logprobs=return_logprobs,
                uniforms=uniforms,
            )
            if info.pipeline is not None
            else sample(
                logits,
                temperature=info.temperatures,
                top_k=info.top_ks,
                top_p=info.top_ps,
                frequency_penalty=info.freq_penalties,
                input_ids=padded_ids,
                input_mask=padded_mask,
                return_logprobs=return_logprobs,
                meta=info.meta,
                uniforms=uniforms,
            )
        )
        if return_logprobs:
            tokens, logprobs = result
        else:
            tokens, logprobs = result, None

        return PendingExecution(
            snapshot=plan,
            sampled_identities=tuple(t.identity for t in requests),
            tokens=tokens,
            logprobs=logprobs,
        )

    def peek_pending(self) -> Optional[PendingExecution]:
        """The in-flight submitted step, if any (not committed)."""
        return self._pending

    def clear_pending(self) -> Optional[PendingExecution]:
        """Detach the in-flight step without committing it.

        The overlap scheduler's commit phase takes ownership this way: the
        executor's ``_pending`` slot frees up for the *next* submit while
        the scheduler still holds (and commits) the returned step.
        """
        pending = self._pending
        self._pending = None
        return pending

    def can_overlap_submit(self) -> bool:
        """Whether the next submit may overlap the in-flight step.

        Frequency penalty is the one refusal: its per-step history rebuild
        reads host-side ``output_ids``, which a pending step has not yet
        produced — the overlap would penalise against a stale prefix.
        """
        pending = self._pending
        if pending is None or pending.committed:
            return True
        cached = self._decode_cache
        return cached is None or not getattr(cached.sampling_info, "has_freq", False)

    def execute_model(self, plan: SchedulerOutput):
        """Yield every launched group's handle, including KV-only prefills.

        Yielding immediately transfers resource ownership to EngineCore even
        when a later backend group fails. The runner never commits core state.
        """
        groups = {}
        for request in plan.requests:
            key = (
                request.phase,
                request.backend,
                request.position if request.phase == "prefill" else None,
            )
            groups.setdefault(key, []).append(request)
        for (phase, backend, start_pos), requests in groups.items():
            subplan = plan.select(requests)
            with attn_backend(backend) if backend is not None else nullcontext():
                if phase == "prefill":
                    _, pending = self.execute_prefill(
                        requests,
                        start_pos=start_pos,
                        num_tokens=[r.num_tokens for r in requests],
                        return_logprobs=plan.return_logprobs,
                        plan=subplan,
                    )
                else:
                    pending = self.submit_decode(
                        requests, plan.return_logprobs, plan=subplan
                    )
                    if pending is None:
                        raise RuntimeError("decode submit requires a drained history")
            yield pending

    def synchronize(self) -> None:
        """Fence work that threw before it could return an execution handle."""
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)

    def execute_decode(
        self,
        requests: List[ExecutionRequest],
        return_logprobs: bool = False,
        *,
        plan: SchedulerOutput,
    ) -> List[int]:
        pending = self.submit_decode(requests, return_logprobs, plan=plan)
        if pending is None:
            raise RuntimeError("execute_decode: submit refused")
        payload = pending.commit()
        if return_logprobs:
            return [(r.token_id, r.logprob) for r in payload.results]
        return [r.token_id for r in payload.results]
