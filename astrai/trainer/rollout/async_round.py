"""One learner coordinating versioned online GRPO rollout rounds."""

import hashlib
import logging
import multiprocessing as mp
import os
import pickle
import time
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import Callable, Dict, Iterator, List, Optional

import torch
from torch import nn

from astrai.trainer.rollout.batching import merge_rollouts, slice_batch
from astrai.trainer.rollout.nccl_transport import NCCLWeightChannel
from astrai.trainer.rollout.protocol import (
    GenerationRequest,
    MessageKind,
    RolloutMessage,
    RolloutProtocolError,
    recv_message,
    send_message,
)
from astrai.trainer.rollout.runner import _score_rewards
from astrai.trainer.rollout.types import (
    RawRollout,
    RolloutResult,
    RolloutVersionError,
    SamplingParams,
)
from astrai.trainer.rollout.worker import RolloutWorkerSpec, run_rollout_worker

logger = logging.getLogger(__name__)


def _response_seeds(
    base_seed: int,
    sample_cursor: int,
    prompts: int,
    group_size: int,
    rank: Optional[int] = None,
) -> List[int]:
    return [
        int.from_bytes(
            hashlib.blake2b(
                (
                    f"{base_seed}:{sample_cursor + i}:{g}"
                    if rank is None
                    else f"{base_seed}:{rank}:{sample_cursor + i}:{g}"
                ).encode(),
                digest_size=8,
            ).digest(),
            "little",
        )
        & ((1 << 63) - 1)
        for i in range(prompts)
        for g in range(group_size)
    ]


@dataclass
class PendingRound:
    total: int
    jobs: Dict[int, List[int]]
    version: int
    round_id: int
    started: float
    request_ids: Dict[int, int]
    seeds: List[int]


class AsyncRoundCoordinator:
    """Bounded one-round lookahead with killable rollout processes."""

    def __init__(
        self,
        source: nn.Module,
        model_fn,
        param_path: str,
        devices: List[str],
        params: SamplingParams,
        reward_model,
        policy_version: int,
        max_batch_size: int,
        max_seq_len: Optional[int],
        model_dtype: torch.dtype,
        max_policy_lag: int = 1,
        max_prompts_per_worker: int = 1,
        worker_timeout_s: float = 600.0,
        worker_target=None,
        weight_channel=None,
        startup_timeout_s: float = 300.0,
        weight_timeout_s: Optional[float] = None,
        random_seed: int = 3407,
        sample_cursor: int = 0,
        learner_rank: Optional[int] = None,
        enable_overlap: bool = True,
    ):
        try:
            pickle.dumps(model_fn)
        except (pickle.PickleError, AttributeError, TypeError) as exc:
            raise ValueError("async_round model_fn must be pickleable") from exc
        self.reward_model = reward_model
        self._policy_version = policy_version
        self.max_policy_lag = max_policy_lag
        self.max_prompts_per_worker = max_prompts_per_worker
        self.worker_timeout_s = worker_timeout_s
        self.weight_timeout_s = (
            worker_timeout_s if weight_timeout_s is None else weight_timeout_s
        )
        self.random_seed = random_seed
        self._sample_cursor = sample_cursor
        self.learner_rank = learner_rank
        self._group_size = params.group_size
        self._devices = list(devices)
        self._round_id = 0
        self._request_id = 0
        self._closed = False
        self.generated_tokens = 0
        self.generation_seconds = 0.0
        self.learner_wait_seconds = 0.0
        self.peak_gpu_memory = {}
        self.worker_cuda_graph_enabled = {}
        self.weights = weight_channel or NCCLWeightChannel(source, policy_version)
        self._nccl_timeout_s = self.weight_timeout_s
        self._ctx = mp.get_context("spawn")
        self._processes = []
        self._pipes = []
        self._worker_versions = []
        self._active_round = None
        self._worker_target = worker_target or run_rollout_worker
        logger.info(
            "async rollout: learner=%s workers=%s CUDA_VISIBLE_DEVICES=%s sampling=%s startup_timeout=%s generation_timeout=%s weight_timeout=%s seed=%s cursor=%s",
            next(source.parameters()).device,
            devices,
            os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            params,
            startup_timeout_s,
            worker_timeout_s,
            self.weight_timeout_s,
            random_seed,
            sample_cursor,
        )
        if next(source.parameters()).device.type == "cuda":
            logger.info(
                "rollout GPU UUIDs: %s",
                {
                    str(device): str(
                        getattr(
                            torch.cuda.get_device_properties(device),
                            "uuid",
                            "unavailable",
                        )
                    )
                    for device in [next(source.parameters()).device]
                    + [torch.device(name) for name in devices]
                },
            )
        try:
            startup_deadline = time.monotonic() + startup_timeout_s
            self._rendezvous_port = self.weights.prepare_rendezvous(
                len(devices) + 1, startup_timeout_s
            )
            for index, device in enumerate(devices):
                parent, child = self._ctx.Pipe(duplex=True)
                spec = RolloutWorkerSpec(
                    device=device,
                    model_fn=model_fn,
                    param_path=param_path,
                    params=params,
                    max_batch_size=max_batch_size,
                    max_seq_len=max_seq_len,
                    policy_version=policy_version,
                    model_dtype=str(model_dtype).split(".")[-1],
                    rank=index + 1,
                    world_size=len(devices) + 1,
                    rendezvous_port=self._rendezvous_port,
                    nccl_timeout_s=self._nccl_timeout_s,
                    layout=self.weights.layout,
                    max_prompts_per_worker=max_prompts_per_worker,
                    enable_overlap=enable_overlap,
                )
                process = self._ctx.Process(
                    target=self._worker_target,
                    args=(child, spec),
                )
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self._pipes.append(parent)
                self._processes.append(process)
                self._worker_versions.append(None)
            self._collect_replies(
                list(range(len(devices))),
                MessageKind.MODEL_READY,
                startup_deadline,
                request_ids={index: 0 for index in range(len(devices))},
                version=policy_version,
            )
            self._sync_weights(policy_version)
            ready = self._collect_replies(
                list(range(len(devices))),
                MessageKind.READY,
                startup_deadline,
                request_ids={index: 0 for index in range(len(devices))},
                version=policy_version,
            )
            for index, message in ready.items():
                payload = message.payload
                self._worker_versions[index] = message.policy_version
                self.worker_cuda_graph_enabled[devices[index]] = (
                    payload.cuda_graph_enabled
                )
                self.peak_gpu_memory[devices[index]] = payload.peak_gpu_memory
        except BaseException:
            self.close(force=True)
            raise

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def _collect_replies(
        self,
        indices,
        kind,
        deadline,
        request_ids,
        round_id=None,
        version=None,
        *,
        accept_ready_on_expiry=False,
    ):
        pending = set(indices)
        messages = {}
        while pending:
            remaining = deadline - time.monotonic()
            handles = [self._pipes[index] for index in pending]
            handles.extend(self._processes[index].sentinel for index in pending)
            # Lookahead results can be ready while reward or learning runs.
            # Expiry permits collecting buffered replies, never another wait.
            ready = (
                wait(handles, timeout=max(0.0, remaining))
                if remaining > 0 or accept_ready_on_expiry
                else []
            )
            if not ready:
                if remaining <= 0:
                    raise TimeoutError(
                        f"async rollout {kind} timed out on workers {sorted(pending)}"
                    )
                continue
            for index in list(pending):
                conn = self._pipes[index]
                if conn in ready:
                    try:
                        message = recv_message(conn)
                    except (EOFError, OSError) as exc:
                        raise RuntimeError(
                            f"async rollout worker {index} exited"
                        ) from exc
                    if message.kind == MessageKind.ERROR:
                        raise RuntimeError(
                            f"async rollout worker {index} failed:\n{message.payload}"
                        )
                    try:
                        message.expect(
                            kind,
                            request_ids[index],
                            round_id=round_id,
                            policy_version=version,
                        )
                    except RolloutProtocolError as exc:
                        raise RolloutProtocolError(
                            f"async rollout worker {index}: {exc}"
                        ) from exc
                    messages[index] = message
                    pending.remove(index)
            for index in pending:
                if self._processes[index].sentinel in ready:
                    raise RuntimeError(f"async rollout worker {index} exited")
        return messages

    def _send_command(self, index, kind, *, round_id=None, version=None, payload=None):
        if not self._processes[index].is_alive():
            raise RuntimeError(f"async rollout worker {index} exited")
        self._request_id += 1
        request_id = self._request_id
        send_message(
            self._pipes[index],
            RolloutMessage(kind, request_id, round_id, version, payload),
        )
        return request_id

    def _sync_weights(self, version):
        missing = [
            index
            for index in range(len(self._processes))
            if self._worker_versions[index] != version
        ]
        if not missing:
            return
        if len(missing) != len(self._processes):
            raise RuntimeError("NCCL weight broadcast requires all workers")
        if self.weights.version != version:
            raise RuntimeError("learner weights have not committed requested version")
        self.weights.begin_fanout(missing)
        started = time.perf_counter()
        request_ids = {
            index: self._send_command(index, MessageKind.WEIGHT_SYNC, version=version)
            for index in missing
        }
        self.weights.connect(
            rank=0,
            world_size=len(self._processes) + 1,
            port=self._rendezvous_port,
            timeout_s=self._nccl_timeout_s,
        )
        self.weights.broadcast()
        acks = self._collect_replies(
            missing,
            MessageKind.WEIGHT_SYNC_ACK,
            time.monotonic() + self.weight_timeout_s,
            request_ids,
            version=version,
        )
        for index in acks:
            self.weights.acknowledge(index)
            self._worker_versions[index] = version
        self.weights.fanout_seconds += time.perf_counter() - started

    def apply_weight_update(self, policy_version, update):
        target = self._policy_version + 1 if policy_version is None else policy_version
        if target <= self._policy_version:
            raise ValueError("policy_version must advance")
        result = update(target)
        self._policy_version = target
        self.weights.mark_committed(target)
        return result

    def step(self):
        """Compatibility with BaseStrategy.optimizer_step."""

    def iter_rounds(
        self, batches, stop_requested: Callable[[], bool]
    ) -> Iterator[RolloutResult]:
        """Prefetch one round while the current round is scored and learned."""
        try:
            if stop_requested():
                return
            batches = iter(batches)
            batch = next(batches, None)
            if batch is None:
                return
            handle = self.submit_round(batch)
            while batch is not None and not stop_requested():
                try:
                    raw = self._collect_raw(handle)
                except RolloutVersionError:
                    # A stale round contributes no gradient or sample accounting.
                    handle = self.submit_round(batch, seeds=handle.seeds)
                    raw = self._collect_raw(handle)

                next_batch = None if stop_requested() else next(batches, None)
                next_handle = (
                    self.submit_round(next_batch) if next_batch is not None else None
                )
                rewards = _score_rewards(self.reward_model, raw).to(device="cpu")
                yield RolloutResult.from_raw(raw, rewards)
                batch, handle = next_batch, next_handle
        except BaseException:
            # A failed reward or abandoned iterator must retire its lookahead.
            try:
                self.close(force=True)
            except Exception:
                logger.exception("Failed to close rollout workers after round failure")
            raise

    def submit_round(
        self, batch: Dict, seeds: Optional[List[int]] = None
    ) -> PendingRound:
        if self._closed:
            raise RuntimeError("async rollout coordinator is closed")
        if self._active_round is not None:
            raise RuntimeError("only one rollout round may be in flight")
        total = len(next(iter(batch.values())))
        if total == 0:
            raise ValueError("async rollout round cannot be empty")
        fresh = seeds is None
        if fresh:
            seeds = _response_seeds(
                self.random_seed,
                self._sample_cursor,
                total,
                self._group_size,
                self.learner_rank,
            )
        if len(seeds) != total * self._group_size:
            raise ValueError("rollout seeds must match the prompt/response count")
        version = self._policy_version
        jobs = {
            index: list(range(index, total, len(self._processes)))
            for index in range(min(total, len(self._processes)))
        }
        try:
            self._sync_weights(version)
            self._round_id += 1
            request_ids = {}
            for index, indices in jobs.items():
                request_ids[index] = self._send_command(
                    index,
                    MessageKind.GENERATE,
                    round_id=self._round_id,
                    version=version,
                    payload=GenerationRequest(
                        slice_batch(batch, indices, total),
                        [
                            seeds[i * self._group_size + g]
                            for i in indices
                            for g in range(self._group_size)
                        ],
                    ),
                )
            handle = PendingRound(
                total,
                jobs,
                version,
                self._round_id,
                time.monotonic(),
                request_ids,
                seeds,
            )
            self._active_round = handle
            if fresh:
                self._sample_cursor += total
            return handle
        except BaseException:
            self.close(force=True)
            raise

    def collect_round(self, handle: PendingRound) -> RolloutResult:
        """Collect and score one round without starting another generation."""
        raw = self._collect_raw(handle)
        rewards = _score_rewards(self.reward_model, raw).to(device="cpu")
        return RolloutResult.from_raw(raw, rewards)

    def _collect_raw(self, handle: PendingRound) -> RawRollout:
        if handle is not self._active_round:
            raise RuntimeError("round handle is not active")
        wait_started = time.perf_counter()
        try:
            messages = self._collect_replies(
                list(handle.jobs),
                MessageKind.RESULT,
                handle.started + self.worker_timeout_s,
                handle.request_ids,
                round_id=handle.round_id,
                version=handle.version,
                accept_ready_on_expiry=True,
            )
        except BaseException:
            self.close(force=True)
            raise
        finally:
            self._active_round = None
        self.learner_wait_seconds += time.perf_counter() - wait_started
        parts = []
        durations = []
        for index, indices in handle.jobs.items():
            message = messages[index]
            payload = message.payload
            if payload.rollout.policy_version != handle.version:
                raise RolloutVersionError("mixed rollout versions")
            parts.append((indices, payload.rollout))
            durations.append(payload.generation_seconds)
            device = str(self._devices[index])
            self.peak_gpu_memory[device] = max(
                self.peak_gpu_memory.get(device, 0), payload.peak_gpu_memory
            )
        lag = self._policy_version - handle.version
        if lag < 0 or lag > self.max_policy_lag:
            raise RolloutVersionError(
                f"rollout policy lag {lag} exceeds {self.max_policy_lag}"
            )
        raw = merge_rollouts(parts, handle.total)
        self.generated_tokens += int(raw.response_mask.sum().item())
        self.generation_seconds += max(durations)
        return raw

    def close(self, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        # A worker sending an uncollected result may never read STOP or
        # join NCCL shutdown. Abandon lookahead without waiting for peers.
        force = force or self._active_round is not None
        self._active_round = None
        weights_closed = False
        try:
            if not force:
                for index, process in enumerate(self._processes):
                    if process.is_alive():
                        try:
                            self._send_command(index, MessageKind.STOP)
                        except (BrokenPipeError, EOFError, OSError, RuntimeError):
                            pass
                # NCCL shutdown involves every rank. Join only after the
                # learner has entered shutdown alongside the workers.
                self.weights.close()
                weights_closed = True
                deadline = time.monotonic() + 5.0
                for process in self._processes:
                    process.join(max(0.0, deadline - time.monotonic()))
        finally:
            for process in self._processes:
                if process.is_alive():
                    process.terminate()
            deadline = time.monotonic() + 5.0
            for process in self._processes:
                process.join(max(0.0, deadline - time.monotonic()))
                if process.is_alive():
                    process.kill()
                    process.join(1.0)
            for conn in self._pipes:
                conn.close()
            if not weights_closed:
                self.weights.close(force=True)
