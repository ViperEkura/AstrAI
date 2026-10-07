"""One learner coordinating versioned online GRPO rollout rounds."""

import multiprocessing as mp
import pickle
import time
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import Dict, List, Optional

import torch
from torch import nn

from astrai.trainer.rollout.batching import merge_rollouts, slice_batch
from astrai.trainer.rollout.protocol import (
    MessageKind,
    RolloutMessage,
    RolloutProtocolError,
    recv_message,
    send_message,
)
from astrai.trainer.rollout.runner import _score_rewards
from astrai.trainer.rollout.types import (
    RolloutResult,
    RolloutVersionError,
    SamplingParams,
)
from astrai.trainer.rollout.weight_transport import SharedWeightBuffer
from astrai.trainer.rollout.worker import RolloutWorkerSpec, run_rollout_worker


class WeightSnapshotError(RuntimeError):
    """The learner step committed, but staging its new weights failed."""


@dataclass
class PendingRound:
    total: int
    jobs: Dict[int, List[int]]
    version: int
    round_id: int
    started: float
    request_ids: Dict[int, int]


class AsyncRoundCoordinator:
    """Bounded one-round lookahead with four killable rollout processes."""

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
        self._devices = list(devices)
        self._round_id = 0
        self._request_id = 0
        self._closed = False
        self.generated_tokens = 0
        self.generation_seconds = 0.0
        self.learner_wait_seconds = 0.0
        self.peak_gpu_memory = {}
        self.worker_cuda_graph_enabled = {}
        self.worker_shared_memory_pinned = {}
        self.weights = SharedWeightBuffer(source)
        self._ctx = mp.get_context("spawn")
        self._processes = []
        self._pipes = []
        self._worker_versions = []
        self._active_round = None
        self._worker_target = worker_target or run_rollout_worker
        try:
            startup_deadline = time.monotonic() + 300.0
            self.weights.snapshot(policy_version)
            for device in devices:
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
                    shm_name=self.weights.name,
                    layout=self.weights.layout,
                    max_prompts_per_worker=max_prompts_per_worker,
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
                self.worker_shared_memory_pinned[devices[index]] = (
                    payload.shared_memory_pinned
                )
        except BaseException:
            self.close(force=True)
            raise

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def _collect_replies(
        self, indices, kind, deadline, request_ids, round_id=None, version=None
    ):
        pending = set(indices)
        messages = {}
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"async rollout {kind} timed out on workers {sorted(pending)}"
                )
            handles = [self._pipes[index] for index in pending]
            handles.extend(self._processes[index].sentinel for index in pending)
            ready = wait(handles, timeout=remaining)
            if not ready:
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

    def _sync_weights(self, indices, version):
        missing = [
            index for index in indices if self._worker_versions[index] != version
        ]
        if not missing:
            return
        if self.weights.version != version:
            raise RuntimeError("learner weights were not staged for requested version")
        self.weights.begin_fanout(missing)
        started = time.perf_counter()
        request_ids = {
            index: self._send_command(index, MessageKind.WEIGHT, version=version)
            for index in missing
        }
        acks = self._collect_replies(
            missing,
            MessageKind.WEIGHT_ACK,
            time.monotonic() + self.worker_timeout_s,
            request_ids,
            version=version,
        )
        for index, message in acks.items():
            self.weights.acknowledge(index)
            self._worker_versions[index] = version
            device = str(self._devices[index])
            self.peak_gpu_memory[device] = max(
                self.peak_gpu_memory.get(device, 0), message.payload.peak_gpu_memory
            )
        self.weights.fanout_seconds += time.perf_counter() - started

    def apply_weight_update(self, policy_version, update):
        target = self._policy_version + 1 if policy_version is None else policy_version
        if target <= self._policy_version:
            raise ValueError("policy_version must advance")
        result = update(target)
        self._policy_version = target
        try:
            self.weights.snapshot(target)
        except BaseException as exc:
            raise WeightSnapshotError(
                f"optimizer committed policy version {target}, but weight staging failed"
            ) from exc
        return result

    def step(self):
        """Compatibility with BaseStrategy.optimizer_step."""

    def submit_round(self, batch: Dict) -> PendingRound:
        if self._closed:
            raise RuntimeError("async rollout coordinator is closed")
        if self._active_round is not None:
            raise RuntimeError("only one rollout round may be in flight")
        total = len(next(iter(batch.values())))
        if total == 0:
            raise ValueError("async rollout round cannot be empty")
        version = self._policy_version
        jobs = {
            index: list(range(index, total, len(self._processes)))
            for index in range(min(total, len(self._processes)))
        }
        try:
            self._sync_weights(list(jobs), version)
            self._round_id += 1
            request_ids = {}
            for index, indices in jobs.items():
                request_ids[index] = self._send_command(
                    index,
                    MessageKind.GENERATE,
                    round_id=self._round_id,
                    version=version,
                    payload=slice_batch(batch, indices, total),
                )
            handle = PendingRound(
                total, jobs, version, self._round_id, time.monotonic(), request_ids
            )
            self._active_round = handle
            return handle
        except BaseException:
            self.close(force=True)
            raise

    def collect_round(self, handle: PendingRound) -> RolloutResult:
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
        rewards = _score_rewards(self.reward_model, raw).to(device="cpu")
        self.generated_tokens += int(raw.response_mask.sum().item())
        self.generation_seconds += max(durations)
        return RolloutResult(
            prompts=raw.prompts,
            prompt_mask=raw.prompt_mask,
            responses=raw.responses,
            response_mask=raw.response_mask,
            rewards=rewards,
            logprobs_old=raw.logprobs_old,
            policy_version=raw.policy_version,
            prompt_texts=raw.prompt_texts,
            response_texts=raw.response_texts,
            finish_reasons=raw.finish_reasons,
        )

    def close(self, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if not force:
            for index, process in enumerate(self._processes):
                if process.is_alive():
                    try:
                        self._send_command(index, MessageKind.STOP)
                    except (BrokenPipeError, EOFError, OSError, RuntimeError):
                        pass
            deadline = time.monotonic() + 5.0
            for process in self._processes:
                process.join(max(0.0, deadline - time.monotonic()))
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
        self.weights.close()
