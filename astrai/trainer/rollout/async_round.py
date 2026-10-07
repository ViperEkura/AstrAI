"""One learner and four isolated rollout processes for online GRPO."""

import ctypes
import logging
import multiprocessing as mp
import pickle
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import wait
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from astrai.parallel.executor import strip_compile_prefix
from astrai.trainer.backend import ReplicaBackend
from astrai.trainer.rollout.generator import RolloutGenerator
from astrai.trainer.rollout.runner import _score_rewards
from astrai.trainer.rollout.types import (
    RawRollout,
    RolloutResult,
    RolloutVersionError,
    SamplingParams,
)

logger = logging.getLogger(__name__)


class WeightSnapshotError(RuntimeError):
    """The learner step committed, but staging its new weights failed."""


def _register_shared(shm, device) -> bool:
    """Pin this process's mapping of the shared weight buffer when possible."""
    pointer = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
    try:
        with torch.cuda.device(device):
            status = torch.cuda.cudart().cudaHostRegister(pointer, len(shm.buf), 0)
    except (OSError, RuntimeError) as exc:
        logger.warning("CUDA could not register rollout shared memory: %s", exc)
        return False
    if status.value:
        logger.warning("CUDA could not register rollout shared memory: %s", status)
        return False
    return True


def _unregister_shared(shm, device) -> None:
    pointer = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
    try:
        with torch.cuda.device(device):
            status = torch.cuda.cudart().cudaHostUnregister(pointer)
        if status.value:
            logger.warning(
                "CUDA could not unregister rollout shared memory: %s", status
            )
    except (OSError, RuntimeError) as exc:
        logger.warning("CUDA could not unregister rollout shared memory: %s", exc)


def _shared_views(shm, layout):
    views = {}
    for name, shape, dtype_name, offset, nbytes in layout:
        dtype = getattr(torch, dtype_name)
        if nbytes:
            views[name] = torch.frombuffer(
                shm.buf, dtype=dtype, count=nbytes // dtype.itemsize, offset=offset
            ).reshape(shape)
        else:
            views[name] = torch.empty(shape, dtype=dtype)
    return views


def _slice_batch(batch: Dict, indices: List[int], total: int) -> Dict:
    result = {}
    for key, value in batch.items():
        if isinstance(value, Tensor) and value.shape[0] == total:
            result[key] = value[indices]
        elif isinstance(value, (list, tuple)) and len(value) == total:
            result[key] = [value[index] for index in indices]
        else:
            result[key] = value
    return result


def _merge_rollouts(
    parts: List[Tuple[List[int], RawRollout]], total: int
) -> RawRollout:
    """Restore input order while retaining prompt-left and response-right pads."""
    first = parts[0][1]
    versions = {raw.policy_version for _, raw in parts}
    if versions != {first.policy_version}:
        raise RolloutVersionError(f"mixed rollout versions: {sorted(versions)}")
    group_size = first.responses.shape[1]
    prompt_len = max(raw.prompts.shape[1] for _, raw in parts)
    response_len = max(raw.responses.shape[2] for _, raw in parts)
    prompts = torch.zeros(total, prompt_len, dtype=torch.long)
    prompt_mask = torch.zeros(total, prompt_len, dtype=torch.bool)
    responses = torch.zeros(total, group_size, response_len, dtype=torch.long)
    response_mask = torch.zeros(total, group_size, response_len, dtype=torch.bool)
    logprobs_old = torch.zeros(total, group_size, response_len, dtype=torch.float)
    prompt_texts = [""] * total
    response_texts = [[] for _ in range(total)]
    finish_reasons = [[] for _ in range(total)]
    for indices, raw in parts:
        local_prompt_len = raw.prompts.shape[1]
        local_response_len = raw.responses.shape[2]
        for local, global_index in enumerate(indices):
            prompts[global_index, -local_prompt_len:] = raw.prompts[local]
            prompt_mask[global_index, -local_prompt_len:] = raw.prompt_mask[local]
            responses[global_index, :, :local_response_len] = raw.responses[local]
            response_mask[global_index, :, :local_response_len] = raw.response_mask[
                local
            ]
            logprobs_old[global_index, :, :local_response_len] = raw.logprobs_old[local]
            prompt_texts[global_index] = raw.prompt_texts[local]
            response_texts[global_index] = raw.response_texts[local]
            finish_reasons[global_index] = raw.finish_reasons[local]
    return RawRollout(
        prompts=prompts,
        prompt_mask=prompt_mask,
        responses=responses,
        response_mask=response_mask,
        logprobs_old=logprobs_old,
        policy_version=first.policy_version,
        prompt_texts=prompt_texts,
        response_texts=response_texts,
        finish_reasons=finish_reasons,
    )


class SharedWeightPublisher:
    """One GPU-to-host snapshot and one shared-memory copy per policy version."""

    def __init__(self, source: nn.Module):
        self._source = strip_compile_prefix(dict(source.state_dict(keep_vars=True)))
        self.layout = []
        offset = 0
        for name, tensor in self._source.items():
            offset = (offset + 63) & ~63
            nbytes = tensor.numel() * tensor.element_size()
            self.layout.append(
                (
                    name,
                    tuple(tensor.shape),
                    str(tensor.dtype).split(".")[-1],
                    offset,
                    nbytes,
                )
            )
            offset += nbytes
        self._shm = SharedMemory(create=True, size=max(offset, 1))
        self._cuda_device = next(
            (tensor.device for tensor in self._source.values() if tensor.is_cuda),
            None,
        )
        self.registered = False
        try:
            if self._cuda_device is not None:
                self.registered = _register_shared(self._shm, self._cuda_device)
            self._views = _shared_views(self._shm, self.layout)
            self._staged = (
                {
                    name: torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
                    for name, tensor in self._source.items()
                }
                if self._cuda_device is not None and not self.registered
                else {}
            )
        except BaseException:
            if self.registered:
                _unregister_shared(self._shm, self._cuda_device)
            self._shm.close()
            self._shm.unlink()
            raise
        self.version: Optional[int] = None
        self._pending = set()
        self.snapshot_seconds = 0.0
        self.fanout_seconds = 0.0

    @property
    def name(self) -> str:
        return self._shm.name

    def snapshot(self, version: int) -> None:
        if self._pending:
            raise RuntimeError("previous weight transfer has not been acknowledged")
        started = time.perf_counter()
        with torch.no_grad():
            for name, source in self._source.items():
                destination = (
                    self._views[name]
                    if self.registered or not source.is_cuda
                    else self._staged[name]
                )
                destination.copy_(source, non_blocking=source.device.type == "cuda")
            cuda_devices = {
                tensor.device for tensor in self._source.values() if tensor.is_cuda
            }
            for device in cuda_devices:
                torch.cuda.current_stream(device).synchronize()
            if not self.registered and self._staged:
                for name, source in self._source.items():
                    if source.is_cuda:
                        self._views[name].copy_(self._staged[name])
        self.version = version
        self.snapshot_seconds += time.perf_counter() - started

    def begin_fanout(self, indices: List[int]) -> None:
        if self._pending:
            raise RuntimeError("previous weight transfer is still in flight")
        self._pending = set(indices)

    def acknowledge(self, index: int) -> None:
        if index not in self._pending:
            raise RuntimeError("unexpected weight transfer acknowledgement")
        self._pending.remove(index)

    def close(self) -> None:
        self._views.clear()
        if self.registered:
            _unregister_shared(self._shm, self._cuda_device)
        self._shm.close()
        self._shm.unlink()


def _copy_shared_weights(views, layout, target, pinned, device):
    if set(target) != {entry[0] for entry in layout}:
        raise RuntimeError("rollout replica state keys differ from learner")
    started = time.perf_counter()
    with torch.no_grad(), torch.cuda.device(device):
        for name, shape, dtype_name, _, _ in layout:
            dest = target[name]
            if tuple(dest.shape) != shape:
                raise RuntimeError(f"rollout replica shape differs for {name}")
            shared = views[name]
            if shared.dtype != getattr(torch, dtype_name):
                raise RuntimeError(f"rollout shared dtype differs for {name}")
            host = shared
            if pinned is not None:
                host = pinned[name]
                host.copy_(shared)
            dest.copy_(host, non_blocking=True)
        # Both the ACK and the next shared-memory publication require H2D
        # completion. Registered mappings and fallback pinned buffers persist.
        torch.cuda.current_stream(device).synchronize()
    return time.perf_counter() - started


def _rollout_worker_main(
    conn,
    device,
    model_fn,
    param_path,
    params,
    max_batch_size,
    max_seq_len,
    policy_version,
    model_dtype,
    shm_name,
    layout,
    max_prompts_per_worker,
):
    shm = None
    registered = False
    views = None
    try:
        torch.cuda.set_device(device)
        from astrai.tokenize import AutoTokenizer

        shm = SharedMemory(name=shm_name)
        registered = _register_shared(shm, device)
        views = _shared_views(shm, layout)
        model = model_fn().to(device=device, dtype=getattr(torch, model_dtype))
        model.requires_grad_(False)
        model.eval()
        target = dict(model.state_dict(keep_vars=True))
        pinned = (
            None
            if registered
            else {
                name: torch.empty(
                    shape, dtype=getattr(torch, dtype_name), pin_memory=True
                )
                for name, shape, dtype_name, _, _ in layout
            }
        )
        _copy_shared_weights(views, layout, target, pinned, device)
        tokenizer = AutoTokenizer.from_pretrained(param_path)
        backend = ReplicaBackend(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            policy_version=policy_version,
            enable_cuda_graph=True,
        )
        generator = RolloutGenerator(
            backend=backend, tokenizer=tokenizer, params=params, output_device="cpu"
        )
        conn.send(
            (
                "ready",
                policy_version,
                backend.scheduler.cuda_graph_enabled,
                torch.cuda.max_memory_allocated(device),
                registered,
            )
        )
        while True:
            command = conn.recv()
            kind = command[0]
            if kind == "stop":
                break
            if kind == "weight":
                version = command[1]
                if version <= backend.policy_version:
                    raise RuntimeError("weight version must advance")

                def copy_weights(_version):
                    return _copy_shared_weights(views, layout, target, pinned, device)

                duration = backend.apply_weight_update(version, copy_weights)
                conn.send(
                    (
                        "weight_ack",
                        version,
                        duration,
                        torch.cuda.max_memory_allocated(device),
                    )
                )
            elif kind == "generate":
                _, round_id, version, chunk = command
                if backend.policy_version != version:
                    raise RuntimeError(
                        "rollout worker did not receive requested version"
                    )
                started = time.perf_counter()
                total = len(next(iter(chunk.values())))
                pieces = []
                for begin in range(0, total, max_prompts_per_worker):
                    indices = list(
                        range(begin, min(begin + max_prompts_per_worker, total))
                    )
                    pieces.append(
                        (
                            indices,
                            generator.generate(_slice_batch(chunk, indices, total)),
                        )
                    )
                raw = _merge_rollouts(pieces, total)
                conn.send(
                    (
                        "result",
                        round_id,
                        version,
                        raw,
                        time.perf_counter() - started,
                        torch.cuda.max_memory_allocated(device),
                    )
                )
            else:
                raise RuntimeError(f"unknown rollout command: {kind}")
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        conn.close()
        if shm is not None:
            if views is not None:
                views.clear()
            if registered:
                _unregister_shared(shm, device)
            shm.close()


@dataclass
class RoundHandle:
    batch: Dict
    jobs: List[Tuple[int, List[int]]]
    version: int
    round_id: int
    started: float


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
        self._closed = False
        self.generated_tokens = 0
        self.generation_seconds = 0.0
        self.learner_wait_seconds = 0.0
        self.peak_gpu_memory = {}
        self.worker_cuda_graph_enabled = {}
        self.worker_shared_memory_pinned = {}
        self.publisher = SharedWeightPublisher(source)
        self._ctx = mp.get_context("spawn")
        self._processes = []
        self._pipes = []
        self._worker_versions = []
        self._active_round = None
        self._worker_target = worker_target or _rollout_worker_main
        try:
            startup_deadline = time.monotonic() + 300.0
            self.publisher.snapshot(policy_version)
            for device in devices:
                parent, child = self._ctx.Pipe(duplex=True)
                process = self._ctx.Process(
                    target=self._worker_target,
                    args=(
                        child,
                        device,
                        model_fn,
                        param_path,
                        params,
                        max_batch_size,
                        max_seq_len,
                        policy_version,
                        str(model_dtype).split(".")[-1],
                        self.publisher.name,
                        self.publisher.layout,
                        max_prompts_per_worker,
                    ),
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
            ready = self._wait_for(list(range(len(devices))), "ready", startup_deadline)
            for index, message in ready.items():
                self._worker_versions[index] = message[1]
                self.worker_cuda_graph_enabled[devices[index]] = message[2]
                self.peak_gpu_memory[devices[index]] = message[3]
                self.worker_shared_memory_pinned[devices[index]] = message[4]
        except BaseException:
            self.close(force=True)
            raise

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def _wait_for(self, indices, kind, deadline, round_id=None, version=None):
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
                        message = conn.recv()
                    except (EOFError, OSError) as exc:
                        raise RuntimeError(
                            f"async rollout worker {index} exited"
                        ) from exc
                    if message[0] == "error":
                        raise RuntimeError(
                            f"async rollout worker {index} failed:\n{message[1]}"
                        )
                    if message[0] != kind:
                        raise RuntimeError(
                            f"async rollout worker {index} sent unexpected {message[0]}"
                        )
                    if kind == "result" and message[1] != round_id:
                        raise RuntimeError("async rollout worker sent stale round ID")
                    if kind == "weight_ack" and message[1] != version:
                        raise RuntimeError(
                            "async rollout worker acknowledged wrong version"
                        )
                    messages[index] = message
                    pending.remove(index)
            for index in pending:
                if self._processes[index].sentinel in ready:
                    raise RuntimeError(f"async rollout worker {index} exited")
        return messages

    def _send(self, index, message):
        if not self._processes[index].is_alive():
            raise RuntimeError(f"async rollout worker {index} exited")
        self._pipes[index].send(message)

    def _load_version(self, indices, version):
        missing = [
            index for index in indices if self._worker_versions[index] != version
        ]
        if not missing:
            return
        if self.publisher.version != version:
            raise RuntimeError("learner weights were not staged for requested version")
        self.publisher.begin_fanout(missing)
        started = time.perf_counter()
        for index in missing:
            self._send(index, ("weight", version))
        acks = self._wait_for(
            missing,
            "weight_ack",
            time.monotonic() + self.worker_timeout_s,
            version=version,
        )
        for index, message in acks.items():
            self.publisher.acknowledge(index)
            self._worker_versions[index] = version
            device = str(self._devices[index])
            self.peak_gpu_memory[device] = max(
                self.peak_gpu_memory.get(device, 0), message[3]
            )
        self.publisher.fanout_seconds += time.perf_counter() - started

    def apply_weight_update(self, policy_version, update):
        target = self._policy_version + 1 if policy_version is None else policy_version
        if target <= self._policy_version:
            raise ValueError("policy_version must advance")
        result = update(target)
        self._policy_version = target
        try:
            self.publisher.snapshot(target)
        except BaseException as exc:
            raise WeightSnapshotError(
                f"optimizer committed policy version {target}, but weight staging failed"
            ) from exc
        return result

    def step(self):
        """Compatibility with BaseStrategy.optimizer_step."""

    def submit_round(self, batch: Dict) -> RoundHandle:
        if self._closed:
            raise RuntimeError("async rollout coordinator is closed")
        if self._active_round is not None:
            raise RuntimeError("only one rollout round may be in flight")
        total = len(next(iter(batch.values())))
        if total == 0:
            raise ValueError("async rollout round cannot be empty")
        version = self._policy_version
        jobs = []
        for index in range(len(self._processes)):
            indices = list(range(index, total, len(self._processes)))
            if indices:
                jobs.append((index, indices))
        try:
            self._load_version([index for index, _ in jobs], version)
            self._round_id += 1
            for index, indices in jobs:
                self._send(
                    index,
                    (
                        "generate",
                        self._round_id,
                        version,
                        _slice_batch(batch, indices, total),
                    ),
                )
            handle = RoundHandle(batch, jobs, version, self._round_id, time.monotonic())
            self._active_round = handle
            return handle
        except BaseException:
            self.close(force=True)
            raise

    def collect_round(self, handle: RoundHandle) -> RolloutResult:
        if handle is not self._active_round:
            raise RuntimeError("round handle is not active")
        wait_started = time.perf_counter()
        try:
            messages = self._wait_for(
                [index for index, _ in handle.jobs],
                "result",
                handle.started + self.worker_timeout_s,
                round_id=handle.round_id,
            )
        except BaseException:
            self.close(force=True)
            raise
        finally:
            self._active_round = None
        self.learner_wait_seconds += time.perf_counter() - wait_started
        parts = []
        durations = []
        for index, indices in handle.jobs:
            message = messages[index]
            if (
                message[2] != handle.version
                or message[3].policy_version != handle.version
            ):
                raise RolloutVersionError("mixed rollout versions")
            parts.append((indices, message[3]))
            durations.append(message[4])
            device = str(self._devices[index])
            self.peak_gpu_memory[device] = max(
                self.peak_gpu_memory.get(device, 0), message[5]
            )
        lag = self._policy_version - handle.version
        if lag < 0 or lag > self.max_policy_lag:
            raise RolloutVersionError(
                f"rollout policy lag {lag} exceeds {self.max_policy_lag}"
            )
        raw = _merge_rollouts(parts, len(next(iter(handle.batch.values()))))
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
                        self._pipes[index].send(("stop",))
                    except (BrokenPipeError, EOFError, OSError):
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
        self.publisher.close()
