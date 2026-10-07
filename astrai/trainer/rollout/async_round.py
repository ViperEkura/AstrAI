"""One learner and multiple in-process rollout replicas for online GRPO."""

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from astrai.parallel.executor import strip_compile_prefix
from astrai.trainer.backend import ReplicaBackend, _device_context
from astrai.trainer.rollout.generator import RolloutGenerator
from astrai.trainer.rollout.runner import _score_rewards
from astrai.trainer.rollout.types import RawRollout, RolloutResult, RolloutVersionError


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


class PinnedWeightFanout:
    """Stage one coherent learner snapshot and publish it to idle replicas.

    This machine has no CUDA peer access between its 5090s.  A single pinned
    CPU copy is shared by the four H2D transfers, and is not overwritten until
    every receiving stream has completed its copy.
    """

    def __init__(self, source: nn.Module, backends: List[ReplicaBackend]):
        self.backends = backends
        source_state = strip_compile_prefix(dict(source.state_dict(keep_vars=True)))
        self._source = source_state
        self._staged = {}
        for name, tensor in source_state.items():
            self._staged[name] = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=tensor.device.type == "cuda",
            )
        self._targets = []
        for backend in backends:
            target = dict(backend.model.state_dict(keep_vars=True))
            if set(target) != set(source_state):
                raise RuntimeError("rollout replica state keys differ from learner")
            for name, tensor in target.items():
                if tensor.shape != source_state[name].shape:
                    raise RuntimeError(f"rollout replica shape differs for {name}")
            self._targets.append(target)
        self.version: Optional[int] = None
        self._readers = 0
        self._ready = threading.Condition()
        self.snapshot_seconds = 0.0
        self.fanout_seconds = 0.0

    def snapshot(self, version: int) -> None:
        with self._ready:
            self._ready.wait_for(lambda: self._readers == 0)
        started = time.perf_counter()
        with torch.no_grad():
            for name, source in self._source.items():
                self._staged[name].copy_(
                    source, non_blocking=source.device.type == "cuda"
                )
            if any(tensor.device.type == "cuda" for tensor in self._source.values()):
                torch.cuda.current_stream(
                    next(
                        tensor.device
                        for tensor in self._source.values()
                        if tensor.device.type == "cuda"
                    )
                ).synchronize()
        self.version = version
        self.snapshot_seconds += time.perf_counter() - started

    def begin_fanout(self, readers: int) -> None:
        with self._ready:
            if self._readers:
                raise RuntimeError("previous weight fanout is still in flight")
            self._readers = readers

    def publish_one(self, index: int, version: int) -> None:
        started = time.perf_counter()
        try:
            if self.version != version:
                raise RuntimeError("staged policy version changed during fanout")
            backend = self.backends[index]
            if backend.policy_version == version:
                return
            target = self._targets[index]

            def copy_weights(_version):
                with torch.no_grad(), _device_context(backend.device):
                    for name, dest in target.items():
                        dest.copy_(
                            self._staged[name],
                            non_blocking=dest.device.type == "cuda",
                        )
                    # The policy version may become visible only after all
                    # transfers have completed, including on another stream.
                    if torch.device(backend.device).type == "cuda":
                        torch.cuda.current_stream(backend.device).synchronize()

            backend.apply_weight_update(version, copy_weights)
        finally:
            with self._ready:
                self._readers -= 1
                self.fanout_seconds += time.perf_counter() - started
                self._ready.notify_all()


@dataclass
class RoundHandle:
    batch: Dict
    futures: List[Tuple[List[int], Future]]
    version: int
    started: float


class AsyncRoundCoordinator:
    """Bounded one-round lookahead: generate on replicas, learn on one GPU."""

    def __init__(
        self,
        source: nn.Module,
        generators: List[RolloutGenerator],
        reward_model,
        policy_version: int,
        max_policy_lag: int = 1,
        max_prompts_per_worker: int = 1,
    ):
        self.generators = generators
        self.reward_model = reward_model
        self._policy_version = policy_version
        self.max_policy_lag = max_policy_lag
        self.max_prompts_per_worker = max_prompts_per_worker
        self.publisher = PinnedWeightFanout(
            source, [generator.backend for generator in generators]
        )
        self._workers = ThreadPoolExecutor(
            max_workers=len(generators), thread_name_prefix="rollout"
        )
        self._closed = False
        self.generated_tokens = 0
        self.generation_seconds = 0.0
        self.learner_wait_seconds = 0.0

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def apply_weight_update(self, policy_version, update):
        target = self._policy_version + 1 if policy_version is None else policy_version
        if target <= self._policy_version:
            raise ValueError("policy_version must advance")
        result = update(target)
        # The optimizer has mutated the learner even if staging fails; keep
        # checkpoint metadata truthful and let the caller abort on failure.
        self._policy_version = target
        self.publisher.snapshot(target)
        return result

    def step(self):
        """Compatibility with BaseStrategy.optimizer_step."""

    def submit_round(self, batch: Dict) -> RoundHandle:
        if self._closed:
            raise RuntimeError("async rollout coordinator is closed")
        total = len(next(iter(batch.values())))
        if total == 0:
            raise ValueError("async rollout round cannot be empty")
        version = self._policy_version
        jobs = []
        for index in range(len(self.generators)):
            indices = list(range(index, total, len(self.generators)))
            if indices:
                jobs.append((index, indices, _slice_batch(batch, indices, total)))
        needs_copy = self.publisher.version == version
        if needs_copy:
            self.publisher.begin_fanout(len(jobs))
        futures = []
        try:
            for index, indices, chunk in jobs:
                future = self._workers.submit(
                    self._generate_one, index, chunk, version, needs_copy
                )
                futures.append((indices, future))
        except BaseException:
            if needs_copy:
                with self.publisher._ready:
                    self.publisher._readers -= len(jobs) - len(futures)
                    self.publisher._ready.notify_all()
            raise
        return RoundHandle(batch, futures, version, time.perf_counter())

    def _generate_one(
        self, index: int, chunk: Dict, version: int, needs_copy: bool
    ) -> Tuple[RawRollout, float]:
        if needs_copy:
            self.publisher.publish_one(index, version)
        elif self.generators[index].policy_version != version:
            raise RuntimeError("rollout replica did not receive the requested version")
        started = time.perf_counter()
        total = len(next(iter(chunk.values())))
        pieces = []
        for begin in range(0, total, self.max_prompts_per_worker):
            indices = list(
                range(begin, min(begin + self.max_prompts_per_worker, total))
            )
            part = self.generators[index].generate(_slice_batch(chunk, indices, total))
            pieces.append((indices, part))
        raw = _merge_rollouts(pieces, total)
        return raw, time.perf_counter() - started

    def collect_round(self, handle: RoundHandle) -> RolloutResult:
        wait_started = time.perf_counter()
        parts = []
        durations = []
        by_future = {future: indices for indices, future in handle.futures}
        for future in as_completed(by_future):
            try:
                raw, duration = future.result()
                parts.append((by_future[future], raw))
                durations.append(duration)
            except BaseException as exc:
                for other in by_future:
                    other.cancel()
                raise RuntimeError("async rollout worker failed") from exc
        self.learner_wait_seconds += time.perf_counter() - wait_started
        versions = {raw.policy_version for _, raw in parts}
        if versions != {handle.version}:
            raise RolloutVersionError(f"mixed rollout versions: {sorted(versions)}")
        lag = self._policy_version - handle.version
        if lag < 0 or lag > self.max_policy_lag:
            # No part of this round may enter a learner update.
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

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._workers.shutdown(wait=True, cancel_futures=True)
