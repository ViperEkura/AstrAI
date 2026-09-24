"""Rollout backends: where online-RL generation physically runs.

A :class:`~astrai.trainer.rollout.RolloutGenerator` is location-agnostic —
it renders prompts, expands groups, pads and decodes.  Where the
prefill/decode loop executes, on which model object and device, is a
backend concern:

- :class:`ColocatedBackend` wraps an :class:`InferenceScheduler` that
  shares the *training* model object in-process (the historical rollout
  path; weight updates are free because there is only one model).
- :class:`ReplicaBackend` owns a frozen copy of the policy on its own
  device with its own scheduler and KV pool.  Fresh weights arrive via a
  :class:`P2PCopyPublisher` invoked inside the policy-version lock, so a
  replica generation can never observe new-version weights that are
  actually stale.

The :class:`RolloutBackend` protocol is the seam: anything that can
generate given tokenized prompts and expose the policy-version protocol
can serve as a rollout backend.
"""

from contextlib import nullcontext
from typing import Callable, List, Optional, Protocol, TypeVar

import torch
from torch import nn

from astrai.inference.scheduler import InferenceScheduler
from astrai.parallel.executor import strip_compile_prefix

T = TypeVar("T")


def _device_context(device):
    """Pin CUDA work to ``device``; a no-op on CPU.

    CUDA graph capture/replay binds to the calling thread's current
    device (``CudaGraphContext`` uses a bare ``torch.cuda.CUDAGraph()``),
    so a scheduler living on a non-current device must be constructed
    and driven under ``torch.cuda.device``.
    """
    if isinstance(device, torch.device) and device.type == "cuda":
        return torch.cuda.device(device)
    if isinstance(device, str) and device.startswith("cuda"):
        return torch.cuda.device(torch.device(device))
    return nullcontext()


class RolloutBackend(Protocol):
    """Where rollout generation runs, behind the policy-version protocol."""

    @property
    def device(self):
        """Device the backend generates on (rollout tensors start here)."""

    @property
    def policy_version(self) -> int:
        """Version of the weights used for subsequent generations."""

    @property
    def runtime_released(self) -> bool:
        """Whether the inference runtime is released."""

    def release(self) -> bool:
        """Release inference runtime resources."""

    def resume(self) -> bool:
        """Restore inference runtime resources."""

    def generate(self, prompt_ids_list: List[List[int]], **kwargs):
        """Run prefill+decode to completion; see ``run_batch`` for kwargs."""

    def update_weights(self, policy_version: int) -> int:
        """Acknowledge externally applied weights and invalidate KV."""

    def apply_weight_update(
        self, policy_version: Optional[int], update: Callable[[int], T]
    ) -> T:
        """Mutate weights and publish the version atomically."""

    def with_policy_snapshot(self, inspect: Callable[[int], T]) -> T:
        """Run ``inspect`` while the policy version is stable."""


class ColocatedBackend:
    """Scheduler wrapping the training model object itself (the default).

    Weight updates mutate the shared object in place, so publishing a
    version is bookkeeping only.  Generation toggles the shared model to
    eval mode and restores its prior mode afterwards.
    """

    def __init__(self, scheduler: InferenceScheduler):
        self.scheduler = scheduler

    @property
    def device(self):
        return self.scheduler.device

    @property
    def policy_version(self) -> int:
        return self.scheduler.policy_version

    @property
    def runtime_released(self) -> bool:
        return self.scheduler.runtime_released

    def release(self) -> bool:
        return self.scheduler.release()

    def resume(self) -> bool:
        return self.scheduler.resume()

    def generate(self, prompt_ids_list: List[List[int]], **kwargs):
        model = self.scheduler._executor.model
        was_training = model.training
        model.eval()
        try:
            return self.scheduler.run_batch(prompt_ids_list, **kwargs)
        finally:
            model.train(was_training)

    def update_weights(self, policy_version: int) -> int:
        return self.scheduler.update_weights(policy_version)

    def apply_weight_update(
        self, policy_version: Optional[int], update: Callable[[int], T]
    ) -> T:
        return self.scheduler.apply_weight_update(policy_version, update)

    def with_policy_snapshot(self, inspect: Callable[[int], T]) -> T:
        return self.scheduler.with_policy_snapshot(inspect)


class ReplicaBackend:
    """Frozen policy copy on its own device, with its own scheduler/KV pool.

    The model is expected to be already placed on ``device`` and frozen
    (see :func:`astrai.trainer.train_context.create_ref_model`); it never
    toggles out of eval mode.  All scheduler work — construction (CUDA
    graph capture) and generation — runs pinned to ``device``.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device,
        max_batch_size: int = 16,
        max_seq_len: Optional[int] = None,
        policy_version: int = 0,
        enable_cuda_graph: bool = True,
    ):
        model.eval()
        with _device_context(device):
            self.scheduler = InferenceScheduler(
                model=model,
                tokenizer=tokenizer,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                device=str(device) if isinstance(device, torch.device) else device,
                enable_cuda_graph=enable_cuda_graph,
                policy_version=policy_version,
            )
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device

    @property
    def policy_version(self) -> int:
        return self.scheduler.policy_version

    @property
    def runtime_released(self) -> bool:
        return self.scheduler.runtime_released

    def release(self) -> bool:
        return self.scheduler.release()

    def resume(self) -> bool:
        return self.scheduler.resume()

    def generate(self, prompt_ids_list: List[List[int]], **kwargs):
        with _device_context(self.device):
            return self.scheduler.run_batch(prompt_ids_list, **kwargs)

    def update_weights(self, policy_version: int) -> int:
        return self.scheduler.update_weights(policy_version)

    def apply_weight_update(
        self, policy_version: Optional[int], update: Callable[[int], T]
    ) -> T:
        return self.scheduler.apply_weight_update(policy_version, update)

    def with_policy_snapshot(self, inspect: Callable[[int], T]) -> T:
        return self.scheduler.with_policy_snapshot(inspect)


class WeightPublisher(Protocol):
    """Fan out the training model's new weights to a rollout backend.

    Implementations run inside the policy-version lock as part of the
    atomic commit (see
    :meth:`BaseStrategy.optimizer_step
    <astrai.trainer.strategy.BaseStrategy.optimizer_step>`): copying the
    weights and advancing the receiving backend's version must be
    indivisible from the trainer's own version publication, otherwise a
    generation could observe new-version weights that are actually stale.
    """

    def publish(self, policy_version: int, source: nn.Module) -> None:
        """Copy ``source`` weights and acknowledge ``policy_version``.

        Args:
            policy_version: The version the receiving backend must expose
                after this call; monotonically increasing.
            source: The (unwrapped) training model to copy from.
        """
        ...


class P2PCopyPublisher:
    """Copies training weights into a :class:`ReplicaBackend` each commit.

    Parameter pairs (source tensor, replica tensor) are resolved once from
    ``state_dict(keep_vars=True)`` on the first publish — no per-step
    state-dict materialization, and cross-device ``copy_`` rides the P2P
    or host path automatically.  A full copy runs per optimizer step, so
    on a 1B bf16 policy this is ~2GB of traffic per step by design: pay
    it only when the isolation (separate KV pool, non-blocking val) is
    worth it.
    """

    def __init__(self, backend: ReplicaBackend):
        self._backend = backend
        self._pairs = None

    def publish(self, policy_version: int, source: nn.Module) -> None:
        if self._pairs is None:
            src = strip_compile_prefix(dict(source.state_dict(keep_vars=True)))
            dst = dict(self._backend.model.state_dict(keep_vars=True))
            missing = set(dst) - set(src)
            if missing:
                raise RuntimeError(
                    f"replica has parameters absent from the training "
                    f"model: {sorted(missing)[:5]}"
                )
            self._pairs = [(src[name], dst[name]) for name in dst]
        with torch.no_grad():
            for src_tensor, dst_tensor in self._pairs:
                dst_tensor.copy_(src_tensor)
        self._backend.update_weights(policy_version)
