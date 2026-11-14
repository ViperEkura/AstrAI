"""Rollout backends: where online-RL generation physically runs.

A :class:`~astrai.trainer.rollout.RolloutGenerator` is location-agnostic —
it renders prompts, expands groups, pads and decodes.  Where the
prefill/decode loop executes, on which model object and device, is a
backend concern:

- :class:`ColocatedBackend` wraps an :class:`EngineCore` that
  shares the *training* model object in-process (the historical rollout
  path; weight updates are free because there is only one model).
- :class:`ReplicaBackend` owns a frozen copy of the policy on its own
  device with its own engine core and KV pool.  Fresh weights arrive via a
  :class:`P2PCopyPublisher` invoked inside the policy-version lock, so a
  replica generation can never observe new-version weights that are
  actually stale.

The :class:`RolloutBackend` protocol is the seam: anything that can
generate given tokenized prompts and expose the policy-version protocol
can serve as a rollout backend.
"""

import uuid
from contextlib import nullcontext
from typing import Any, Callable, List, Optional, Protocol, TypeVar

import torch
from torch import nn

from astrai.inference.core.engine_core import EngineCore
from astrai.inference.core.events import FINISH_STOP_TOKEN
from astrai.inference.core.request import GenerationResult, Request
from astrai.inference.sampling_rng import validate_seed
from astrai.parallel.executor import strip_compile_prefix

T = TypeVar("T")


def _device_context(device):
    """Pin CUDA work to ``device``; a no-op on CPU.

    CUDA graph replay and scheduler work use the calling thread's current
    device, so a scheduler living on another GPU must be constructed and
    driven under ``torch.cuda.device``.
    """
    if isinstance(device, torch.device) and device.type == "cuda":
        return torch.cuda.device(device)
    if isinstance(device, str) and device.startswith("cuda"):
        return torch.cuda.device(torch.device(device))
    return nullcontext()


def run_batch(
    engine_core: EngineCore,
    prompt_ids_list: List[List[int]],
    *,
    max_tokens: Optional[int] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    frequency_penalty: float = 0.0,
    rep_window: int = 64,
    return_logprobs: bool = False,
    return_details: bool = False,
    seeds: Optional[List[int]] = None,
    request_seeds: Optional[List[int]] = None,
) -> List[Any]:
    """Generate a rollout batch and restore its original request order."""
    if seeds is not None and (
        len(seeds) != len(prompt_ids_list)
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
            for seed in seeds
        )
    ):
        raise ValueError("seeds must contain one integer in [0, 2**63) per request")
    if request_seeds is not None:
        if seeds is not None:
            raise ValueError("seeds and request_seeds are mutually exclusive")
        if len(request_seeds) != len(prompt_ids_list):
            raise ValueError("one sampling seed is required per request")
        for seed in request_seeds:
            validate_seed(seed)
    requests, errors = [], []
    batch_id = uuid.uuid4().hex
    for index, ids in enumerate(prompt_ids_list):
        error = None
        if not ids:
            error = "prompt_empty"
        elif len(ids) >= engine_core.max_seq_len:
            error = "prompt_too_long"
        limit = (
            engine_core.max_seq_len - len(ids)
            if max_tokens is None
            else min(max_tokens, engine_core.max_seq_len - len(ids))
        )
        if error is None and limit <= 0:
            error = "max_tokens_non_positive"
        request = None
        if error is None:
            request = Request(
                f"batch_{batch_id}_{index:08d}",
                ids,
                limit,
                temperature,
                top_p,
                top_k,
                frequency_penalty,
                rep_window,
                seed=None if seeds is None else seeds[index],
                sampling_seed=None if request_seeds is None else request_seeds[index],
            )
        requests.append(request)
        errors.append(error)
    engine_core.execute_requests(
        [request for request in requests if request is not None],
        return_logprobs=return_logprobs,
    )
    details = []
    for request, error in zip(requests, errors):
        if request is None:
            details.append(GenerationResult([], [], "rejected", error))
        else:
            reason = (
                "rejected"
                if request.error_reason
                else "stop"
                if request.finish_reason == FINISH_STOP_TOKEN
                else "length"
            )
            details.append(
                GenerationResult(
                    list(request.output_ids),
                    list(request.output_logprobs),
                    reason,
                    request.error_reason,
                )
            )
    if return_details:
        return details
    if return_logprobs:
        return [(result.token_ids, result.logprobs) for result in details]
    return [result.token_ids for result in details]


class RolloutBackend(Protocol):
    """Where rollout generation runs, behind the policy-version protocol."""

    @property
    def device(self):
        """Device the backend generates on (rollout tensors start here)."""

    @property
    def policy_version(self) -> int:
        """Version of the weights used for subsequent generations."""

    def generate(self, prompt_ids_list: List[List[int]], **kwargs):
        """Run prefill+decode to completion with rollout sampling arguments."""

    def update_weights(self, policy_version: int) -> int:
        """Acknowledge externally applied weights and invalidate KV."""

    def apply_weight_update(
        self, policy_version: Optional[int], update: Callable[[int], T]
    ) -> T:
        """Mutate weights and publish the version atomically."""

    def with_policy_snapshot(self, inspect: Callable[[int], T]) -> T:
        """Run ``inspect`` while the policy version is stable."""


class ColocatedBackend:
    """EngineCore wrapping the training model object itself (the default).

    Weight updates mutate the shared object in place, so publishing a
    version is bookkeeping only.  Generation toggles the shared model to
    eval mode and restores its prior mode afterwards.
    """

    def __init__(self, engine_core: EngineCore):
        self.engine_core = engine_core

    @property
    def device(self):
        return self.engine_core.device

    @property
    def policy_version(self) -> int:
        return self.engine_core.policy_version

    def generate(self, prompt_ids_list: List[List[int]], **kwargs):
        def generate_with_snapshot(_version):
            model = self.engine_core.model
            was_training = model.training
            model.eval()
            try:
                return run_batch(self.engine_core, prompt_ids_list, **kwargs)
            finally:
                model.train(was_training)

        return self.engine_core.with_policy_snapshot(generate_with_snapshot)

    def update_weights(self, policy_version: int) -> int:
        return self.engine_core.update_weights(policy_version)

    def apply_weight_update(
        self, policy_version: Optional[int], update: Callable[[int], T]
    ) -> T:
        return self.engine_core.apply_weight_update(policy_version, update)

    def with_policy_snapshot(self, inspect: Callable[[int], T]) -> T:
        return self.engine_core.with_policy_snapshot(inspect)


class ReplicaBackend:
    """Frozen policy copy on its own device, with its own core/KV pool.

    The model is expected to be already placed on ``device`` and frozen
    (see :func:`astrai.trainer.components.create_ref_model`); it never
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
        enable_overlap: bool = False,
    ):
        model.eval()
        with _device_context(device):
            self.engine_core = EngineCore(
                model=model,
                tokenizer=tokenizer,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                device=str(device) if isinstance(device, torch.device) else device,
                enable_cuda_graph=enable_cuda_graph,
                enable_overlap=enable_overlap,
                policy_version=policy_version,
            )
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device

    @property
    def policy_version(self) -> int:
        return self.engine_core.policy_version

    def generate(self, prompt_ids_list: List[List[int]], **kwargs):
        with _device_context(self.device):
            return run_batch(self.engine_core, prompt_ids_list, **kwargs)

    def update_weights(self, policy_version: int) -> int:
        return self.engine_core.update_weights(policy_version)

    def apply_weight_update(
        self, policy_version: Optional[int], update: Callable[[int], T]
    ) -> T:
        return self.engine_core.apply_weight_update(policy_version, update)

    def with_policy_snapshot(self, inspect: Callable[[int], T]) -> T:
        return self.engine_core.with_policy_snapshot(inspect)


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
        def copy_weights(_version):
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

        self._backend.apply_weight_update(policy_version, copy_weights)
