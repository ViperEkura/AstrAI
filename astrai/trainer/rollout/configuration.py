"""Resolve asynchronous rollout parameters before workers are created."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from astrai.config import TrainConfig
from astrai.trainer.rollout.types import SamplingParams


@dataclass(frozen=True)
class ResolvedAsyncRolloutConfig:
    learner_device: str
    devices: Tuple[str, ...]
    params: SamplingParams
    max_seq_len: Optional[int]
    max_prompts_per_worker: int
    startup_timeout_s: float
    generation_timeout_s: float
    weight_timeout_s: float
    max_policy_lag: int
    random_seed: int
    sample_cursor: int

    @property
    def max_batch_size(self) -> int:
        return self.params.group_size * self.max_prompts_per_worker


def resolve_async_rollout(
    config: TrainConfig,
    learner_device: torch.device,
    params: SamplingParams,
    max_seq_len: Optional[int],
    sample_cursor: int,
) -> ResolvedAsyncRolloutConfig:
    devices = tuple(torch.device(name) for name in config.rollout_devices)
    if not devices or any(d.type != "cuda" or d.index is None for d in devices):
        raise ValueError("async_round requires indexed CUDA rollout devices")
    if len(set(devices)) != len(devices):
        raise ValueError("async_round requires distinct rollout_devices")
    available = torch.cuda.device_count()
    if any(d.index >= available for d in devices):
        raise ValueError(f"rollout_devices exceed available CUDA devices ({available})")
    if learner_device.type != "cuda" or learner_device.index is None:
        raise ValueError("async_round learner must use an indexed CUDA device")
    if learner_device in devices:
        raise ValueError("rollout_devices must exclude the learner device")
    if params.group_size < 2:
        raise ValueError("online_grpo group_size must be >= 2")
    return ResolvedAsyncRolloutConfig(
        learner_device=str(learner_device),
        devices=tuple(str(d) for d in devices),
        params=params,
        max_seq_len=max_seq_len,
        max_prompts_per_worker=max(
            1, (config.batch_per_device + len(devices) - 1) // len(devices)
        ),
        startup_timeout_s=config.rollout_startup_timeout_s,
        generation_timeout_s=config.rollout_worker_timeout_s,
        weight_timeout_s=config.rollout_weight_timeout_s,
        max_policy_lag=config.rollout_max_policy_lag,
        random_seed=config.random_seed,
        sample_cursor=sample_cursor,
    )
