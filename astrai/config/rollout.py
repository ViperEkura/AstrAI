"""Pure configuration rules shared by rollout validation and setup."""

from typing import List, Tuple


def resolve_rollout_devices(devices: List[str], learner_count: int) -> Tuple[str, ...]:
    """Normalize an async worker layout without consulting CUDA availability."""
    if learner_count < 1:
        raise ValueError("async_round requires at least one learner")
    if not devices:
        raise ValueError("async_round requires at least one rollout device")
    indices = []
    for device in devices:
        if (
            not device.startswith("cuda:")
            or not device[5:].isascii()
            or not device[5:].isdigit()
        ):
            raise ValueError("async_round rollout_devices must be indexed CUDA devices")
        indices.append(int(device[5:]))
    if len(set(indices)) != len(indices):
        raise ValueError("async_round requires distinct rollout_devices")
    if len(indices) < learner_count:
        raise ValueError("async_round requires at least one rollout device per learner")
    if set(indices).intersection(range(learner_count)):
        raise ValueError("rollout_devices must exclude all learner devices")
    return tuple(f"cuda:{index}" for index in indices)
