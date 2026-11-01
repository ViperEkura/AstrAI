"""Optimizer-step counts shared by scheduling and progress reporting."""

from typing import Iterator, Optional, Tuple


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def learner_update_slices(
    total: int, update_epochs: int = 1, minibatch_prompts: Optional[int] = None
) -> Iterator[Tuple[slice, bool]]:
    """Yield prompt ranges in replay order, marking the final window update."""
    size = minibatch_prompts or total
    if total <= 0 or update_epochs <= 0 or size <= 0:
        raise ValueError("learner updates require positive sizes and replay epochs")
    for epoch in range(update_epochs):
        for begin in range(0, total, size):
            end = min(begin + size, total)
            yield slice(begin, end), epoch + 1 == update_epochs and end == total


def global_epoch_steps(
    dataset_len: int,
    start_offset: int,
    global_batch: int,
    tail_policy: str,
    *,
    online: bool = False,
    rl_update_epochs: int = 1,
    rl_minibatch_prompts: Optional[int] = None,
) -> int:
    """Count commits from unpadded global windows, including a flush tail."""
    full, tail = divmod(dataset_len - start_offset, global_batch)
    tail = tail if tail_policy == "flush" else 0
    if not online:
        return full + int(tail > 0)
    group_size = rl_minibatch_prompts or global_batch
    return rl_update_epochs * (
        full * ceil_div(global_batch, group_size)
        + (ceil_div(tail, group_size) if tail else 0)
    )
