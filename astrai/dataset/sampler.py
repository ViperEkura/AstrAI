from typing import Iterator, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class GlobalBatchSampler(Sampler[List[int]]):
    """Shard committed global optimizer windows without repeating data.

    Every replica emits the same number of microbatches. An empty slot reads
    sample zero solely to run the distributed forward/backward; the trainer
    assigns that slot zero weight and never advances its data cursor for it.
    """

    def __init__(
        self,
        data_source: Dataset,
        dp_size: int,
        dp_rank: int,
        batch_size: int,
        accumulation_steps: int,
        seed: int,
        tail_policy: str,
        epoch: int = 0,
        start_offset: int = 0,
    ):
        self.num_samples = len(data_source)
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.seed = seed
        self.tail_policy = tail_policy
        self.epoch = epoch
        self.start_offset = start_offset
        self.global_batch = dp_size * batch_size * accumulation_steps
        if self.num_samples < 1:
            raise ValueError("global training sampler requires a nonempty dataset")
        if (
            start_offset < 0
            or start_offset > self.num_samples
            or (start_offset != self.num_samples and start_offset % self.global_batch)
        ):
            raise ValueError(
                "global sampler offset must be a committed window boundary"
            )

    def set_epoch(self, epoch: int, start_offset: int = 0) -> None:
        self.epoch = epoch
        self.start_offset = start_offset

    def windows(self) -> Iterator[tuple[int, int, int]]:
        capacity = self.dp_size * self.batch_size
        for start in range(self.start_offset, self.num_samples, self.global_batch):
            count = min(self.global_batch, self.num_samples - start)
            if count < self.global_batch and self.tail_policy == "drop":
                break
            yield start, count, (count + capacity - 1) // capacity

    def real_count(self, window_start: int, microbatch: int) -> int:
        first = window_start + microbatch * self.dp_size * self.batch_size
        first += self.dp_rank * self.batch_size
        return max(0, min(self.batch_size, self.num_samples - first))

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.num_samples, generator=generator).tolist()
        for start, _count, microbatches in self.windows():
            for microbatch in range(microbatches):
                first = start + microbatch * self.dp_size * self.batch_size
                first += self.dp_rank * self.batch_size
                local = indices[first : min(first + self.batch_size, self.num_samples)]
                yield local or [indices[0]]

    def __len__(self) -> int:
        full, tail = divmod(self.num_samples - self.start_offset, self.global_batch)
        partial = 0
        if tail and self.tail_policy == "flush":
            capacity = self.dp_size * self.batch_size
            partial = (tail + capacity - 1) // capacity
        return full * self.accumulation_steps + partial


class RDSampler(Sampler[int]):
    """Padded rank shards for validation and asynchronous rollout rounds.

    Worker prefetch can advance this iterator ahead of the learner. Checkpoints
    use the committed TrainContext cursor, never this mutable read position.
    """

    def __init__(
        self,
        data_source: Dataset,
        start_epoch: int = 0,
        start_iter: int = 0,
        seed: int = 42,
        drop_last: bool = False,
        shuffle: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        self.epoch = start_epoch
        self.iter = start_iter
        self.seed = seed
        self.num_samples = len(data_source)

        if process_group is not None:
            # input process group
            self.rank = dist.get_rank(process_group)
            self.num_replicas = dist.get_world_size(process_group)

        elif dist.is_available() and dist.is_initialized():
            # use default process group
            process_group = dist.group.WORLD
            self.rank = dist.get_rank()
            self.num_replicas = dist.get_world_size()

        else:
            # single process
            self.rank = 0
            self.num_replicas = 1

        self.drop_last = drop_last
        self.shuffle = shuffle

        offset = 0 if drop_last else self.num_replicas - 1
        self.num_samples_per_replica = (self.num_samples + offset) // self.num_replicas
        self.total_size = self.num_samples_per_replica * self.num_replicas
        self.iter = self.iter % self.num_samples_per_replica

        self._indices = None

    def _get_indices(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=generator).tolist()
        else:
            indices = torch.arange(self.num_samples).tolist()

        if not self.drop_last and self.num_samples < self.total_size:
            padding_size = self.total_size - len(indices)
            indices += indices[:padding_size]

        local_indices = indices[self.rank : self.total_size : self.num_replicas]

        self.iter = self.iter % self.num_samples_per_replica
        self._indices = local_indices[self.iter :]

    def __iter__(self):
        if self._indices is None:
            self._get_indices()

        for i in self._indices:
            self.iter += 1
            yield i

        self.epoch += 1
        self._indices = None
        self.iter = self.iter % self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        """Start the next epoch without loading batches the trainer discarded."""
        self.epoch = epoch
        self.iter = 0
        self._indices = None

    @property
    def _remaining(self):
        remaining = self.num_samples_per_replica - self.iter
        return max(remaining, 0)

    def __len__(self):
        return self._remaining
