from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class RDSampler(Sampler[int]):
    """Resumable Distributed Sampler.

    A distributed sampler that supports checkpoint-based resume: iteration
    state (epoch, position) is tracked so training can continue from the
    exact sample after a restart.  Shards the dataset across
    ``dist.world_size`` replicas with optional shuffling.
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

    @property
    def _remaining(self):
        remaining = self.num_samples_per_replica - self.iter
        return max(remaining, 0)

    def __len__(self):
        return self._remaining
