"""The mutable state one training run carries through its lifecycle."""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset

from astrai.config.train_config import TrainConfig
from astrai.parallel.executor import BaseExecutor
from astrai.parallel.topology import ParallelTopology
from astrai.serialization import Checkpoint
from astrai.trainer.callbacks.metric_util import GradSNRTracker
from astrai.trainer.rollout import RolloutEvaluator, RolloutRunner
from astrai.trainer.strategy import BaseStrategy

logger = logging.getLogger(__name__)


def samples_per_step(config: TrainConfig, dp_size: int) -> int:
    """Samples one optimizer step consumes; the unit of the data cursor.

    The cursor, the saved ``consumed_samples`` and the resume check all read
    this one definition, so "one step" is the same number of samples wherever
    it is counted.
    """
    return config.batch_per_device * dp_size * config.grad_accum_steps


@dataclass
class TrainContext:
    model: nn.Module = field(default=None)
    strategy: BaseStrategy = field(default=None)
    dataset: Optional[Dataset] = field(default=None)
    val_dataset: Optional[Dataset] = field(default=None)
    dataloader: DataLoader = field(default=None)
    #: Iterator the trainer is draining right now. It is state (not just a
    #: cleanup detail): the shutdown hook closes it so a worker death cannot
    #: abort the save, and a reader can see what the run is consuming.
    train_iterator: Any = field(default=None)
    optimizer: Optimizer = field(default=None)
    scheduler: LRScheduler = field(default=None)
    checkpoint: Checkpoint = field(default=None)
    config: TrainConfig = field(default=None)
    model_config: Dict = field(default_factory=dict)
    executor: BaseExecutor = field(default=None)
    epoch: int = field(default=0)
    consumed_samples: int = field(default=0)
    #: Offset in the unpadded global epoch permutation at the last commit.
    epoch_sample_offset: int = field(default=0)
    #: A partially committed multi-update window must never be published.
    checkpoint_safe: bool = field(default=True)
    loss: float = field(default=0.0)
    metrics: Dict[str, float] = field(default_factory=dict)
    grad_norm: Optional[float] = field(default=None)
    grad_snr_tracker: GradSNRTracker = field(default_factory=GradSNRTracker)
    grad_snr_value: Optional[float] = field(default=None)
    val_dataloader: Optional[DataLoader] = field(default=None)
    val_loss: Optional[float] = field(default=None)
    #: Online-strategy validation: reward-statistics evaluator under its
    #: own sampling params. ``None`` keeps the legacy validate_online path.
    val_evaluator: Optional[RolloutEvaluator] = field(default=None)
    async_rollout: Optional[RolloutRunner] = field(default=None)
    #: Committed optimizer steps. Advanced once per commit in every rollout
    #: mode, so a checkpoint's step number never has to be derived from the
    #: data cursor (a batch that is replayed yields several steps).
    optimizer_steps_completed: int = field(default=0)
    #: Samples read since the previous optimizer commit. An incomplete
    #: epoch-tail window is discarded and never enters this count.
    pending_samples: int = field(default=0)
    #: A stop request that every rank has agreed on, so leaving the loop cannot
    #: strand a peer inside a collective.
    stop_agreed: bool = field(default=False)

    world_size: int = field(default=1)
    rank: int = field(default=0)
    topology: Optional[ParallelTopology] = field(default=None)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    param_path: Optional[str] = field(default=None)

    _stop_event: threading.Event = field(default_factory=threading.Event)
    #: Named cleanup steps, run by the session before any observer writes.
    _shutdown_hooks: Dict[str, Callable[[], None]] = field(default_factory=dict)

    def __post_init__(self):
        # A topology is always answerable: contexts built outside the
        # builder (tests, tools) still get rank-layout math from their
        # world_size, with a trivial layout by definition.
        if self.topology is None:
            self.topology = ParallelTopology(self.world_size)

    @property
    def dp_size(self) -> int:
        """Data-parallel degree — the unit for data sharding and step math.

        Ranks sharing a dp_rank consume the same batch (cp peers hold
        different sequence slices of it), so sample and step counts derived
        from world size must use this instead.
        """
        return self.topology.dp_size

    @property
    def dataset_len(self) -> int:
        if self.dataset is None:
            raise RuntimeError("Training dataset has not been initialized")
        return len(self.dataset)

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        self._stop_event.set()

    def on_shutdown(self, name: str, hook: Callable[[], None]) -> None:
        """Register a cleanup step for the session to run at shutdown.

        Registering the same name twice replaces the hook, so a loop that runs
        once per epoch does not accumulate dead iterators.
        """
        self._shutdown_hooks[name] = hook

    def run_shutdown(self) -> None:
        """Run the registered cleanup steps, even when one of them raises."""
        error = None
        for name, hook in list(self._shutdown_hooks.items()):
            try:
                hook()
            except BaseException as failure:
                logger.exception("Shutdown hook %r failed", name)
                error = error or failure
        if error is not None:
            raise error

    @property
    def samples_per_step(self) -> int:
        """Samples one optimizer step consumes; the unit of the data cursor."""
        return samples_per_step(self.config, self.dp_size)

    @property
    def optimizer_step(self) -> int:
        """Committed optimizer steps; the checkpoint's ``meta.optimizer_step``."""
        return self.optimizer_steps_completed
