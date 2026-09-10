"""Context (sequence) parallelism primitives — universal, executor-free.

Splits the *sequence* dimension across a cp group.  Everything except
attention is per-token and runs on plain local tensors; the experimental
``torch.distributed.tensor.experimental.context_parallel`` context manager
(1) shards the caller's batch buffers in place along their sequence dims
(contiguous chunks; the head-tail load balancer stays disabled) and
(2) patches ``F.scaled_dot_product_attention`` so the SDPA call inside the
model becomes ring attention over the cp group.  The patch accepts plain
``[B, H, S, D]`` q/k/v, so no model code sees a DTensor.

Loss normalization: each rank computes ``cross_entropy(..., reduction="sum")``
over its local (load-balanced) slice.  ``mean_loss`` turns that into the
true global mean via one (sum, count) all-reduce, and ``scaled_loss`` scales
it by ``cp_size`` — data-parallel gradient reducers (DDP / FSDP) average over
groups that contain all cp peers, so without the factor the averaged
gradient would be ``1/cp`` of the single-device value.

Strategy composition: :class:`CPStrategy` decorates any strategy that
implements the token-mean two-phase protocol (:class:`TokenMeanStrategy`)
— the strategy runs its normal forward/reduction code on whatever
sequence it is handed, and the decorator owns the shard entry and the
loss rescale.  Strategies never grow a CP twin method.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Protocol, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor.experimental import context_parallel
from torch.nn.attention import SDPBackend, sdpa_kernel

from astrai.extension.backend.attention import TorchNativeBackend, get_backend
from astrai.parallel.topology import ParallelTopology

try:  # private torch module — torch may move or rename it between versions
    from torch.distributed.tensor.experimental._context_parallel._attention import (
        _cp_options,
    )
except ImportError:  # pragma: no cover - load balancing then stays default
    _cp_options = None

logger = logging.getLogger(__name__)


class LossReduction(Enum):
    """How a training loss reduces over the batch.

    The declared reduction decides context-parallel compatibility:
    TOKEN_MEAN losses shard freely (each rank sums its own tokens), while
    SEQUENCE losses need cross-shard protocols that do not exist yet —
    logprob shifts that straddle shard boundaries and per-sequence
    reductions would need cp-group all-reduces.
    """

    TOKEN_MEAN = auto()
    SEQUENCE = auto()


@dataclass
class TokenLoss:
    """Local token-level reduction — the CP-compatible loss currency.

    ``loss_sum`` carries grad; ``token_count`` is the number of tokens
    that contributed.  ``mean()`` is what a single-process run would
    backward; under CP the counts (and, for reporting, the sums) are
    group-wide.
    """

    loss_sum: Tensor
    token_count: Tensor

    def mean(self) -> Tensor:
        return self.loss_sum / self.token_count.clamp(min=1.0)


class TokenMeanStrategy(Protocol):
    """The strategy contract :class:`CPStrategy` composes with.

    Mirrors the phase methods of ``BaseStrategy`` (astrai.trainer.strategy)
    so this layer stays trainer-import free.  The forward/output types are
    opaque to the decorator and typed ``Any`` here; concretely they are
    the strategy's forward-result and loss-output dataclasses.
    """

    loss_reduction: LossReduction

    def prepare_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]: ...

    def shard_spec(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[List[Tensor], List[int]]: ...

    def forward_tokens(self, batch: Dict[str, Tensor]) -> Any: ...

    def reduce_loss(self, forward: Any, batch: Dict[str, Tensor]) -> TokenLoss: ...

    def token_loss_output(
        self, loss: Tensor, reported_loss: Tensor, forward: Any
    ) -> Any: ...


def _sdpa_is_native() -> bool:
    """Whether the active attention backend routes through torch SDPA.

    Ring attention hooks ``F.scaled_dot_product_attention`` only; the CUDA
    paged kernels and FlashAttention never see the patch.  Fail loudly per
    the repo's no-silent-fallback convention.
    """
    return isinstance(get_backend(), TorchNativeBackend)


class CPState:
    """Per-run context-parallel coordinator.

    Held by whoever drives the forward pass (a strategy today, the serving
    runtime later); knows only the topology, never an executor.
    """

    def __init__(self, topology: ParallelTopology):
        if topology.cp_size <= 1 or topology.cp_mesh is None:
            raise ValueError(
                f"CPState requires an active cp dimension, got cp_size="
                f"{topology.cp_size}"
            )
        self.topology = topology
        self._disable_load_balance()

    @staticmethod
    def _disable_load_balance() -> None:
        """Fall back to contiguous (unbalanced) sequence sharding.

        torch 2.11's head-tail load balancer is broken in our setup: the
        partial-attention merge mixes with the mem-efficient kernel's
        max(q,kv)-shaped LSE, and even on flash kernels the balanced path
        produced a halved loss in the cp=2 equivalence check.  Contiguous
        chunks plus the ring's causal SKIP logic are exact — later ranks
        just do more cross-chunk attention work.
        """
        if _cp_options is not None:
            _cp_options.enable_load_balance = False

    @property
    def size(self) -> int:
        return self.topology.cp_size

    @property
    def rank(self) -> int:
        return self.topology.cp_rank

    @contextmanager
    def shard(self, buffers: List[Tensor], seq_dims: Sequence[int]):
        """Shard ``buffers`` in place across the cp group inside the context.

        Buffers are mutated to the rank's local sequence slice (contiguous;
        see :meth:`CPState._disable_load_balance`) and not restored on exit.
        All buffers in one call share the same rearrangement, so
        position-aligned tensors (inputs, targets, positions) stay aligned.
        """
        if not _sdpa_is_native():
            raise RuntimeError(
                "context parallelism requires the torch-native attention "
                "backend (SDPA); the active backend bypasses the ring hook"
            )

        # The ring's partial-attention merge assumes logsumexp carries one
        # entry per *query* row.  The mem-efficient kernel returns LSE sized
        # max(q_len, kv_len) instead (verified on torch 2.11), which breaks
        # the merge with a shape error; math would silently skip the ring.
        # Restrict to the flash/cuDNN kernels, whose LSE is q_len-shaped.
        #
        # Buffers are not restored on exit: autograd saves input/target ids
        # (embedding and cross-entropy backward need them), and the restore
        # resize_+copy_ would bump their versions and abort backward.  The
        # caller consumes the batch inside the context, so shards are final.
        with (
            sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]),
            context_parallel(
                self.topology.cp_mesh,
                buffers=list(buffers),
                buffer_seq_dims=list(seq_dims),
                no_restore_buffers=set(buffers),
            ),
        ):
            yield

    def mean_loss(self, loss_sum: Tensor, token_count: Tensor) -> tuple[Tensor, Tensor]:
        """Global-mean loss and its cp-scaled training counterpart.

        Returns ``(mean, scaled)``: ``mean`` is the true global mean (sum
        over the whole cp group divided by the global token count — a plain
        value for logging/validation) and ``scaled = local_sum * cp_size /
        global_count`` is the tensor to backward.  Data-parallel gradient
        reducers average over groups containing all cp peers, so without
        the ``cp_size`` factor the averaged gradient would be ``1/cp`` of
        the single-device value.
        """
        stats = torch.stack(
            [
                loss_sum.detach().float(),
                token_count.detach().float(),
            ]
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=self.topology.cp_group)
        total, count = stats[0], stats[1].clamp(min=1.0)
        mean = (total / count).detach()
        scaled = loss_sum * (self.size / count)
        return mean, scaled


class CPStrategy:
    """Run a token-mean strategy under context parallelism (decorator).

    The wrapped strategy implements the two-phase protocol and stays
    CP-oblivious: its forward and reduction run unchanged on full
    sequences and on this rank's shard alike.  This wrapper owns the CP
    specifics — entering the shard between batch preparation and the
    forward, keeping the reduction inside it, and rescaling the local
    token loss to the global mean.  Everything else (model, optimizer
    hooks, rollout plumbing) proxies to the wrapped strategy, so trainer
    code is unaware of the wrapping.
    """

    def __init__(self, inner: TokenMeanStrategy, cp: CPState):
        if getattr(inner, "loss_reduction", None) is not LossReduction.TOKEN_MEAN:
            raise ValueError(
                f"{type(inner).__name__} does not declare a token-mean "
                f"loss reduction; context parallelism cannot shard it"
            )
        self.inner = inner
        self.cp = cp

    def __getattr__(self, name: str) -> Any:
        inner = self.__dict__.get("inner")
        if inner is None:  # pre-__init__ or unpickled access
            raise AttributeError(name)
        return getattr(inner, name)

    def __call__(self, batch: Dict[str, Tensor]) -> Any:
        # CP training is offline-only: sequence-reducing online strategies
        # are refused at construction, so there is no rollout dispatch here.
        return self.compute_loss_output(batch)

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.compute_loss_output(batch)["loss"]

    def compute_loss_output(self, batch: Dict[str, Tensor]) -> Any:
        batch = self.inner.prepare_batch(batch)
        buffers, seq_dims = self.inner.shard_spec(batch)
        with self.cp.shard(buffers, seq_dims):
            forward = self.inner.forward_tokens(batch)
            tokens = self.inner.reduce_loss(forward, batch)
        mean, scaled = self.cp.mean_loss(tokens.loss_sum, tokens.token_count)
        return self.inner.token_loss_output(scaled, mean, forward)
