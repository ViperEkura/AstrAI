"""Trainer rank layout and per-dimension communication groups.

The mesh is (dp, cp, tp), with an additional innermost pp dimension for
pipeline training. Each pipeline consists of contiguous ranks. Peers at
the same stage in different replicas share a data-parallel group.
"""

import logging
from typing import Dict, Optional, Tuple

import torch.distributed as dist
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

logger = logging.getLogger(__name__)

# Meshes cached per (world, dp, cp, tp, pp, device) so repeated topology builds
# in one process never duplicate process-group creation.
_MESH_CACHE: Dict[Tuple[int, int, int, int, int, str], DeviceMesh] = {}


class ParallelTopology:
    """Rank layout for ``world_size = dp_size * cp_size * tp_size * pp_size``.

    Attributes:
        dp_size / cp_size / tp_size: decomposition of the world.
        global_rank: rank in the WORLD group.
        dp_rank / cp_rank / tp_rank: positions inside each group.
        dp_group / cp_group / tp_group: one subgroup per dimension.  All
            three are real groups whenever a mesh exists — a singleton
            where the dimension is inactive — and all three are ``None``
            on a trivial layout (no mesh, the world group plays every
            role, no new communicators: byte-identical to the legacy
            single-dimension path).
        cp_mesh: 1-D ``DeviceMesh`` over the cp dimension, consumed by the
            experimental ``context_parallel`` API.
    """

    def __init__(
        self,
        world_size: int,
        cp_size: int = 1,
        tp_size: int = 1,
        device_type: str = "cuda",
        pp_size: int = 1,
    ):
        if tp_size < 1 or cp_size < 1:
            raise ValueError(
                f"cp_size and tp_size must be >= 1, got cp={cp_size} tp={tp_size}"
            )
        if pp_size < 1 or world_size % (cp_size * tp_size * pp_size) != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"cp_size * tp_size * pp_size ({cp_size * tp_size * pp_size})"
            )
        self.world_size = world_size
        self.cp_size = cp_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = world_size // (cp_size * tp_size * pp_size)
        self.device_type = device_type

        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_rank = (self.global_rank // pp_size) % tp_size
        self.cp_rank = (self.global_rank // (pp_size * tp_size)) % cp_size
        self.pp_rank = self.global_rank % pp_size
        self.dp_rank = self.global_rank // (cp_size * tp_size * pp_size)

        self.mesh: Optional[DeviceMesh] = None
        self.dp_group = None
        self.cp_group = None
        self.tp_group = None

        # Only materialize a mesh when a parallel dimension is active — the
        # trivial path creates no groups and stays byte-identical to legacy.
        if world_size > 1 and (cp_size > 1 or tp_size > 1 or pp_size > 1):
            key = (world_size, self.dp_size, cp_size, tp_size, pp_size, device_type)
            mesh = _MESH_CACHE.get(key)
            if mesh is None:
                mesh = init_device_mesh(
                    device_type,
                    (self.dp_size, cp_size, tp_size)
                    + ((pp_size,) if pp_size > 1 else ()),
                    mesh_dim_names=("dp", "cp", "tp")
                    + (("pp",) if pp_size > 1 else ()),
                )
                _MESH_CACHE[key] = mesh
            self.mesh = mesh
            # Every dimension carries a group once a mesh exists — a
            # singleton where the dimension is inactive.  Consumers can
            # reduce on their own dimension unconditionally, with no
            # world-group fallback that could leak across dimensions (data
            # sharding with dp_size == 1 must never mix cp peers).
            self.dp_group = mesh.get_group(0)
            self.cp_group = mesh.get_group(1)
            self.tp_group = mesh.get_group(2)

            logger.info(
                "parallel topology: world=%d dp=%d cp=%d tp=%d pp=%d "
                "(rank %d -> dp=%d cp=%d tp=%d pp=%d)",
                world_size,
                self.dp_size,
                cp_size,
                tp_size,
                pp_size,
                self.global_rank,
                self.dp_rank,
                self.cp_rank,
                self.tp_rank,
                self.pp_rank,
            )

    @property
    def cp_mesh(self) -> Optional[DeviceMesh]:
        """1-D ``DeviceMesh`` over the cp dimension, consumed by the
        experimental ``context_parallel`` API.  ``None`` when the dimension
        is inactive (no mesh) — :class:`astrai.parallel.cp.CPState` gates
        on this."""
        return self.mesh["cp"] if self.mesh is not None else None

    @property
    def is_trivial(self) -> bool:
        return self.cp_size == 1 and self.tp_size == 1 and self.pp_size == 1

    def is_dp_leader(self) -> bool:
        """Whether this rank leads its dp group (cp_rank == tp_rank == 0).

        Ranks with the same dp_rank consume the same batch: cp peers hold
        different sequence slices of it, tp peers identical copies.
        """
        return self.cp_rank == 0 and self.tp_rank == 0 and self.pp_rank == 0

    def samples_per_replica(self, dataset_len: int) -> int:
        """Dataset length as one dp replica sees it (ceil over ``dp_size``).

        Matches ``RDSampler``'s no-drop-last sharding, so callers can derive
        per-replica epoch/step counts consistently with the sampler.
        """
        return (dataset_len + self.dp_size - 1) // self.dp_size

    def reduce_sum(self, values: Tensor) -> Tensor:
        """Sum ``values`` across the dp dimension; single-process no-op.

        The reduce runs on the dp group (the world group on a trivial
        layout).  cp peers are deliberately excluded: they hold values
        derived from the same batch, and summing them in would
        double-count.
        """
        if self.dp_size <= 1 or not dist.is_initialized():
            return values
        dist.all_reduce(values, op=dist.ReduceOp.SUM, group=self.dp_group)
        return values

    def reduce_mean(self, values: Tensor) -> Tensor:
        """Average ``values`` across the dp dimension; single-process no-op."""
        if self.dp_size <= 1 or not dist.is_initialized():
            return values
        return self.reduce_sum(values) / self.dp_size


def build_topology(
    cp_size: int = 1, tp_size: int = 1, device_type: str = "cuda", pp_size: int = 1
) -> ParallelTopology:
    """Build the topology for the current process group.

    Requires the WORLD group to be initialized when any dimension is active.
    """
    world = dist.get_world_size() if dist.is_initialized() else 1
    if (cp_size > 1 or tp_size > 1 or pp_size > 1) and not dist.is_initialized():
        raise RuntimeError(
            "context/tensor parallelism requires an initialized process "
            "group; run inside setup_parallel/spawn_parallel_fn"
        )
    return ParallelTopology(world, cp_size, tp_size, device_type, pp_size)
