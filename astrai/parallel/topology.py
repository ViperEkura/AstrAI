"""Parallel topology: decomposition of the global world into (dp, cp, tp).

This module is the universal rank-layout layer shared by every parallel
consumer (trainer, sampler, future inference runtime).  It knows nothing
about executors, strategies, or checkpoints.

Rank mapping is row-major over ``mesh = (dp, cp, tp)``::

    global_rank = dp_idx * (cp * tp) + cp_idx * tp + tp_idx

so the tp dimension is innermost (contiguous ranks — fastest interconnect)
and each cp group is a contiguous block when ``tp_size == 1``.

``tp_size > 1`` activates the tp dimension for
:class:`astrai.parallel.tp.TPState`; combined cp+tp layouts are refused at
the trainer level until their interaction is verified.  ``cp_size ==
tp_size == 1`` builds a trivial topology with no new process groups —
byte-identical to the pre-topology single-dimension behavior.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import logging
from typing import Optional

import torch.distributed as dist
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

logger = logging.getLogger(__name__)

# Meshes cached per (world, dp, cp, tp, device) so repeated topology builds
# in one process never duplicate process-group creation.
_MESH_CACHE: dict[tuple[int, int, int, int, str], DeviceMesh] = {}


class ParallelTopology:
    """Rank layout for ``world_size = dp_size * cp_size * tp_size``.

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
    ):
        if tp_size < 1 or cp_size < 1:
            raise ValueError(
                f"cp_size and tp_size must be >= 1, got cp={cp_size} tp={tp_size}"
            )
        if world_size % (cp_size * tp_size) != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"cp_size * tp_size ({cp_size * tp_size})"
            )
        self.world_size = world_size
        self.cp_size = cp_size
        self.tp_size = tp_size
        self.dp_size = world_size // (cp_size * tp_size)
        self.device_type = device_type

        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_rank = self.global_rank % tp_size
        self.cp_rank = (self.global_rank // tp_size) % cp_size
        self.dp_rank = self.global_rank // (cp_size * tp_size)

        self.mesh: Optional[DeviceMesh] = None
        self.dp_group = None
        self.cp_group = None
        self.tp_group = None

        # Only materialize a mesh when a parallel dimension is active — the
        # trivial path creates no groups and stays byte-identical to legacy.
        if world_size > 1 and (cp_size > 1 or tp_size > 1):
            key = (world_size, self.dp_size, cp_size, tp_size, device_type)
            mesh = _MESH_CACHE.get(key)
            if mesh is None:
                mesh = init_device_mesh(
                    device_type,
                    (self.dp_size, cp_size, tp_size),
                    mesh_dim_names=("dp", "cp", "tp"),
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
                "parallel topology: world=%d dp=%d cp=%d tp=%d "
                "(rank %d -> dp=%d cp=%d tp=%d)",
                world_size,
                self.dp_size,
                cp_size,
                tp_size,
                self.global_rank,
                self.dp_rank,
                self.cp_rank,
                self.tp_rank,
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
        return self.cp_size == 1 and self.tp_size == 1

    def is_dp_leader(self) -> bool:
        """Whether this rank leads its dp group (cp_rank == tp_rank == 0).

        Ranks with the same dp_rank consume the same batch: cp peers hold
        different sequence slices of it, tp peers identical copies.
        """
        return self.cp_rank == 0 and self.tp_rank == 0

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
    cp_size: int = 1, tp_size: int = 1, device_type: str = "cuda"
) -> ParallelTopology:
    """Build the topology for the current process group.

    Requires the WORLD group to be initialized when any dimension is active.
    """
    world = dist.get_world_size() if dist.is_initialized() else 1
    if (cp_size > 1 or tp_size > 1) and not dist.is_initialized():
        raise RuntimeError(
            "context/tensor parallelism requires an initialized process "
            "group; run inside setup_parallel/spawn_parallel_fn"
        )
    return ParallelTopology(world, cp_size, tp_size, device_type)


_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_GPU_TOKEN = re.compile(r"^GPU(\d+)$")
_KNOWN_LINKS = {"X", "PIX", "PXB", "PHB", "NODE", "SYS"}
_LINK_WEIGHT = {
    "SYS": 1.0,
    "NODE": 2.0,
    "PHB": 3.0,
    "PXB": 4.0,
    "PIX": 5.0,
    "X": 7.0,
}


def _link_weight(link: str) -> float:
    if link.startswith("NV") and link[2:].isdigit():
        return 6.0 + int(link[2:]) / 100.0
    try:
        return _LINK_WEIGHT[link]
    except KeyError as exc:
        raise ValueError(f"unknown GPU topology link: {link}") from exc


@dataclass(frozen=True)
class GPUTopology:
    """A symmetric GPU connectivity matrix parsed from ``nvidia-smi topo -m``."""

    devices: tuple[int, ...]
    links: Mapping[tuple[int, int], str]

    def link(self, left: int, right: int) -> str:
        if left == right:
            return "X"
        try:
            return self.links[(left, right)]
        except KeyError as exc:
            raise ValueError(f"missing topology link GPU{left} -> GPU{right}") from exc

    def affinity(self, left: int, right: int) -> float:
        return _link_weight(self.link(left, right))


@dataclass(frozen=True)
class DLOMeasurement:
    """Measured steady-state collective cost for one physical rank mapping."""

    dp_size: int
    sp_size: int
    device_order: tuple[int, ...]
    dlo_all_gather_ms: float
    sp_collective_ms: float

    @property
    def combined_ms(self) -> float:
        return self.dlo_all_gather_ms + self.sp_collective_ms


@dataclass(frozen=True)
class DLOTopologyPlan:
    """A DP/SP shape and logical-rank-to-physical-device mapping."""

    dp_size: int
    sp_size: int
    device_order: tuple[int, ...]
    dlo_group_kind: str
    dlo_groups: tuple[tuple[int, ...], ...]
    sp_groups: tuple[tuple[int, ...], ...]
    selection_source: str
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "dp_size": self.dp_size,
            "sp_size": self.sp_size,
            "device_order": list(self.device_order),
            "dlo_group_kind": self.dlo_group_kind,
            "dlo_groups": [list(group) for group in self.dlo_groups],
            "sp_groups": [list(group) for group in self.sp_groups],
            "selection_source": self.selection_source,
            "score": self.score,
        }


def parse_nvidia_topology(text: str) -> GPUTopology:
    """Parse and validate the GPU matrix emitted by ``nvidia-smi topo -m``.

    The parser intentionally fails closed on partial or asymmetric matrices.
    CPU-affinity and NUMA columns following the GPU matrix are ignored.
    """

    clean = _ANSI_ESCAPE.sub("", text)
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if (
                len([token for token in line.split() if _GPU_TOKEN.match(token)]) >= 2
                or (
                    len([token for token in line.split() if _GPU_TOKEN.match(token)])
                    == 1
                    and len(line.split()) > 1
                    and line.split()[1] not in _KNOWN_LINKS
                )
            )
        ),
        None,
    )
    if header_index is None:
        raise ValueError("nvidia-smi topology output has no GPU header")

    header = lines[header_index].split()
    devices = tuple(
        int(match.group(1))
        for token in header
        if (match := _GPU_TOKEN.match(token)) is not None
    )
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("GPU topology header is empty or contains duplicate devices")

    rows: dict[int, tuple[str, ...]] = {}
    for line in lines[header_index + 1 :]:
        tokens = line.split()
        match = _GPU_TOKEN.match(tokens[0]) if tokens else None
        if match is None:
            continue
        device = int(match.group(1))
        if device in rows:
            raise ValueError(f"duplicate GPU{device} topology row")
        if len(tokens) < len(devices) + 1:
            raise ValueError(f"incomplete GPU{device} topology row")
        rows[device] = tuple(tokens[1 : len(devices) + 1])

    if set(rows) != set(devices):
        missing = sorted(set(devices) - set(rows))
        extra = sorted(set(rows) - set(devices))
        raise ValueError(f"GPU topology row mismatch: missing={missing}, extra={extra}")

    links: dict[tuple[int, int], str] = {}
    for row_device in devices:
        for column, column_device in enumerate(devices):
            link = rows[row_device][column]
            if link not in _KNOWN_LINKS and not (
                link.startswith("NV") and link[2:].isdigit()
            ):
                raise ValueError(
                    f"unknown topology token {link!r} for GPU{row_device}/GPU{column_device}"
                )
            if row_device == column_device and link != "X":
                raise ValueError(f"GPU{row_device} diagonal must be X, got {link}")
            links[(row_device, column_device)] = link

    for left in devices:
        for right in devices:
            if links[(left, right)] != links[(right, left)]:
                raise ValueError(
                    f"asymmetric GPU topology: GPU{left}/GPU{right} is "
                    f"{links[(left, right)]}/{links[(right, left)]}"
                )
    return GPUTopology(devices=devices, links=links)


def build_parallel_groups(
    device_order: Sequence[int], dp_size: int, sp_size: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Return physical DP and SP groups for an SP-fastest logical rank order."""

    order = tuple(device_order)
    if dp_size <= 0 or sp_size <= 0:
        raise ValueError("DP and SP sizes must be positive")
    if dp_size * sp_size != len(order):
        raise ValueError("DP * SP must equal the device count")
    if len(set(order)) != len(order):
        raise ValueError("device_order must not contain duplicates")

    sp_groups = tuple(
        tuple(order[dp_rank * sp_size : (dp_rank + 1) * sp_size])
        for dp_rank in range(dp_size)
    )
    dp_groups = tuple(
        tuple(order[dp_rank * sp_size + sp_rank] for dp_rank in range(dp_size))
        for sp_rank in range(sp_size)
    )
    return dp_groups, sp_groups


def dlo_groups_for_plan(
    device_order: Sequence[int], dp_size: int, sp_size: int
) -> tuple[str, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Select the DLO group using DP-first, SP-second backend semantics."""

    dp_groups, sp_groups = build_parallel_groups(device_order, dp_size, sp_size)
    if dp_size > 1:
        return "dp", dp_groups, sp_groups
    if sp_size > 1:
        return "sp", sp_groups, sp_groups
    return "rank-local", tuple((device,) for device in device_order), sp_groups


def _mean_group_affinity(
    topology: GPUTopology, groups: Sequence[Sequence[int]]
) -> float:
    values = [
        topology.affinity(left, right)
        for group in groups
        for left_index, left in enumerate(group)
        for right in group[left_index + 1 :]
    ]
    return sum(values) / len(values) if values else _link_weight("X")


def topology_score(
    topology: GPUTopology,
    device_order: Sequence[int],
    dp_size: int,
    sp_size: int,
    *,
    dlo_weight: float = 4.0,
    sp_weight: float = 1.0,
) -> float:
    """Score a mapping using link labels when measurements are unavailable."""

    if set(device_order) != set(topology.devices):
        raise ValueError("device_order must contain every topology device exactly once")
    _, dlo_groups, sp_groups = dlo_groups_for_plan(device_order, dp_size, sp_size)
    return dlo_weight * _mean_group_affinity(
        topology, dlo_groups
    ) + sp_weight * _mean_group_affinity(topology, sp_groups)


def optimize_device_order(
    topology: GPUTopology,
    dp_size: int,
    sp_size: int,
    *,
    exhaustive_limit: int = 8,
) -> tuple[tuple[int, ...], float]:
    """Find a deterministic topology-label optimum for a fixed DP/SP shape."""

    if dp_size * sp_size != len(topology.devices):
        raise ValueError("DP * SP must equal the topology device count")
    natural = tuple(sorted(topology.devices))
    if len(natural) > exhaustive_limit:
        return natural, topology_score(topology, natural, dp_size, sp_size)

    best_order = natural
    best_score = float("-inf")
    for order in itertools.permutations(natural):
        score = topology_score(topology, order, dp_size, sp_size)
        if score > best_score or (score == best_score and order < best_order):
            best_order = order
            best_score = score
    return best_order, best_score


def select_dlo_plan(
    topology: GPUTopology,
    *,
    concurrent_requests: int = 1,
    dp_size: int | None = None,
    sp_size: int | None = None,
    measurements: Iterable[DLOMeasurement] = (),
) -> DLOTopologyPlan:
    """Select a DP/SP plan, preferring valid measured collective results.

    When no shape is requested explicitly, the largest DP divisor that does
    not exceed ``concurrent_requests`` is selected.  This prevents a single
    request from being silently assigned to multiple data-parallel replicas.
    """

    world_size = len(topology.devices)
    if concurrent_requests <= 0:
        raise ValueError("concurrent_requests must be positive")
    if (dp_size is None) != (sp_size is None):
        raise ValueError("dp_size and sp_size must be provided together")
    if dp_size is None:
        eligible = [
            candidate
            for candidate in range(1, world_size + 1)
            if world_size % candidate == 0 and candidate <= concurrent_requests
        ]
        dp_size = max(eligible)
        sp_size = world_size // dp_size
    assert sp_size is not None
    if dp_size * sp_size != world_size:
        raise ValueError("DP * SP must equal the topology device count")
    if dp_size > concurrent_requests:
        raise ValueError("DP size cannot exceed concurrent request capacity")

    valid_measurements = []
    for measurement in measurements:
        if measurement.dp_size != dp_size or measurement.sp_size != sp_size:
            continue
        if set(measurement.device_order) != set(topology.devices):
            raise ValueError("measured device_order does not match topology devices")
        if measurement.dlo_all_gather_ms <= 0 or measurement.sp_collective_ms <= 0:
            raise ValueError("measured collective latency must be positive")
        valid_measurements.append(measurement)

    if valid_measurements:
        best_measurement = min(
            valid_measurements,
            key=lambda item: (item.combined_ms, item.device_order),
        )
        order = best_measurement.device_order
        score = best_measurement.combined_ms
        source = "measured-collectives"
    else:
        order, score = optimize_device_order(topology, dp_size, sp_size)
        source = "topology-label-fallback"

    kind, dlo_groups, sp_groups = dlo_groups_for_plan(order, dp_size, sp_size)
    return DLOTopologyPlan(
        dp_size=dp_size,
        sp_size=sp_size,
        device_order=order,
        dlo_group_kind=kind,
        dlo_groups=dlo_groups,
        sp_groups=sp_groups,
        selection_source=source,
        score=score,
    )


def parse_device_order(value: str, local_world_size: int) -> tuple[int, ...]:
    """Parse ``ASTRAI_DEVICE_ORDER`` as a complete logical-to-visible map."""

    try:
        order = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(
            "ASTRAI_DEVICE_ORDER must be comma-separated integers"
        ) from exc
    if len(order) != local_world_size:
        raise ValueError(
            f"ASTRAI_DEVICE_ORDER has {len(order)} devices, expected {local_world_size}"
        )
    expected = set(range(local_world_size))
    if set(order) != expected:
        raise ValueError(
            "ASTRAI_DEVICE_ORDER must be a permutation of local visible device "
            f"indices 0..{local_world_size - 1}"
        )
    return order
