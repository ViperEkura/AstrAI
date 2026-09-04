"""Topology-aware device placement for DP/SP layerwise offload.

The planner models the rank order used by vLLM-Omni diffusion workers: sequence
parallel ranks are contiguous and data parallel ranks are the outer dimension.
Distributed layerwise offload (DLO) therefore communicates over DP when DP is
larger than one, otherwise it communicates over SP.  Topology labels are only a
fallback: measured collective timings take precedence when supplied.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

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
