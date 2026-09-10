from __future__ import annotations

import pytest

from astrai.parallel.topology import (
    DLOMeasurement,
    build_parallel_groups,
    parse_device_order,
    parse_nvidia_topology,
    select_dlo_plan,
)

TOPOLOGY_8X_5060_TI = """
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    CPU Affinity    NUMA Affinity
GPU0     X      PHB     NODE    NODE    NODE    NODE    NODE    NODE    0-63            0
GPU1    PHB      X      NODE    NODE    NODE    NODE    NODE    NODE    0-63            0
GPU2    NODE    NODE     X      PHB     NODE    NODE    NODE    NODE    0-63            0
GPU3    NODE    NODE    PHB      X      NODE    NODE    NODE    NODE    0-63            0
GPU4    NODE    NODE    NODE    NODE     X      PHB     NODE    NODE    0-63            0
GPU5    NODE    NODE    NODE    NODE    PHB      X      NODE    NODE    0-63            0
GPU6    NODE    NODE    NODE    NODE    NODE    NODE     X      PHB     0-63            0
GPU7    NODE    NODE    NODE    NODE    NODE    NODE    PHB      X      0-63            0
Legend:
  X    = Self
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge
"""


def test_parse_nvidia_topology_and_reject_partial_matrix() -> None:
    topology = parse_nvidia_topology("\x1b[0m" + TOPOLOGY_8X_5060_TI)
    assert topology.devices == tuple(range(8))
    assert topology.link(0, 1) == "PHB"
    assert topology.link(0, 2) == "NODE"

    with pytest.raises(ValueError, match="row mismatch"):
        parse_nvidia_topology(
            TOPOLOGY_8X_5060_TI.replace("GPU7    NODE", "CPU7    NODE")
        )


def test_parse_single_gpu_topology() -> None:
    topology = parse_nvidia_topology("GPU0 CPU Affinity NUMA Affinity\nGPU0 X 0-31 0\n")
    assert topology.devices == (0,)
    assert topology.link(0, 0) == "X"


def test_sp_fastest_group_layout_matches_dlo_dp_precedence() -> None:
    order = (0, 2, 4, 6, 1, 3, 5, 7)
    dp_groups, sp_groups = build_parallel_groups(order, dp_size=2, sp_size=4)
    assert dp_groups == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert sp_groups == ((0, 2, 4, 6), (1, 3, 5, 7))

    plan = select_dlo_plan(
        parse_nvidia_topology(TOPOLOGY_8X_5060_TI),
        concurrent_requests=2,
    )
    assert plan.dlo_group_kind == "dp"
    assert plan.dp_size == 2
    assert plan.sp_size == 4
    assert plan.dlo_groups == dp_groups


def test_single_request_is_dp1_sp8_and_uses_sp_for_dlo() -> None:
    plan = select_dlo_plan(
        parse_nvidia_topology(TOPOLOGY_8X_5060_TI), concurrent_requests=1
    )
    assert (plan.dp_size, plan.sp_size) == (1, 8)
    assert plan.dlo_group_kind == "sp"
    assert plan.dlo_groups == (tuple(range(8)),)

    with pytest.raises(ValueError, match="concurrent request"):
        select_dlo_plan(
            parse_nvidia_topology(TOPOLOGY_8X_5060_TI),
            concurrent_requests=1,
            dp_size=2,
            sp_size=4,
        )


def test_measurement_overrides_topology_label_heuristic() -> None:
    topology = parse_nvidia_topology(TOPOLOGY_8X_5060_TI)
    heuristic = select_dlo_plan(topology, concurrent_requests=2)
    natural = tuple(range(8))
    measured = select_dlo_plan(
        topology,
        concurrent_requests=2,
        measurements=(
            DLOMeasurement(2, 4, heuristic.device_order, 4.0, 1.0),
            DLOMeasurement(2, 4, natural, 1.0, 1.0),
        ),
    )
    assert heuristic.device_order != natural
    assert measured.device_order == natural
    assert measured.selection_source == "measured-collectives"
    assert measured.score == 2.0


@pytest.mark.parametrize(
    ("value", "local_world_size"),
    [("0,2,1,3", 4), ("0", 1)],
)
def test_parse_device_order(value: str, local_world_size: int) -> None:
    assert parse_device_order(value, local_world_size) == tuple(
        int(item) for item in value.split(",")
    )


@pytest.mark.parametrize("value", ["0,1,1,3", "0,1", "0,-1,2,3", "0,1,2,4", "0,x,2,3"])
def test_parse_device_order_rejects_invalid_mapping(value: str) -> None:
    with pytest.raises(ValueError):
        parse_device_order(value, 4)
