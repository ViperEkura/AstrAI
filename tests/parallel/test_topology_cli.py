from __future__ import annotations

import json

from click.testing import CliRunner

from scripts.tools.plan_dlo_topology import main
from tests.parallel.test_topology import TOPOLOGY_8X_5060_TI


def test_plan_cli_emits_astrai_and_distributed_dlo_launch_contract(tmp_path) -> None:
    topology_path = tmp_path / "topology.txt"
    topology_path.write_text(TOPOLOGY_8X_5060_TI, encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["--topology-file", str(topology_path), "--concurrent-requests", "1"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dlo_group_kind"] == "sp"
    assert payload["launch"]["astrai_env"]["ASTRAI_DEVICE_ORDER"] == ("0,1,2,3,4,5,6,7")
    assert (
        "--enable-distributed-layerwise-offload" in payload["launch"]["vllm_omni_args"]
    )
    assert payload["launch"]["vllm_omni_args"][:8] == [
        "--num-gpus",
        "8",
        "--data-parallel-size",
        "1",
        "--usp",
        "8",
        "--ring",
        "1",
    ]
