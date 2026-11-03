"""Launch timing is accounted for without publishing process identities."""

import json
import sys

from examples.rl_reward.experiment import launch
from examples.rl_reward.run import Recipe


def test_controller_captures_startup_and_keeps_commands_and_ids_private(tmp_path):
    # A real independent CPU subprocess models launcher startup. The GPU
    # experiment entry point has separate assigned-resource qualification.
    helper = tmp_path / "timed-child.py"
    helper.write_text(
        "import json, pathlib, sys, time\n"
        "time.sleep(0.15)\n"
        "print('ASTRAI_RUNNER_MAIN_STARTED', flush=True)\n"
        "root=pathlib.Path(sys.argv[1]); root.mkdir()\n"
        "(root/'run_end.rank0.fixture.jsonl').write_text(json.dumps({'budget_completed': True, 'optimizer_step': 400})+'\\n')\n"
    )
    run_root = tmp_path / "run"
    controller_root = tmp_path / "controller"
    controller_root.mkdir()
    recipe = Recipe(
        model_path=str(tmp_path / "model"),
        model_repo="test/native",
        model_revision="fixture-v1",
        train_file=str(tmp_path / "train.jsonl"),
        dev_file=str(tmp_path / "dev.jsonl"),
        dataset_repo="test/data",
        dataset_revision="fixture-v1",
        output_dir=str(run_root),
    )
    settings = {
        "output_dir": str(controller_root),
        "launcher_argv": [
            sys.executable,
            str(helper),
            str(run_root),
            "{script}",
            "{recipe}",
        ],
    }
    receipt = launch(settings, recipe, "baseline.seed3407")
    assert receipt["budget_completed"]
    assert receipt["launcher_startup_seconds"] >= 0.1
    assert receipt["controller_seconds"] >= receipt["launcher_startup_seconds"]
    assert "pid" not in receipt and "argv" not in receipt
    private = json.loads(
        (controller_root / "private" / "baseline.seed3407.process.json").read_text()
    )
    assert private["pid"] > 0 and private["argv"]
