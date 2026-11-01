"""Shutdown must drop the DataLoader workers before writing a checkpoint."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from astrai.parallel.setup import follow_parent
from astrai.parallel.topology import ParallelTopology
from astrai.serialization import Checkpoint
from astrai.trainer import Trainer
from astrai.trainer.callbacks import TrainCallback
from astrai.trainer.components import build_dataloader
from tests.support.rollout import RandomTokenDataset
from tests.support.trainer import create_train_config

ROOT = Path(__file__).resolve().parents[3]

_LOADER_PARENT = """
import sys, time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from astrai.parallel.topology import ParallelTopology
from astrai.trainer.components import build_dataloader
from tests.support.trainer import create_train_config


class SlowDataset(Dataset):
    def __init__(self, length=64, max_length=8, vocab_size=100, **_):
        self.length = length
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.served = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.served += 1
        if self.served > 2:
            time.sleep(120)
        ids = torch.zeros(self.max_length, dtype=torch.long)
        return {"input_ids": ids, "target_ids": ids.clone()}


out = Path(sys.argv[1])
dataset = SlowDataset()
config = create_train_config(
    dataset, str(out), device="cpu", num_workers=2, batch_per_device=2
)
loader = build_dataloader(
    dataset, config.dataset, config, ParallelTopology(1), epoch=0, start_iter=0
)
iterator = iter(loader)
next(iterator)
(out / "workers.pid").write_text(" ".join(str(w.pid) for w in iterator._workers))
time.sleep(120)
"""


class _StopAndRecordWorkers(TrainCallback):
    """Stop after N steps, recording the DataLoader workers first."""

    def __init__(self, steps: int, kill_workers: bool = False):
        self.steps = steps
        self.kill_workers = kill_workers
        self.worker_pids = []

    def after_optimizer_step(self, context):
        if self.worker_pids or context.optimizer_step < self.steps:
            return
        self.worker_pids = [
            worker.pid for worker in getattr(context.train_iterator, "_workers", [])
        ]
        # Stop at a committed window before a worker's SIGCHLD can interrupt us.
        context.request_stop()
        if self.kill_workers:
            for pid in self.worker_pids:
                os.kill(pid, signal.SIGKILL)


def _checkpoints(directory):
    paths = list(directory.glob("epoch_*_step_*"))
    return sorted(paths, key=lambda p: int(p.name.rsplit("_", 1)[1]))


def _alive(pid):
    """A zombie still answers ``kill(pid, 0)``; only a running process counts."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def _config(train_config_factory, directory):
    return train_config_factory(
        RandomTokenDataset(length=64, max_length=8, vocab_size=100),
        str(directory),
        device="cpu",
        n_epoch=2,
        batch_per_device=2,
        grad_accum_steps=2,
        ckpt_interval=1000,
        random_seed=1234,
        num_workers=2,
    )


def _assert_emergency_checkpoint(directory, config, stopper):
    paths = _checkpoints(directory)
    assert paths, "no emergency checkpoint was written"
    meta = json.loads((paths[-1] / "meta.json").read_text())
    assert meta["optimizer_step"] == stopper.steps
    per_step = config.batch_per_device * config.grad_accum_steps
    assert meta["consumed_samples"] == meta["optimizer_step"] * per_step
    Checkpoint.load(str(paths[-1]))
    assert stopper.worker_pids
    assert not any(_alive(pid) for pid in stopper.worker_pids), "workers survived"


def _wait_for_file(path, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


def test_stop_shuts_down_dataloader_workers(
    train_config_factory, trainer_dataset, tmp_path
):
    """A stop closes the workers before the checkpoint save and leaves none behind."""
    config = _config(train_config_factory, tmp_path)
    stopper = _StopAndRecordWorkers(steps=3)
    Trainer(config, callbacks=[stopper]).train()

    _assert_emergency_checkpoint(tmp_path, config, stopper)


def test_worker_killed_at_stop_keeps_the_checkpoint(
    train_config_factory, trainer_dataset, tmp_path
):
    """A worker killed by the stop signal must not cost the emergency checkpoint."""
    config = _config(train_config_factory, tmp_path)
    stopper = _StopAndRecordWorkers(steps=3, kill_workers=True)

    try:
        Trainer(config, callbacks=[stopper]).train()
    except RuntimeError:
        # The dead worker surfaces before shutdown takes over; the save must
        # still happen afterwards.
        pass

    _assert_emergency_checkpoint(tmp_path, config, stopper)


def test_dataloader_workers_carry_the_worker_init_fn(tmp_path):
    """Every multi-worker loader hands the workers the parent-death hook."""
    dataset = RandomTokenDataset(length=8, max_length=4, vocab_size=10)
    config = create_train_config(
        dataset, str(tmp_path), device="cpu", num_workers=2, batch_per_device=2
    )
    loader = build_dataloader(
        dataset, config.dataset, config, ParallelTopology(1), epoch=0, start_iter=0
    )
    assert loader.worker_init_fn is follow_parent


def test_dataloader_workers_die_with_their_rank(tmp_path):
    """Killing a rank must take its blocked DataLoader workers with it."""
    script = tmp_path / "loader_parent.py"
    script.write_text(_LOADER_PARENT)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)], cwd=str(ROOT), env=env
    )
    pids = []
    try:
        pids = [int(part) for part in _wait_for_file(tmp_path / "workers.pid").split()]
        assert len(pids) == 2
        proc.kill()
        proc.wait(timeout=10)

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
            time.sleep(0.05)
        assert not any(_alive(pid) for pid in pids), "workers outlived their rank"
    finally:
        for pid in pids:
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
