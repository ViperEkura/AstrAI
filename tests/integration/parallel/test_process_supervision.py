"""Launcher supervision: a rank, or the launcher itself, can be killed."""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from astrai.parallel import setup as parallel_setup
from astrai.parallel.setup import (
    SHUTDOWN_GRACE_SECONDS,
    agree_stop,
    spawn_parallel_fn,
)
from astrai.signal_handler import register_signal_handlers

ROOT = Path(__file__).resolve().parents[3]


class _Flag:
    """Minimal stop context for ``register_signal_handlers``."""

    def __init__(self):
        self.stopped = False

    def request_stop(self):
        self.stopped = True


def _loop_rank(pid_dir, shutdown_delay=0.0):
    """Rank body: publish the pid, then exchange a reduce until told to stop.

    The loop leaves only on the *agreed* flag: a rank that drops out on a
    locally observed signal while its peer is inside the collective would
    strand that peer, which is the failure this file is about.
    """
    flag = _Flag()
    register_signal_handlers(flag)
    Path(pid_dir, f"rank{dist.get_rank()}.pid").write_text(str(os.getpid()))
    device = torch.device("cpu")
    while True:
        dist.all_reduce(torch.ones(64))
        if agree_stop(flag.stopped, device):
            break
    if shutdown_delay:
        time.sleep(shutdown_delay)
        Path(pid_dir, f"rank{dist.get_rank()}.done").write_text("done")


def _wait_for(path: Path, timeout: float = 60.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _signal_rank(directory, rank: int, signum: int) -> threading.Thread:
    def run():
        try:
            os.kill(_wait_for(Path(directory) / f"rank{rank}.pid"), signum)
        except (AssertionError, OSError):
            pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_killed_rank_stops_the_launcher(tmp_path):
    killer = _signal_rank(tmp_path, 1, signal.SIGKILL)
    started = time.monotonic()
    with pytest.raises(Exception) as info:
        spawn_parallel_fn(
            _loop_rank,
            world_size=2,
            backend="gloo",
            device_type="cpu",
            pid_dir=str(tmp_path),
        )
    elapsed = time.monotonic() - started
    killer.join(timeout=5)

    assert elapsed < SHUTDOWN_GRACE_SECONDS + 30, f"launcher took {elapsed:.1f}s"
    assert "SIGKILL" in str(info.value) or "process 1" in str(info.value)
    assert not any(
        _alive(int((tmp_path / f"rank{rank}.pid").read_text())) for rank in (0, 1)
    ), "a rank survived the failed run"


def test_rank_signal_stops_every_rank(tmp_path):
    killer = _signal_rank(tmp_path, 1, signal.SIGTERM)
    started = time.monotonic()
    with pytest.raises(SystemExit) as info:
        spawn_parallel_fn(
            _loop_rank,
            world_size=2,
            backend="gloo",
            device_type="cpu",
            pid_dir=str(tmp_path),
        )
    elapsed = time.monotonic() - started
    killer.join(timeout=5)

    assert info.value.code == 128 + signal.SIGTERM
    assert elapsed < SHUTDOWN_GRACE_SECONDS + 30, f"launcher took {elapsed:.1f}s"
    assert not any(
        _alive(int((tmp_path / f"rank{rank}.pid").read_text())) for rank in (0, 1)
    ), "a rank outlived the interrupted run"


def test_graceful_stop_waits_for_slow_rank_shutdown(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel_setup, "SHUTDOWN_GRACE_SECONDS", 0.1)
    killer = _signal_rank(tmp_path, 1, signal.SIGTERM)
    with pytest.raises(SystemExit) as info:
        spawn_parallel_fn(
            _loop_rank,
            world_size=2,
            backend="gloo",
            device_type="cpu",
            pid_dir=str(tmp_path),
            shutdown_delay=0.5,
        )
    killer.join(timeout=5)

    assert info.value.code == 128 + signal.SIGTERM
    assert all((tmp_path / f"rank{rank}.done").exists() for rank in (0, 1))


def test_ranks_exit_with_the_launcher(tmp_path):
    script = tmp_path / "launcher.py"
    script.write_text(
        "import sys\n"
        "from astrai.parallel.setup import spawn_parallel_fn\n"
        f"from {__name__} import _loop_rank\n"
        "if __name__ == '__main__':\n"
        "    spawn_parallel_fn(_loop_rank, world_size=2, backend='gloo',\n"
        "                      device_type='cpu', pid_dir=sys.argv[1])\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)], cwd=str(ROOT), env=env
    )
    try:
        pids = [_wait_for(tmp_path / f"rank{rank}.pid") for rank in (0, 1)]
        proc.kill()
        proc.wait(timeout=10)

        deadline = time.monotonic() + 40
        while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
            time.sleep(0.05)
        assert not any(_alive(pid) for pid in pids), "ranks outlived the launcher"
    finally:
        for pid in locals().get("pids", []):
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
