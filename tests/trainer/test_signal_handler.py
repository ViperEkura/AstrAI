import multiprocessing as mp
import os
import signal
import time
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

import astrai.trainer.session as session_module
from astrai.config import ComponentSpec, TrainConfig
from astrai.trainer import Trainer
from astrai.trainer.context.metric import MetricHistory
from astrai.trainer.rollout import worker as worker_module
from astrai.trainer.session import TrainSession
from tests.support.checkpoint import load_checkpoint_meta
from tests.support.models import TINY_CONFIG


class PicklableDataset(Dataset):
    def __init__(self, length=200, max_length=64, vocab_size=1000):
        self.length = length
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
            "target_ids": torch.randint(0, self.vocab_size, (self.max_length,)),
        }


class _ReadyCallback:
    def __init__(self, ready_file):
        self._ready_file = ready_file

    def on_train_begin(self, context):
        with open(self._ready_file, "w") as f:
            f.write("ready")
            f.flush()
            os.fsync(f.fileno())


def _inner_run(batch_per_device, ckpt_interval, ckpt_dir, ready_file):
    train_config = TrainConfig(
        strategy="seq",
        model=ComponentSpec(type="autoregressive_lm", params={"config": TINY_CONFIG}),
        dataset=ComponentSpec(
            type="tests.trainer.test_signal_handler:PicklableDataset",
            params={},
        ),
        optimizer=ComponentSpec(type="torch.optim:AdamW", params={"lr": 0.001}),
        scheduler=ComponentSpec(
            type="cosine",
            params={"warmup_steps": 10, "lr_decay_steps": 10, "min_rate": 0.05},
        ),
        ckpt_dir=ckpt_dir,
        n_epoch=99999,
        batch_per_device=batch_per_device,
        ckpt_interval=ckpt_interval,
        grad_accum_steps=1,
        random_seed=42,
        device_type="cpu",
    )

    trainer = Trainer(train_config)
    trainer.callbacks.insert(0, _ReadyCallback(ready_file))
    trainer.train()


def _spawn_train_and_signal(ckpt_dir, sig, timeout=120):
    ready_file = os.path.join(ckpt_dir, "ready.txt")

    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_inner_run,
        args=(2, 1000, ckpt_dir, ready_file),
    )
    p.start()

    deadline = time.time() + 10
    while time.time() < deadline:
        if os.path.exists(ready_file):
            with open(ready_file) as f:
                if f.read().strip() == "ready":
                    break
        if not p.is_alive():
            break
        time.sleep(0.5)

    assert p.is_alive(), "Training process died before becoming ready"

    os.kill(p.pid, sig)
    p.join(timeout=timeout)

    if p.is_alive():
        p.kill()
        p.join(timeout=5)

    assert not p.is_alive()
    return p.exitcode


@pytest.mark.slow
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_triggers_checkpoint_save(base_test_env, sig):
    exitcode = _spawn_train_and_signal(base_test_env["test_dir"], sig)
    assert exitcode == 0, f"Training process exited with code {exitcode} (expected 0)"

    meta = load_checkpoint_meta(base_test_env["test_dir"])
    assert "consumed_samples" in meta
    assert meta["consumed_samples"] >= 0


def test_train_session_cleans_up_when_start_callback_fails(monkeypatch):
    events = []
    monkeypatch.setattr(
        session_module, "register_signal_handlers", lambda _: events.append("register")
    )
    monkeypatch.setattr(
        session_module,
        "unregister_signal_handlers",
        lambda: events.append("unregister"),
    )
    rollout = SimpleNamespace(close=lambda: events.append("close"))
    context = SimpleNamespace(
        rollout=rollout,
        executor=SimpleNamespace(use_distributed=False),
        stop_requested=False,
        run_shutdown=lambda: events.append("shutdown_hooks"),
        metric_history=MetricHistory(),
    )

    def callback(name, _):
        events.append(name)
        if name == "on_train_begin":
            context.metric_history.stage({"type": "step"})
            raise RuntimeError("startup failed")
        if name == "on_shutdown":
            context.metric_history.commit(1, 2)
            assert not context.metric_history.snapshot().records

    with pytest.raises(RuntimeError, match="startup failed"):
        with TrainSession(lambda: context, callback):
            pass
    assert events == [
        "register",
        "on_train_begin",
        "shutdown_hooks",
        "on_shutdown",
        "close",
        "on_train_end",
        "unregister",
    ]


def test_rollout_worker_follows_its_parent_first(monkeypatch):
    """A replica sets its parent-death signal before touching any state."""

    class ParentDied(RuntimeError):
        pass

    monkeypatch.setattr(
        worker_module, "die_with_parent", lambda: (_ for _ in ()).throw(ParentDied())
    )
    with pytest.raises(ParentDied):
        worker_module.run_rollout_worker(None, None)
