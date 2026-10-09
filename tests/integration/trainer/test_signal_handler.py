import multiprocessing as mp
import os
import signal
import time

import pytest
import torch
import torch.optim as optim
from torch.utils.data import Dataset

from astrai.config import TrainConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.signal_handler import register_signal_handlers
from astrai.trainer import Trainer
from astrai.trainer.schedule import SchedulerFactory
from astrai.trainer.train_context import TrainContext
from tests.support.checkpoint import load_checkpoint_meta
from tests.support.models import make_tiny_config


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


def _build_model():
    config = make_tiny_config()
    device = "cpu"
    return AutoRegressiveLM(config).to(device=device)


class _ReadyCallback:
    def __init__(self, ready_file):
        self._ready_file = ready_file

    def on_train_begin(self, context):
        with open(self._ready_file, "w") as f:
            f.write("ready")
            f.flush()
            os.fsync(f.fileno())


def _inner_run(batch_per_device, ckpt_interval, ckpt_dir, ready_file):
    dataset = PicklableDataset()

    def model_fn():
        return _build_model()

    def optimizer_fn(m):
        return optim.AdamW(m.parameters(), lr=0.001)

    def scheduler_fn(optim):
        return SchedulerFactory.create(
            "cosine", optim, warmup_steps=10, lr_decay_steps=10, min_rate=0.05
        )

    train_config = TrainConfig(
        strategy="seq",
        model_fn=model_fn,
        dataset=dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
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


def test_context_stop_flag():
    ctx = TrainContext()
    assert not ctx.stop_requested
    ctx.request_stop()
    assert ctx.stop_requested


def test_register_signal_handlers():
    ctx = TrainContext()
    register_signal_handlers(ctx)
    assert not ctx.stop_requested
    os.kill(os.getpid(), signal.SIGTERM)
    assert ctx.stop_requested


@pytest.mark.slow
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_triggers_checkpoint_save(base_test_env, sig):
    exitcode = _spawn_train_and_signal(base_test_env["test_dir"], sig)
    assert exitcode == 0, f"Training process exited with code {exitcode} (expected 0)"

    meta = load_checkpoint_meta(base_test_env["test_dir"])
    assert "consumed_samples" in meta
    assert meta["consumed_samples"] >= 0
