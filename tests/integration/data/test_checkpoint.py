import json
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from astrai.parallel.setup import get_rank, spawn_parallel_fn
from astrai.serialization import Checkpoint
from astrai.serialization import checkpoint as checkpoint_module


def test_single_process():
    model = torch.nn.Linear(10, 5)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(3):
        for iteration in range(10):
            x = torch.randn(32, 10)
            loss = model(x).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

    checkpoint = Checkpoint(
        state_dict=model.state_dict(), epoch=3, consumed_samples=120
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint.save(tmpdir)

        loaded_checkpoint = Checkpoint.load(tmpdir)

        assert loaded_checkpoint.epoch == 3
        assert loaded_checkpoint.consumed_samples == 120


def test_checkpoint_with_extra():
    model = torch.nn.Linear(10, 5)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    optimizer.step()

    extra = {
        "optimizer": optimizer.state_dict(),
        "scheduler": {"last_epoch": 5},
    }
    checkpoint = Checkpoint(
        state_dict=model.state_dict(),
        epoch=1,
        consumed_samples=40,
        extra=extra,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint.save(tmpdir)

        assert os.path.exists(os.path.join(tmpdir, "optimizer.pt"))
        assert os.path.exists(os.path.join(tmpdir, "scheduler.pt"))

        loaded = Checkpoint.load(tmpdir)
        assert loaded.extra["scheduler"]["last_epoch"] == 5
        assert "state" in loaded.extra["optimizer"]


def test_checkpoint_is_atomically_published_with_manifest(tmp_path, monkeypatch):
    target = tmp_path / "epoch_1_step_9"
    checkpoint = Checkpoint(
        state_dict={"weight": torch.arange(4)},
        epoch=1,
        consumed_samples=36,
        meta={"optimizer_step": 9, "policy_version": 3},
        config={"hidden_size": 4},
    )
    original_save = checkpoint_module.save_safetensors

    def observe_staging(state_dict, path):
        assert not target.exists()
        assert path.parent.name.startswith(f".{target.name}.tmp-")
        original_save(state_dict, path)

    monkeypatch.setattr(checkpoint_module, "save_safetensors", observe_staging)
    checkpoint.save(target)

    assert target.is_dir()
    assert not list(tmp_path.glob(f".{target.name}.tmp-*"))
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["format_version"] == 1
    assert manifest["optimizer_step"] == 9
    assert manifest["policy_version"] == 3
    assert manifest["tensors"] == ["weight"]
    assert set(manifest["files"]) == {
        "config.json",
        "meta.json",
        "model.safetensors",
    }
    assert len(manifest["files"]["model.safetensors"]["sha256"]) == 64
    assert Checkpoint.load(target, verify_checksums=True).consumed_samples == 36


def test_checkpoint_failure_never_publishes_partial_directory(tmp_path, monkeypatch):
    target = tmp_path / "epoch_0_step_1"
    checkpoint = Checkpoint(state_dict={"weight": torch.ones(2)})

    def fail_save(*args, **kwargs):
        assert not target.exists()
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(checkpoint_module, "save_safetensors", fail_save)
    with pytest.raises(RuntimeError, match="injected write failure"):
        checkpoint.save(target)

    assert not target.exists()
    assert not list(tmp_path.glob(f".{target.name}.tmp-*"))


def test_checkpoint_republish_atomically_replaces_published_directory(tmp_path):
    target = tmp_path / "epoch_0_step_1"
    Checkpoint(state_dict={"weight": torch.ones(2)}).save(target)
    replacement = Checkpoint(
        state_dict={"weight": torch.zeros(3)},
        meta={"optimizer_step": 1},
        config={"hidden_size": 3},
    )
    replacement.save(target)

    assert target.is_dir()
    assert not list(tmp_path.glob(f".{target.name}.tmp-*"))
    assert not list(tmp_path.glob(f".{target.name}.retired-*"))
    loaded = Checkpoint.load(target, verify_checksums=True)
    torch.testing.assert_close(loaded.state_dict["weight"], torch.zeros(3))


def test_checkpoint_checksum_verification_detects_same_size_corruption(tmp_path):
    target = tmp_path / "epoch_0_step_1"
    Checkpoint(state_dict={"weight": torch.ones(2)}).save(target)
    meta_path = target / "meta.json"
    corrupted = meta_path.read_bytes()
    replacement = b"X" if corrupted[0:1] != b"X" else b"Y"
    meta_path.write_bytes(replacement + corrupted[1:])

    with pytest.raises(ValueError, match="checksum mismatch: meta.json"):
        Checkpoint.load(target, verify_checksums=True)


def test_checkpoint_load_accepts_legacy_directory_without_manifest(tmp_path):
    target = tmp_path / "legacy"
    target.mkdir()
    checkpoint_module.save_json(
        {"epoch": 2, "consumed_samples": 20}, target / "meta.json"
    )
    checkpoint_module.save_json({"hidden_size": 2}, target / "config.json")
    checkpoint_module.save_safetensors(
        {"weight": torch.ones(2)}, target / "model.safetensors"
    )

    loaded = Checkpoint.load(target, verify_checksums=True)

    assert loaded.epoch == 2
    assert loaded.consumed_samples == 20


def simple_training():
    model = torch.nn.Linear(10, 5)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(2):
        for iteration in range(5):
            x = torch.randn(16, 10)
            loss = model(x).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

    checkpoint = Checkpoint(
        state_dict=model.state_dict(),
        epoch=2,
        consumed_samples=40,
    )

    rank = get_rank()

    if rank == 0:
        shared_dir = tempfile.mkdtemp()
        checkpoint.save(shared_dir)
    else:
        shared_dir = None

    if dist.is_initialized():
        dir_list = [shared_dir]
        dist.broadcast_object_list(dir_list, src=0)
        shared_dir = dir_list[0]

    loaded = Checkpoint.load(shared_dir)
    assert loaded.epoch == 2


def test_multi_process():
    spawn_parallel_fn(simple_training, world_size=2, backend="gloo")
