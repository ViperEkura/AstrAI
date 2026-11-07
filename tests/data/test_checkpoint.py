import json
import os
import tempfile
import time

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
    assert manifest["format_version"] == 2
    assert manifest["optimizer_step"] == 9
    assert manifest["policy_version"] == 3
    assert manifest["tensors"] == ["weight"]
    assert set(manifest["files"]) == {
        "config.json",
        "meta.json",
        "model.safetensors",
    }
    # v2 publishes without hashing: the rename is the completeness guarantee.
    assert manifest["files"]["model.safetensors"]["size"] > 0
    assert "sha256" not in manifest["files"]["model.safetensors"]
    assert Checkpoint.load(target, verify_checksums=True).consumed_samples == 36
    assert Checkpoint.verify(target)["checksums"] == "absent"


def test_staging_artifact_failure_never_publishes_checkpoint(tmp_path):
    target = tmp_path / "epoch_0_step_1"
    checkpoint = Checkpoint(state_dict={"weight": torch.ones(2)})

    def fail_metric_snapshot(staging_path):
        (staging_path / "metric.jsonl").write_text("partial\n")
        raise RuntimeError("metric snapshot failed")

    with pytest.raises(RuntimeError, match="metric snapshot failed"):
        checkpoint.save(target, prepare_staging=fail_metric_snapshot)

    assert not target.exists()
    assert not list(tmp_path.glob(f".{target.name}.tmp-*"))


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


def test_republishing_a_step_keeps_the_published_directory(tmp_path):
    """Publication is write-once: the same step is never rewritten."""
    target = tmp_path / "epoch_0_step_1"
    assert Checkpoint(state_dict={"weight": torch.ones(2)}).save(target) is True
    published = (target / "model.safetensors").stat().st_mtime_ns

    replacement = Checkpoint(
        state_dict={"weight": torch.zeros(3)},
        meta={"optimizer_step": 1},
        config={"hidden_size": 3},
    )
    assert replacement.save(target) is False

    assert not list(tmp_path.glob(f".{target.name}.tmp-*"))
    loaded = Checkpoint.load(target)
    torch.testing.assert_close(loaded.state_dict["weight"], torch.ones(2))
    assert (target / "model.safetensors").stat().st_mtime_ns == published


def test_verify_detects_same_size_corruption_when_checksums_exist(tmp_path):
    """A v1-style manifest still gets its hashes checked; v2 does not hash."""
    target = tmp_path / "epoch_0_step_1"
    Checkpoint(state_dict={"weight": torch.ones(2)}).save(target)
    manifest = json.loads((target / "manifest.json").read_text())
    for name, descriptor in manifest["files"].items():
        descriptor["sha256"] = checkpoint_module._sha256(target / name)
    (target / "manifest.json").write_text(json.dumps(manifest))

    meta_path = target / "meta.json"
    corrupted = meta_path.read_bytes()
    meta_path.write_bytes((b"X" if corrupted[0:1] != b"X" else b"Y") + corrupted[1:])

    with pytest.raises(ValueError, match="checksum mismatch: meta.json"):
        Checkpoint.verify(target)


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


def test_stale_staging_is_cleaned_before_the_next_save(tmp_path):
    """A killed save leaves a staging dir; the next save must reclaim it."""
    target = tmp_path / "epoch_0_step_9"
    stale = tmp_path / ".epoch_0_step_9.tmp-abandoned"
    stale.mkdir()
    (stale / "model.safetensors").write_bytes(b"partial")
    old = time.time() - 2 * checkpoint_module.STALE_STAGING_SECONDS
    os.utime(stale, (old, old))

    live = tmp_path / ".epoch_0_step_9.tmp-live"
    live.mkdir()

    model = torch.nn.Linear(4, 2)
    Checkpoint(state_dict=model.state_dict(), epoch=0, consumed_samples=9).save(
        str(target)
    )

    assert target.is_dir()
    assert not stale.exists()
    assert live.exists()


def test_republish_keeps_files_the_save_does_not_produce(tmp_path):
    """Re-writing the same step must not drop the files other writers left."""
    target = tmp_path / "epoch_0_step_3"
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    checkpoint = Checkpoint(state_dict=model.state_dict(), epoch=0, consumed_samples=3)
    checkpoint.save(str(target))

    (target / "tokenizer.json").write_text('{"version": 1}')
    (target / "notes.json").write_text('{"loss": 1.0}\n')

    checkpoint.save(str(target))

    assert json.loads((target / "tokenizer.json").read_text()) == {"version": 1}
    assert (target / "notes.json").read_text().strip() == '{"loss": 1.0}'
