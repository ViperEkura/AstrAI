"""Tests for scheduler concurrency."""

import threading

import pytest
import torch

from tests.support.scheduler import (
    _make_real_scheduler,
)


def test_scheduler_weight_versions_are_monotonic_and_acknowledged(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        assert scheduler.policy_version == 0
        assert scheduler.update_weights(1) == 1
        assert scheduler.policy_version == 1
        assert scheduler.get_stats()["policy_version"] == 1
        assert scheduler.update_weights(1) == 1
        with pytest.raises(ValueError, match="cannot move backwards"):
            scheduler.update_weights(0)
        with pytest.raises(ValueError, match="non-negative integer"):
            scheduler.update_weights(True)
    finally:
        scheduler.stop()


def test_scheduler_applies_weight_mutation_and_version_atomically(device):
    scheduler, _tok, model = _make_real_scheduler(device)
    before = next(model.parameters()).detach().clone()

    def mutate(policy_version):
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return "updated"

    try:
        assert scheduler.apply_weight_update(1, mutate) == "updated"
        assert scheduler.policy_version == 1
        assert not torch.equal(next(model.parameters()), before)
        with pytest.raises(ValueError, match="must advance"):
            scheduler.apply_weight_update(1, mutate)

        def failed_mutation(policy_version):
            raise RuntimeError("optimizer failed")

        with pytest.raises(RuntimeError, match="optimizer failed"):
            scheduler.apply_weight_update(2, failed_mutation)
        assert scheduler.policy_version == 1

        # None derives live+1 under the lock: no read-compute-write race
        # on the current version for advance-by-one callers. The derived
        # target version is handed to the update callable.
        seen_versions = []

        def record(policy_version):
            seen_versions.append(policy_version)
            return "updated"

        assert scheduler.apply_weight_update(None, record) == "updated"
        assert scheduler.policy_version == 2
        assert seen_versions == [2]
    finally:
        scheduler.stop()


def test_scheduler_atomic_advance_survives_interleaved_publish(device):
    """A concurrent publish between reading the live version and applying
    the update must not fail ``require_advance`` (regression: callers
    computed live+1 outside the lock, a TOCTOU that raised spuriously)."""
    scheduler, _tok, _model = _make_real_scheduler(device)

    try:
        # Simulate the race directly: a version read that goes stale before
        # apply_weight_update acquires the lock. With None the scheduler
        # re-derives live+1 inside the critical section.
        stale_read = scheduler.policy_version + 1
        scheduler.update_weights(1)
        assert stale_read == 1  # now equals live -> explicit form would raise
        with pytest.raises(ValueError, match="must advance"):
            scheduler.apply_weight_update(stale_read, lambda _version: "ok")
        assert scheduler.apply_weight_update(None, lambda _version: "ok") == "ok"
        assert scheduler.policy_version == 2
    finally:
        scheduler.stop()


def test_scheduler_serializes_policy_snapshot_and_direct_update(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()
    update_finished = threading.Event()
    errors = []

    def inspect(version):
        assert version == 0
        snapshot_started.set()
        assert release_snapshot.wait(timeout=5)

    def take_snapshot():
        try:
            scheduler.with_policy_snapshot(inspect)
        except BaseException as exc:
            errors.append(exc)

    def update():
        try:
            scheduler.update_weights(1)
            update_finished.set()
        except BaseException as exc:
            errors.append(exc)

    snapshot_thread = threading.Thread(target=take_snapshot)
    update_thread = threading.Thread(target=update)
    try:
        snapshot_thread.start()
        assert snapshot_started.wait(timeout=5)
        update_thread.start()
        assert not update_finished.wait(timeout=0.1)
        release_snapshot.set()
        snapshot_thread.join(timeout=5)
        update_thread.join(timeout=5)
        assert not snapshot_thread.is_alive()
        assert not update_thread.is_alive()
        assert errors == []
        assert scheduler.policy_version == 1
    finally:
        release_snapshot.set()
        snapshot_thread.join(timeout=5)
        update_thread.join(timeout=5)
        scheduler.stop()


def test_scheduler_rejects_weight_update_with_queued_tasks(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    request_id = scheduler.add_request("queued")
    try:
        with pytest.raises(RuntimeError, match="while requests are queued"):
            scheduler.update_weights(1)
        scheduler.cancel_request(request_id)
        assert scheduler.update_weights(1) == 1
    finally:
        scheduler.stop()
