"""Async GRPO round invariants independent of the inference scheduler."""

import time
from threading import Barrier, Event, Thread

import pytest
import torch

from astrai.trainer.rollout.async_round import AsyncRoundCoordinator
from astrai.trainer.rollout.types import RawRollout, RolloutVersionError


class _Backend:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cpu")
        self.policy_version = 0

    def apply_weight_update(self, version, update):
        assert version > self.policy_version
        update(version)
        self.policy_version = version


class _Generator:
    def __init__(self, backend, barrier=None, fail=False):
        self.backend = backend
        self.barrier = barrier
        self.fail = fail

    @property
    def policy_version(self):
        return self.backend.policy_version

    def generate(self, batch):
        if self.barrier is not None:
            self.barrier.wait(timeout=3)
        if self.fail:
            raise RuntimeError("worker failed")
        prompt_ids = [int(value) for value in batch["instruction"]]
        prompts = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(-1)
        responses = (prompts + 1).unsqueeze(-1)
        return RawRollout(
            prompts=prompts,
            prompt_mask=torch.ones_like(prompts, dtype=torch.bool),
            responses=responses,
            response_mask=torch.ones_like(responses, dtype=torch.bool),
            logprobs_old=-responses.float() / 100,
            policy_version=self.backend.policy_version,
            prompt_texts=[str(value) for value in prompt_ids],
            response_texts=[[str(value + 1)] for value in prompt_ids],
            finish_reasons=[["stop"] for _ in prompt_ids],
        )


class _Reward:
    def score(self, prompts, responses):
        return torch.ones(len(prompts), 1)


def _coordinator(barrier=None, failed_worker=None, max_prompts_per_worker=1):
    source = torch.nn.Linear(2, 2)
    generators = [
        _Generator(
            _Backend(torch.nn.Linear(2, 2)),
            barrier=barrier,
            fail=index == failed_worker,
        )
        for index in range(4)
    ]
    return source, AsyncRoundCoordinator(
        source,
        generators,
        _Reward(),
        0,
        max_prompts_per_worker=max_prompts_per_worker,
    )


def test_four_workers_generate_one_ordered_round_and_receive_one_snapshot():
    source, coordinator = _coordinator(barrier=Barrier(4))
    try:
        result = coordinator.collect_round(
            coordinator.submit_round({"instruction": ["0", "1", "2", "3"]})
        )
        assert result.prompts[:, 0].tolist() == [0, 1, 2, 3]
        assert result.responses[:, 0, 0].tolist() == [1, 2, 3, 4]
        assert result.logprobs_old.shape == result.responses.shape
        torch.testing.assert_close(result.logprobs_old, -result.responses.float() / 100)
        assert result.policy_version == 0

        def update(_version):
            with torch.no_grad():
                source.weight.add_(1)

        coordinator.apply_weight_update(None, update)
        assert coordinator.policy_version == 1
        result = coordinator.collect_round(
            coordinator.submit_round({"instruction": ["4", "5", "6", "7"]})
        )
        assert result.policy_version == 1
        for generator in coordinator.generators:
            assert generator.policy_version == 1
            torch.testing.assert_close(generator.backend.model.weight, source.weight)
    finally:
        coordinator.close()


def test_worker_chunks_preserve_prompt_order_and_logprob_alignment():
    _, coordinator = _coordinator(max_prompts_per_worker=2)
    try:
        result = coordinator.collect_round(
            coordinator.submit_round(
                {"instruction": [str(index) for index in range(16)]}
            )
        )
        assert result.prompts[:, 0].tolist() == list(range(16))
        assert result.responses[:, 0, 0].tolist() == list(range(1, 17))
        assert result.logprobs_old.shape == result.responses.shape
        torch.testing.assert_close(result.logprobs_old, -result.responses.float() / 100)
    finally:
        coordinator.close()


def test_stale_or_mixed_round_is_rejected_before_training():
    _, coordinator = _coordinator()
    try:
        handle = coordinator.submit_round({"instruction": ["0", "1", "2", "3"]})
        coordinator._policy_version = 2
        with pytest.raises(RolloutVersionError, match="policy lag"):
            coordinator.collect_round(handle)
    finally:
        coordinator.close()

    _, coordinator = _coordinator()
    original = coordinator.generators[0].generate

    def wrong_version(batch):
        raw = original(batch)
        raw.policy_version = 1
        return raw

    coordinator.generators[0].generate = wrong_version
    try:
        handle = coordinator.submit_round({"instruction": ["0", "1", "2", "3"]})
        with pytest.raises(RolloutVersionError, match="mixed rollout versions"):
            coordinator.collect_round(handle)
    finally:
        coordinator.close()


def test_failed_worker_wakes_collector():
    _, coordinator = _coordinator(failed_worker=2)
    try:
        handle = coordinator.submit_round({"instruction": ["0", "1", "2", "3"]})
        started = time.perf_counter()
        with pytest.raises(RuntimeError, match="async rollout worker failed"):
            coordinator.collect_round(handle)
        assert time.perf_counter() - started < 3
    finally:
        coordinator.close()


def test_pinned_snapshot_waits_for_replica_transfer_to_finish():
    _, coordinator = _coordinator()
    try:
        publisher = coordinator.publisher
        publisher.snapshot(1)
        backend = coordinator.generators[0].backend
        transfer_started = Event()
        transfer_done = Event()
        snapshot_done = Event()
        original_apply = backend.apply_weight_update

        def delayed_apply(version, update):
            def wait_after_copy(value):
                update(value)
                transfer_started.set()
                assert transfer_done.wait(timeout=3)

            return original_apply(version, wait_after_copy)

        backend.apply_weight_update = delayed_apply
        publisher.begin_fanout(1)
        transfer = Thread(target=publisher.publish_one, args=(0, 1))
        transfer.start()
        assert transfer_started.wait(timeout=3)
        assert backend.policy_version == 0

        def next_snapshot():
            publisher.snapshot(2)
            snapshot_done.set()

        snapshot = Thread(target=next_snapshot)
        snapshot.start()
        assert not snapshot_done.wait(timeout=0.05)
        transfer_done.set()
        transfer.join(timeout=3)
        snapshot.join(timeout=3)
        assert not transfer.is_alive() and not snapshot.is_alive()
        assert backend.policy_version == 1
        assert publisher.version == 2
    finally:
        coordinator.close()


def test_weight_publish_during_generation_keeps_round_version_stable():
    source, coordinator = _coordinator()
    entered = Barrier(5)
    release = Event()
    try:
        for generator in coordinator.generators:
            original = generator.generate
            first_call = [True]

            def held_generate(batch, original=original, first_call=first_call):
                if first_call[0]:
                    first_call[0] = False
                    entered.wait(timeout=3)
                    assert release.wait(timeout=3)
                return original(batch)

            generator.generate = held_generate

        handle = coordinator.submit_round({"instruction": ["0", "1", "2", "3"]})
        entered.wait(timeout=3)
        coordinator.apply_weight_update(
            None, lambda _version: source.weight.data.add_(1)
        )
        assert all(
            generator.policy_version == 0 for generator in coordinator.generators
        )
        release.set()
        assert coordinator.collect_round(handle).policy_version == 0
        next_round = coordinator.submit_round({"instruction": ["4", "5", "6", "7"]})
        assert coordinator.collect_round(next_round).policy_version == 1
    finally:
        release.set()
        coordinator.close()
