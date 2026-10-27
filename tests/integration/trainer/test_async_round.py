"""Process protocol invariants independent of the inference scheduler."""

import time
from functools import partial
from threading import Thread
from types import SimpleNamespace

import pytest
import torch

from astrai.trainer.callbacks.checkpoint import CheckpointCallback
from astrai.trainer.rollout.async_round import AsyncRoundCoordinator, _response_seeds
from astrai.trainer.rollout.protocol import (
    GenerationResult,
    MessageKind,
    RolloutMessage,
    RolloutProtocolError,
    WorkerReady,
    recv_message,
    send_message,
)
from astrai.trainer.rollout.types import (
    RawRollout,
    ResolvedAsyncRolloutConfig,
    RolloutVersionError,
    SamplingParams,
)


def _fake_worker(conn, spec):
    device = spec.device
    policy_version = spec.policy_version
    generated = 0
    try:
        send_message(
            conn,
            RolloutMessage(
                MessageKind.MODEL_READY,
                request_id=0,
                policy_version=policy_version,
            ),
        )
        initial = recv_message(conn)
        assert initial.kind == MessageKind.WEIGHT_SYNC
        send_message(
            conn,
            RolloutMessage(
                MessageKind.WEIGHT_SYNC_ACK,
                request_id=initial.request_id,
                policy_version=policy_version,
            ),
        )
        send_message(
            conn,
            RolloutMessage(
                MessageKind.READY,
                request_id=0,
                policy_version=policy_version,
                payload=WorkerReady(False, 0),
            ),
        )
        while True:
            message = recv_message(conn)
            if message.kind == MessageKind.STOP:
                return
            if message.kind == MessageKind.WEIGHT_SYNC:
                if device == "slow_ack":
                    time.sleep(0.3)
                policy_version = message.policy_version
                send_message(
                    conn,
                    RolloutMessage(
                        MessageKind.WEIGHT_SYNC_ACK,
                        request_id=message.request_id,
                        policy_version=policy_version,
                    ),
                )
                continue
            round_id, version, chunk = (
                message.round_id,
                message.policy_version,
                message.payload.batch,
            )
            if device == "hang":
                time.sleep(999)
            if device == "fail":
                send_message(
                    conn,
                    RolloutMessage(
                        MessageKind.ERROR,
                        request_id=message.request_id,
                        payload="worker failed",
                    ),
                )
                return
            time.sleep(0.15)
            prompt_ids = [int(value) for value in chunk["instruction"]]
            prompts = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(-1)
            responses = (prompts + 1).unsqueeze(-1)
            raw = RawRollout(
                prompts=prompts,
                prompt_mask=torch.ones_like(prompts, dtype=torch.bool),
                responses=responses,
                response_mask=torch.ones_like(responses, dtype=torch.bool),
                logprobs_old=-responses.float() / 100,
                policy_version=version
                + (device == "stale" or (device == "stale_once" and generated == 0)),
                prompt_texts=[str(value) for value in prompt_ids],
                response_texts=[[str(value + 1)] for value in prompt_ids],
                finish_reasons=[["stop"] for _ in prompt_ids],
            )
            send_message(
                conn,
                RolloutMessage(
                    MessageKind.RESULT,
                    request_id=message.request_id + (device == "wrong_request"),
                    round_id=round_id + (device == "wrong_round"),
                    policy_version=version + (device == "wrong_version"),
                    payload=GenerationResult(raw, 0.15, 0),
                ),
            )
            generated += 1
    finally:
        conn.close()


class _FakeWeightChannel:
    def __init__(self, source, version):
        self.layout = [
            (name, tuple(tensor.shape), str(tensor.dtype).split(".")[-1])
            for name, tensor in source.state_dict().items()
        ]
        self.version = version
        self._pending = set()
        self.fanout_seconds = 0.0
        self.broadcast_calls = 0

    def prepare_rendezvous(self, _world_size, _timeout_s):
        return 0

    def connect(self, **_kwargs):
        pass

    def broadcast(self):
        self.broadcast_calls += 1

    def mark_committed(self, version):
        if self._pending:
            raise RuntimeError("previous weight broadcast has not been acknowledged")
        self.version = version

    def begin_fanout(self, indices):
        self._pending = set(indices)

    def acknowledge(self, index):
        self._pending.remove(index)

    def close(self, force=False):
        pass


class _Reward:
    def score(self, prompts, responses):
        return torch.ones(len(prompts), 1)


def _resolved_config(
    devices=None,
    timeout=3.0,
    max_prompts_per_worker=1,
    learner_rank=None,
    sample_cursor=0,
):
    return ResolvedAsyncRolloutConfig(
        learner_device="cpu",
        devices=tuple(devices or ["a", "b", "c", "d"]),
        params=SamplingParams(max_tokens=1, group_size=1),
        max_seq_len=8,
        max_prompts_per_worker=max_prompts_per_worker,
        startup_timeout_s=300.0,
        generation_timeout_s=timeout,
        weight_timeout_s=timeout,
        max_policy_lag=1,
        random_seed=3407,
        sample_cursor=sample_cursor,
        learner_rank=learner_rank,
    )


def _coordinator(**kwargs):
    source = torch.nn.Linear(2, 2)
    return source, AsyncRoundCoordinator(
        config=_resolved_config(**kwargs),
        source=source,
        model_fn=partial(torch.nn.Linear, 2, 2),
        param_path="unused",
        reward_model=_Reward(),
        policy_version=0,
        model_dtype=torch.float32,
        worker_target=_fake_worker,
        weight_channel=_FakeWeightChannel(source, 0),
    )


def _collect(coordinator, values):
    return coordinator.collect_round(
        coordinator.submit_round({"instruction": [str(value) for value in values]})
    )


def test_four_processes_generate_ordered_round_and_receive_one_broadcast():
    source, coordinator = _coordinator()
    try:
        started = time.perf_counter()
        result = _collect(coordinator, range(4))
        assert time.perf_counter() - started < 0.6
        assert result.prompts[:, 0].tolist() == [0, 1, 2, 3]
        assert result.responses[:, 0, 0].tolist() == [1, 2, 3, 4]
        torch.testing.assert_close(result.logprobs_old, -result.responses.float() / 100)
        assert result.policy_version == 0
        assert coordinator.weights.broadcast_calls == 1

        def update(_version):
            with torch.no_grad():
                source.weight.add_(1)

        coordinator.apply_weight_update(None, update)
        assert coordinator.policy_version == 1
        assert _collect(coordinator, range(4, 8)).policy_version == 1
        assert coordinator._worker_versions == [1] * 4
        assert coordinator.weights.broadcast_calls == 2
        assert not coordinator.weights._pending
    finally:
        coordinator.close()


def test_worker_chunks_preserve_prompt_order_and_logprob_alignment():
    _, coordinator = _coordinator(max_prompts_per_worker=2)
    try:
        result = _collect(coordinator, range(16))
        assert result.prompts[:, 0].tolist() == list(range(16))
        assert result.responses[:, 0, 0].tolist() == list(range(1, 17))
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

    _, coordinator = _coordinator(devices=["stale_once", "b", "c", "d"])
    try:
        with pytest.raises(RolloutVersionError, match="mixed rollout versions"):
            _collect(coordinator, range(4))
        assert _collect(coordinator, range(4)).policy_version == 0
    finally:
        coordinator.close()

    _, coordinator = _coordinator(devices=["stale", "b", "c", "d"])
    try:
        with pytest.raises(RolloutVersionError, match="mixed rollout versions"):
            _collect(coordinator, range(4))
    finally:
        coordinator.close()


@pytest.mark.parametrize(
    ("bad_worker", "reason"),
    [
        ("wrong_request", "wrong request ID"),
        ("wrong_round", "stale round ID"),
        ("wrong_version", "wrong policy version"),
    ],
)
def test_protocol_rejects_mismatched_response_envelope(bad_worker, reason):
    _, coordinator = _coordinator(devices=[bad_worker, "b", "c", "d"])
    with pytest.raises(RolloutProtocolError, match=reason):
        _collect(coordinator, range(4))
    assert all(not process.is_alive() for process in coordinator._processes)


def test_weight_sync_ack_has_no_payload():
    ack = RolloutMessage(MessageKind.WEIGHT_SYNC_ACK, request_id=7, policy_version=3)
    ack.expect(MessageKind.WEIGHT_SYNC_ACK, 7, policy_version=3)
    with pytest.raises(RolloutProtocolError, match="wrong policy version"):
        ack.expect(MessageKind.WEIGHT_SYNC_ACK, 7, policy_version=4)
    with pytest.raises(RolloutProtocolError, match="must not carry a payload"):
        RolloutMessage(
            MessageKind.WEIGHT_SYNC_ACK,
            request_id=7,
            policy_version=3,
            payload={"peak_gpu_memory": 0},
        ).expect(MessageKind.WEIGHT_SYNC_ACK, 7, policy_version=3)


def test_failed_worker_wakes_collector_and_kills_hung_peer():
    _, coordinator = _coordinator(devices=["hang", "fail", "c", "d"])
    handle = coordinator.submit_round({"instruction": ["0", "1", "2", "3"]})
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="worker 1 failed"):
        coordinator.collect_round(handle)
    assert time.perf_counter() - started < 3
    assert all(not process.is_alive() for process in coordinator._processes)


def test_timeout_kills_all_workers():
    _, coordinator = _coordinator(devices=["hang", "b", "c", "d"], timeout=0.4)
    handle = coordinator.submit_round({"instruction": ["0", "1", "2", "3"]})
    with pytest.raises(TimeoutError, match="timed out"):
        coordinator.collect_round(handle)
    assert all(not process.is_alive() for process in coordinator._processes)


def test_committed_version_cannot_advance_before_ack():
    source, coordinator = _coordinator(devices=["slow_ack", "b", "c", "d"])
    try:
        coordinator.apply_weight_update(
            None, lambda _version: source.weight.data.add_(1)
        )
        errors = []

        def submit():
            try:
                _collect(coordinator, range(4))
            except BaseException as exc:
                errors.append(exc)

        thread = Thread(target=submit)
        thread.start()
        deadline = time.monotonic() + 2
        while not coordinator.weights._pending and time.monotonic() < deadline:
            time.sleep(0.01)
        assert coordinator.weights._pending
        with pytest.raises(RuntimeError, match="not been acknowledged"):
            coordinator.weights.mark_committed(2)
        thread.join(timeout=3)
        assert not thread.is_alive() and not errors
        assert coordinator.weights.version == 1
    finally:
        coordinator.close()


def test_weight_publication_during_generation_keeps_round_version():
    source, coordinator = _coordinator()
    try:
        handle = coordinator.submit_round({"instruction": ["0", "1", "2", "3"]})
        coordinator.apply_weight_update(
            None, lambda _version: source.weight.data.add_(1)
        )
        assert coordinator.collect_round(handle).policy_version == 0
        assert _collect(coordinator, range(4, 8)).policy_version == 1
    finally:
        coordinator.close()


def test_version_commit_failure_marks_optimizer_commit():
    source, coordinator = _coordinator()
    try:

        def fail_version_commit(_version):
            raise RuntimeError("version commit failed")

        coordinator.weights.mark_committed = fail_version_commit
        before = source.weight.detach().clone()
        with pytest.raises(RuntimeError, match="version commit failed"):
            coordinator.apply_weight_update(
                None, lambda _version: source.weight.data.add_(1)
            )
        assert coordinator.policy_version == 1
        torch.testing.assert_close(source.weight, before + 1)
    finally:
        coordinator.close()


def test_model_factory_must_be_pickleable():
    with pytest.raises(ValueError, match="model_fn must be pickleable"):
        AsyncRoundCoordinator(
            config=_resolved_config(),
            source=torch.nn.Linear(2, 2),
            model_fn=lambda: torch.nn.Linear(2, 2),
            param_path="unused",
            reward_model=_Reward(),
            policy_version=0,
            model_dtype=torch.float32,
            worker_target=_fake_worker,
        )


def test_uncommitted_step_writes_no_checkpoint():
    """A step that never committed leaves nothing new to persist."""
    callback = CheckpointCallback("unused", interval=1)
    callback.last_ckpt_step = 2
    callback.last_ckpt_cursor = 10
    callback._saved = True
    saved = []
    callback._save_checkpoint = lambda _context: saved.append(True)
    context = SimpleNamespace(optimizer_step=2, consumed_samples=10)
    callback.on_shutdown(context)
    callback.on_train_end(context)
    assert saved == []


def test_prefetch_resume_cursor_and_retry_keep_response_seeds():
    _, pool = _coordinator()
    try:
        first = pool.submit_round({"instruction": [str(i) for i in range(4)]})
        pool.collect_round(first)
        next_round = pool.submit_round({"instruction": ["4", "5"]})
        pool.collect_round(next_round)
        retry = pool.submit_round({"instruction": ["4", "5"]}, seeds=next_round.seeds)
        pool.collect_round(retry)
        assert retry.seeds == next_round.seeds
        assert pool._sample_cursor == 6
    finally:
        pool.close()

    assert next_round.seeds == _response_seeds(pool.config.random_seed, 4, 2, 1)


def test_two_learners_keep_local_order_and_distinct_response_seeds():
    sources = []
    coordinators = []
    try:
        for rank, devices in enumerate((["r0a", "r0b"], ["r1a", "r1b"])):
            source, coordinator = _coordinator(
                devices=devices, learner_rank=rank, sample_cursor=6
            )
            sources.append(source)
            coordinators.append(coordinator)
        handles = [
            coordinator.submit_round({"instruction": ["10", "11", "12", "13"]})
            for coordinator in coordinators
        ]
        for rank, (coordinator, handle) in enumerate(zip(coordinators, handles)):
            assert handle.jobs == {0: [0, 2], 1: [1, 3]}
            assert handle.seeds == _response_seeds(3407, 6, 4, 1, rank)
            result = coordinator.collect_round(handle)
            assert result.prompts[:, 0].tolist() == [10, 11, 12, 13]
            assert result.policy_version == 0
            assert coordinator.weights.broadcast_calls == 1
        assert handles[0].seeds != handles[1].seeds
        for source, coordinator in zip(sources, coordinators):
            coordinator.apply_weight_update(
                None, lambda _version, source=source: source.weight.data.add_(1)
            )
            assert _collect(coordinator, [20, 21]).policy_version == 1
            assert coordinator.weights.broadcast_calls == 2
    finally:
        for coordinator in coordinators:
            coordinator.close()
