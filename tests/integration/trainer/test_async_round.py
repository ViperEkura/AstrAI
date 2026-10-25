"""Process protocol invariants independent of the inference scheduler."""

import time
from functools import partial
from threading import Event, Thread
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
from astrai.trainer.rollout.types import RawRollout, RolloutVersionError, SamplingParams


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


def _coordinator(
    devices=None,
    timeout=3.0,
    max_prompts_per_worker=1,
    learner_rank=None,
    sample_cursor=0,
):
    source = torch.nn.Linear(2, 2)
    return source, AsyncRoundCoordinator(
        source=source,
        model_fn=partial(torch.nn.Linear, 2, 2),
        param_path="unused",
        devices=devices or ["a", "b", "c", "d"],
        params=SamplingParams(max_tokens=1, group_size=1),
        reward_model=_Reward(),
        policy_version=0,
        max_batch_size=1,
        max_seq_len=8,
        model_dtype=torch.float32,
        worker_timeout_s=timeout,
        max_prompts_per_worker=max_prompts_per_worker,
        worker_target=_fake_worker,
        weight_channel=_FakeWeightChannel(source, 0),
        learner_rank=learner_rank,
        sample_cursor=sample_cursor,
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
            source=torch.nn.Linear(2, 2),
            model_fn=lambda: torch.nn.Linear(2, 2),
            param_path="unused",
            devices=["a", "b", "c", "d"],
            params=SamplingParams(max_tokens=1, group_size=1),
            reward_model=_Reward(),
            policy_version=0,
            max_batch_size=1,
            max_seq_len=8,
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

    assert next_round.seeds == _response_seeds(pool.random_seed, 4, 2, 1)


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


def test_iter_rounds_dispatches_next_generation_before_scoring(monkeypatch):
    _, pool = _coordinator(devices=["a"])
    events = []
    submitted = []
    submit = pool.submit_round

    def record_submit(batch, **kwargs):
        handle = submit(batch, **kwargs)
        submitted.append(handle)
        events.append(("submit", int(batch["instruction"][0])))
        return handle

    def score(prompts, responses):
        index = int(prompts[0])
        events.append(("score", index))
        if index < 2:
            assert pool._active_round is submitted[index + 1]
            with pytest.raises(RuntimeError, match="only one rollout round"):
                pool.submit_round({"instruction": ["99"]})
        else:
            assert pool._active_round is None
        return torch.tensor([[index + 0.5]])

    monkeypatch.setattr(pool, "submit_round", record_submit)
    monkeypatch.setattr(pool.reward_model, "score", score)
    try:
        batches = [{"instruction": [str(index)]} for index in range(3)]
        results = []
        for rollout in pool.iter_rounds(batches, lambda: False):
            index = int(rollout.prompts[0, 0])
            events.append(("yield", index))
            results.append(rollout)
        assert events == [
            ("submit", 0),
            ("submit", 1),
            ("score", 0),
            ("yield", 0),
            ("submit", 2),
            ("score", 1),
            ("yield", 1),
            ("score", 2),
            ("yield", 2),
        ]
        assert [result.responses.item() for result in results] == [1, 2, 3]
        assert [result.rewards.item() for result in results] == [0.5, 1.5, 2.5]
        assert pool._sample_cursor == 3
        assert pool._active_round is None
    finally:
        pool.close()


@pytest.mark.parametrize("error_type", [RuntimeError, RolloutVersionError])
def test_iter_rounds_reward_failure_closes_prefetched_workers(monkeypatch, error_type):
    _, pool = _coordinator(devices=["a"])
    failure = error_type("reward failed after prefetch")
    score_calls = []

    def score(prompts, responses):
        score_calls.append(prompts)
        assert pool._active_round is not None
        assert pool._active_round.round_id == 2
        raise failure

    monkeypatch.setattr(pool.reward_model, "score", score)
    try:
        rounds = pool.iter_rounds(
            [{"instruction": ["0"]}, {"instruction": ["1"]}], lambda: False
        )
        with pytest.raises(error_type, match="reward failed after prefetch") as error:
            next(rounds)
        assert error.value is failure
        assert score_calls == [["0"]]
        assert pool._round_id == pool._sample_cursor == 2
        assert pool._closed and pool._active_round is None
        assert all(not process.is_alive() for process in pool._processes)
    finally:
        pool.close(force=True)


def test_iter_rounds_stale_prefetch_retries_same_seeds_once(monkeypatch):
    _, pool = _coordinator(devices=["a"])
    submitted = []
    scored = []
    submit = pool.submit_round

    def record_submit(batch, **kwargs):
        handle = submit(batch, **kwargs)
        submitted.append((batch["instruction"][0], handle, pool._sample_cursor))
        return handle

    def score(prompts, responses):
        scored.append(prompts[0])
        return torch.ones(len(prompts), 1)

    monkeypatch.setattr(pool, "submit_round", record_submit)
    monkeypatch.setattr(pool.reward_model, "score", score)
    try:
        rounds = pool.iter_rounds(
            [{"instruction": [str(index)]} for index in range(3)], lambda: False
        )
        assert next(rounds).policy_version == 0
        pool.apply_weight_update(None, lambda _version: None)
        pool.apply_weight_update(None, lambda _version: None)
        retried = next(rounds)
        assert retried.prompts.item() == 1
        assert retried.policy_version == 2
        assert [prompt for prompt, _, _ in submitted] == ["0", "1", "1", "2"]
        assert [handle.version for _, handle, _ in submitted] == [0, 0, 2, 2]
        assert [cursor for _, _, cursor in submitted] == [1, 2, 2, 3]
        assert submitted[1][1].seeds == submitted[2][1].seeds
        assert pool._active_round is submitted[3][1]
        assert scored == ["0", "1"]
        assert next(rounds).prompts.item() == 2
        with pytest.raises(StopIteration):
            next(rounds)
        assert scored == ["0", "1", "2"]
        assert pool._sample_cursor == 3
        assert pool._active_round is None
    finally:
        pool.close(force=True)


def test_iter_rounds_persistently_stale_result_does_not_retry_forever(monkeypatch):
    _, pool = _coordinator(devices=["stale"])
    submitted = []
    submit = pool.submit_round

    def record_submit(batch, **kwargs):
        assert len(submitted) < 2, "a rejected retry must not submit a third time"
        handle = submit(batch, **kwargs)
        submitted.append(handle)
        return handle

    monkeypatch.setattr(pool, "submit_round", record_submit)
    monkeypatch.setattr(
        pool.reward_model,
        "score",
        lambda *_: pytest.fail("a stale rollout must never be scored"),
    )
    try:
        rounds = pool.iter_rounds([{"instruction": ["0"]}], lambda: False)
        with pytest.raises(RolloutVersionError, match="mixed rollout versions"):
            next(rounds)
        assert len(submitted) == 2
        assert submitted[0].seeds == submitted[1].seeds
        assert pool._sample_cursor == 1
        assert pool._closed and pool._active_round is None
        assert all(not process.is_alive() for process in pool._processes)
    finally:
        pool.close(force=True)


def test_iter_rounds_initial_stop_does_not_read_or_submit():
    _, pool = _coordinator(devices=["a"])

    def batches():
        pytest.fail("a stopped iterator must not consume a batch")
        yield {"instruction": ["0"]}

    try:
        assert list(pool.iter_rounds(batches(), lambda: True)) == []
        assert pool._round_id == pool._sample_cursor == 0
        assert pool._active_round is None
    finally:
        pool.close()


def test_iter_rounds_stop_during_collect_finishes_current_without_prefetch(
    monkeypatch,
):
    _, pool = _coordinator(devices=["a"])
    stopped = Event()
    collect_raw = pool._collect_raw
    scored = []

    def stop_after_collect(handle):
        raw = collect_raw(handle)
        stopped.set()
        return raw

    def batches():
        yield {"instruction": ["0"]}
        pytest.fail("stop during collection must not consume the next batch")

    def score(prompts, responses):
        scored.append(prompts)
        assert pool._active_round is None
        return torch.ones(len(prompts), 1)

    monkeypatch.setattr(pool, "_collect_raw", stop_after_collect)
    monkeypatch.setattr(pool.reward_model, "score", score)
    try:
        results = list(pool.iter_rounds(batches(), stopped.is_set))
        assert len(results) == 1
        assert results[0].prompts.item() == 0
        assert results[0].rewards.item() == 1
        assert scored == [["0"]]
        assert pool._round_id == pool._sample_cursor == 1
        assert pool._active_round is None
    finally:
        pool.close()


def test_iter_rounds_stop_after_yield_leaves_prefetch_for_owner_to_close():
    _, pool = _coordinator(devices=["a"])
    stopped = Event()
    try:
        rounds = pool.iter_rounds(
            [{"instruction": [str(index)]} for index in range(3)], stopped.is_set
        )
        assert next(rounds).prompts.item() == 0
        active = pool._active_round
        assert active is not None
        stopped.set()
        with pytest.raises(StopIteration):
            next(rounds)
        assert pool._active_round is active
        assert not pool._closed
        assert pool._round_id == pool._sample_cursor == 2
    finally:
        pool.close()
    assert pool._active_round is None
    assert all(not process.is_alive() for process in pool._processes)


@pytest.mark.parametrize("pending", [False, True], ids=["idle", "pending-generation"])
def test_default_close_orders_worker_and_weight_channel_cleanup(monkeypatch, pending):
    _, pool = _coordinator(devices=["hang"])
    channel_closes = []

    def close_channel(force=False):
        channel_closes.append(force)
        if pending:
            assert force, "busy workers cannot cooperate in weight-channel shutdown"
            assert all(not process.is_alive() for process in pool._processes)

    monkeypatch.setattr(pool.weights, "close", close_channel)
    try:
        if pending:
            pool.submit_round({"instruction": ["0"]})
        pool.close()
        assert channel_closes == [pending]
        assert all(not process.is_alive() for process in pool._processes)
        pool.close()
        assert channel_closes == [pending], "closing twice must not repeat cleanup"
    finally:
        pool.close(force=True)


def test_closing_round_iterator_closes_prefetched_workers():
    _, pool = _coordinator(devices=["a"])
    try:
        rounds = pool.iter_rounds(
            [{"instruction": ["0"]}, {"instruction": ["1"]}], lambda: False
        )
        next(rounds)
        assert pool._active_round is not None
        rounds.close()
        assert pool._closed and pool._active_round is None
        assert all(not process.is_alive() for process in pool._processes)
    finally:
        pool.close(force=True)


@pytest.mark.parametrize("missing_reply", [False, True])
def test_expired_result_deadline_accepts_only_already_ready_replies(missing_reply):
    devices = ["a", "hang"] if missing_reply else ["a"]
    _, pool = _coordinator(devices=devices)
    try:
        handle = pool.submit_round(
            {"instruction": [str(index) for index in range(len(devices))]}
        )
        assert pool._pipes[0].poll(10), "fake worker did not deliver its result"
        # Simulate delayed collection (e.g. a slow reward call) after the
        # result is ready, without depending on elapsed-time performance.
        handle.started = time.monotonic() - pool.worker_timeout_s - 1
        if missing_reply:
            with pytest.raises(TimeoutError, match="timed out"):
                pool.collect_round(handle)
            assert pool._closed
            assert all(not process.is_alive() for process in pool._processes)
        else:
            result = pool.collect_round(handle)
            assert result.prompts.item() == 0
            assert result.rewards.item() == 1
        assert pool._active_round is None
    finally:
        pool.close(force=True)
