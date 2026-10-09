"""Deterministic CPU gates for execution identity, lifecycle and resource ownership."""

import threading
from unittest.mock import MagicMock, patch

from astrai.inference.core.engine_core import EngineCore
from tests.support.core_contracts import scheduler as scheduler


def test_stop_cannot_clear_a_restarted_successor_after_join():
    sched = MagicMock()
    core = EngineCore(sched, threading.RLock())
    old, new = MagicMock(), MagicMock()
    old.is_alive.return_value = True
    new.is_alive.return_value = True
    core.loop_thread = old
    core._shutdown = MagicMock()

    def join(timeout):
        old.is_alive.return_value = False
        with patch(
            "astrai.inference.core.engine_core.threading.Thread", return_value=new
        ):
            core.start()  # exact stop-join / final-cleanup race window

    old.join.side_effect = join
    assert core.stop() is False
    assert core.loop_thread is new
    assert not core.stop_event.is_set()
    core._shutdown.assert_not_called()


def test_score_holds_policy_lock_through_forward_and_cleanup(scheduler):
    sched, _events = scheduler
    entered, release, published = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    errors = []

    def score(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return [-1.0]

    def do_score():
        try:
            assert sched.score_ids([[10]], [[11]]) == [-1.0]
        except BaseException as exc:
            errors.append(exc)

    def update():
        try:
            sched.apply_weight_update(None, lambda _: published.set())
        except BaseException as exc:
            errors.append(exc)

    with patch.object(sched._executor, "execute_score", side_effect=score):
        score_thread = threading.Thread(target=do_score)
        update_thread = threading.Thread(target=update)
        score_thread.start()
        assert entered.wait(3)
        update_thread.start()
        assert not published.wait(0.05)
        release.set()
        score_thread.join(3)
        update_thread.join(3)
    assert not errors and published.is_set() and sched.policy_version == 1
    assert not score_thread.is_alive() and not update_thread.is_alive()


def test_start_waits_for_weight_mutation(scheduler):
    sched, _events = scheduler
    entered, release = threading.Event(), threading.Event()

    def mutate(_):
        entered.set()
        assert release.wait(3)

    update = threading.Thread(target=lambda: sched.apply_weight_update(None, mutate))
    start = threading.Thread(target=sched.start)
    update.start()
    assert entered.wait(3)
    start.start()
    assert sched._loop_thread is None
    release.set()
    update.join(3)
    start.join(3)
    assert not update.is_alive() and not start.is_alive()
    assert sched.policy_version == 1
    sched.stop()
