import pytest

from astrai.bench import find_checkpoints, watch_checkpoints


def make_ckpt(parent, name, complete=True):
    d = parent / name
    d.mkdir()
    (d / "meta.json").write_text("{}", encoding="utf-8")
    if complete:
        (d / "model.safetensors").write_text("x", encoding="utf-8")
    return d


def recorder():
    """plan_jobs stand-in recording (ckpt, tag, benchmarks) calls."""

    class Rec:
        def __init__(self):
            self.calls = []

        def plan(self, ckpt, tag, benchmarks, **kwargs):
            self.calls.append((ckpt, tag, tuple(benchmarks)))
            return [], []

    return Rec()


RESULTS = {"results_dir": "r", "logs_dir": "l"}  # nonexistent dirs: fine, no outputs


def test_find_checkpoints_complete_and_ordered(tmp_path):
    make_ckpt(tmp_path, "epoch_2_step_100")
    make_ckpt(tmp_path, "epoch_1_step_4385")
    make_ckpt(tmp_path, "epoch_1_step_200", complete=False)
    (tmp_path / "epoch_1_step_300").mkdir()
    (tmp_path / "metric.jsonl").write_text("", encoding="utf-8")
    found = find_checkpoints(str(tmp_path))
    assert [name for name, _ in found] == ["epoch_1_step_4385", "epoch_2_step_100"]


def test_find_checkpoints_empty_dir(tmp_path):
    assert find_checkpoints(str(tmp_path)) == []


def test_watch_once_processes_each_checkpoint_once(tmp_path, monkeypatch):
    make_ckpt(tmp_path, "epoch_1_step_100")
    make_ckpt(tmp_path, "epoch_1_step_200")
    rec = recorder()
    monkeypatch.setattr("astrai.bench.plan_jobs", rec.plan)
    watch_checkpoints(str(tmp_path), ["ifeval"], gpus=[0], once=True, **RESULTS)
    assert [c[1] for c in rec.calls] == ["epoch_1_step_100", "epoch_1_step_200"]


def test_watch_retries_busy_gpus_then_keeps_polling(tmp_path, monkeypatch):
    make_ckpt(tmp_path, "epoch_1_step_100")
    state = {"n": 0}

    def flaky(ckpt, tag, benchmarks, **kwargs):
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("need 1 free GPUs, found 0")
        return [], []

    monkeypatch.setattr("astrai.bench.plan_jobs", flaky)
    polls = {"n": 0}

    def fake_sleep(_s):
        polls["n"] += 1
        if polls["n"] >= 5:
            raise KeyboardInterrupt

    monkeypatch.setattr("astrai.bench.time.sleep", fake_sleep)
    with pytest.raises(KeyboardInterrupt):
        watch_checkpoints(str(tmp_path), ["ifeval"], gpus=[0], **RESULTS)
    assert state["n"] == 3


def test_watch_once_skips_when_gpus_busy(tmp_path, monkeypatch):
    make_ckpt(tmp_path, "epoch_1_step_100")

    def busy(ckpt, tag, benchmarks, **kwargs):
        raise RuntimeError("need 1 free GPUs, found 0")

    monkeypatch.setattr("astrai.bench.plan_jobs", busy)
    watch_checkpoints(str(tmp_path), ["ifeval"], gpus=[0], once=True, **RESULTS)


def test_watch_splits_benchmarks_into_waves(tmp_path, monkeypatch):
    make_ckpt(tmp_path, "epoch_1_step_100")
    rec = recorder()
    monkeypatch.setattr("astrai.bench.plan_jobs", rec.plan)
    watch_checkpoints(
        str(tmp_path),
        ["ifeval", "humaneval", "mbpp2"],
        gpus=[0, 1],
        once=True,
        **RESULTS,
    )
    assert [c[2] for c in rec.calls] == [("ifeval", "humaneval"), ("mbpp2",)]


def test_watch_tag_prefix_and_every_filter(tmp_path, monkeypatch):
    make_ckpt(tmp_path, "epoch_1_step_100")
    make_ckpt(tmp_path, "epoch_1_step_150")
    make_ckpt(tmp_path, "epoch_1_step_200")
    rec = recorder()
    monkeypatch.setattr("astrai.bench.plan_jobs", rec.plan)
    watch_checkpoints(
        str(tmp_path), ["ifeval"], gpus=[0], tag="mix4", every=100, once=True, **RESULTS
    )
    assert [c[1] for c in rec.calls] == [
        "mix4_epoch_1_step_100",
        "mix4_epoch_1_step_200",
    ]
