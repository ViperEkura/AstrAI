import json
import math

import pytest

from astrai.evaluation import (
    EvaluationResult,
    Metric,
    collect_results,
    exact_match,
    load_result,
    mean,
    pass_at_k,
    perplexity,
    render_table,
    save_result,
)


def test_scores_and_token_weighted_perplexity():
    assert exact_match("A", "A")
    assert not exact_match(" A", "A")
    assert not exact_match("a", "A")
    assert mean([True, False, True]) == pytest.approx(2 / 3)
    assert mean(iter([2, 4])) == 3
    assert perplexity(2 * 1 + 8 * 3, 10) == pytest.approx(math.exp(2.6))
    assert perplexity(0, 10) == 1
    for n in range(1, 15):
        for correct in range(n + 1):
            for k in range(1, n + 1):
                expected = 1 - math.comb(n - correct, k) / math.comb(n, k)
                assert pass_at_k(n, correct, k) == pytest.approx(expected)


@pytest.mark.parametrize(
    "args",
    [(0, 0, 1), (2, 3, 1), (3, -1, 1), (3, 1, 0), (3, 1, 4), (3.0, 1, 1), (True, 0, 1)],
)
def test_pass_at_k_rejects_invalid_counts(args):
    with pytest.raises(ValueError):
        pass_at_k(*args)


def test_empty_denominators_and_nonfinite_values():
    for values in ([], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            mean(values)
    for total, count in [(1, 0), (1, -1), (-1, 2), (float("nan"), 2), (2, 1.5)]:
        with pytest.raises(ValueError):
            perplexity(total, count)


@pytest.mark.parametrize(
    "value,unit",
    [
        (float("nan"), "number"),
        (float("inf"), "number"),
        (True, "count"),
        (1.5, "count"),
        (-1, "count"),
        (1.1, "ratio"),
        (0, "percent"),
    ],
)
def test_metric_rejects_invalid_values(value, unit):
    with pytest.raises(ValueError):
        Metric(value, unit)


def test_roundtrip_units_null_and_tables(tmp_path):
    result = EvaluationResult(
        "custom",
        10,
        {
            "accuracy": Metric(0.25, "ratio"),
            "passed": Metric(2, "count"),
            "ppl": Metric(12.5, "number"),
            "unavailable": Metric(None, "ratio"),
        },
        {"prompt": "中文", "effective_counts": {"accuracy": 8}},
    )
    path = tmp_path / "custom_mid.json"
    save_result(path, result)
    assert load_result(path) == result
    assert json.loads(path.read_text())["metrics"]["passed"] == {
        "value": 2,
        "unit": "count",
    }
    collected = collect_results(tmp_path)
    assert collected == {"mid": {"custom": result}}
    table = render_table(collected)
    assert "25.00%" in table and "| 2 |" in table and "12.5" in table and "—" in table
    save_result(path, result)
    assert not list(tmp_path.glob("*.tmp"))


def test_collect_filters_details_and_arbitrary_task_names(tmp_path):
    for task, tag in [("mbpp2", "base"), ("new_task", "mid")]:
        save_result(
            tmp_path / f"{task}_{tag}.json",
            EvaluationResult(task, 1, {"acc": Metric(1, "ratio")}),
        )
    (tmp_path / "new_task_mid.details.json").write_text("[]")
    assert set(collect_results(tmp_path)) == {"base", "mid"}
    assert set(collect_results(tmp_path, benchmarks=["mbpp2"])) == {"base"}
    assert set(collect_results(tmp_path, tags=["mid"])) == {"mid"}
    assert set(collect_results([tmp_path / "new_task_mid.json"])) == {"mid"}


def test_legacy_rejected_and_never_overwritten(tmp_path):
    path = tmp_path / "mmlu_old.json"
    old = '{"_overall": {"accuracy": 0.27}}'
    path.write_text(old)
    for read in (
        lambda: load_result(path),
        lambda: collect_results(tmp_path),
        lambda: save_result(path, EvaluationResult("mmlu", 1, {})),
    ):
        with pytest.raises(
            ValueError, match="mmlu_old.json.*unsupported result format"
        ):
            read()
    assert path.read_text() == old


def test_atomic_failure_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    result = EvaluationResult("test", 1, {"count": Metric(1, "count")})
    save_result(path, result)
    original = path.read_bytes()

    def fail_replace(*args):
        raise OSError("injected replace failure")

    monkeypatch.setattr("astrai.evaluation.results.os.replace", fail_replace)
    with pytest.raises(OSError):
        save_result(path, EvaluationResult("test", 2, {}))
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_validation_on_save_and_unit_conflicts(tmp_path):
    result = EvaluationResult("task", 1, {"score": Metric(0.5, "ratio")})
    result.metrics["score"].value = float("nan")
    with pytest.raises(ValueError):
        save_result(tmp_path / "invalid.json", result)
    with pytest.raises(ValueError, match="Conflicting units"):
        render_table(
            {
                "a": {
                    "task": EvaluationResult("task", 1, {"score": Metric(1, "count")})
                },
                "b": {
                    "task": EvaluationResult("task", 1, {"score": Metric(1, "ratio")})
                },
            }
        )
