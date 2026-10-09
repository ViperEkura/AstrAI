"""Unit tests for astrai.bench result extraction and suite planning."""

import pytest

from astrai.bench import (
    collect_results,
    extract_metrics,
    parse_results_name,
    plan_jobs,
    render_table,
)

HUMANEVAL = {
    "_summary": {"pass@1": 0.5, "pass@10": 0.9},
    "HumanEval/0": {"n": 20, "passed": 10, "pass@1": 0.5, "pass@10": 1.0},
    "HumanEval/1": {"n": 20, "passed": 0, "pass@1": 0.0, "pass@10": 0.0},
    "HumanEval/2": {"n": 20, "passed": 12, "pass@1": 0.6, "pass@10": 1.0},
}

IFEVAL = {
    "0": {
        "num_constraints": 2,
        "num_passed": 1,
        "constraints": [
            {"instruction_id": "detectable_format:json_format", "passed": True},
            {
                "instruction_id": "detectable_format:number_bullet_lists",
                "passed": False,
            },
        ],
    },
    "1": {
        "num_constraints": 2,
        "num_passed": 2,
        "constraints": [
            {"instruction_id": "detectable_format:json_format", "passed": True},
            {"instruction_id": "punctuation:no_comma", "passed": True},
        ],
    },
}


def test_extract_humaneval_excludes_summary():
    out = extract_metrics(HUMANEVAL, "humaneval")
    assert out["pass@1"] == 0.5
    assert out["pass@10"] == 0.9
    assert out["zero_pass"] == 1
    assert out["strong_pass"] == 2


def test_extract_ifeval_subitems():
    out = extract_metrics(IFEVAL, "ifeval")
    assert out["acc"] == pytest.approx(0.75)
    assert out["json_format"] == 1.0
    assert out["num_bullet_lists"] == 0.0


def test_extract_ifeval_prefers_official_aggregate():
    # the script's _summary excludes unsupported constraints from the
    # denominator, so it wins over the per-problem sum
    payload = {
        **IFEVAL,
        "_summary": {"overall_accuracy": 0.3897, "total_constraints": 793},
    }
    out = extract_metrics(payload, "ifeval")
    assert out["acc"] == pytest.approx(0.3897)
    assert out["json_format"] == 1.0


def test_extract_mmlu_and_hellaswag():
    assert extract_metrics({"_overall": {"accuracy": 0.27}}, "mmlu") == {"acc": 0.27}
    hs = extract_metrics({"_summary": {"acc": 0.24, "acc_norm": 0.25}}, "hellaswag")
    assert hs == {"acc": 0.24, "acc_norm": 0.25}


def test_extract_unknown_benchmark_raises():
    with pytest.raises(KeyError):
        extract_metrics({}, "nonexistent")


def test_parse_results_name_prefix_priority():
    assert parse_results_name("mbpp2_mix3.json") == ("mbpp2", "mix3")
    assert parse_results_name("mbpp_mix3.json") == ("mbpp", "mix3")
    assert parse_results_name("hellaswag_base.json") == ("hellaswag", "base")


def test_parse_results_name_rejects_non_results():
    assert parse_results_name("mbpp2_mix3_completions.json") is None
    assert parse_results_name("mmlu.json") is None  # no tag
    assert parse_results_name("stray.json") is None
    assert parse_results_name("notes.txt") is None


def test_render_table_percent_and_counts():
    collected = {
        "mix3": {"mmlu": {"acc": 0.27}, "mbpp": {"pass@1": 0.0427, "zero_pass": 392}},
        "base": {"mmlu": {"acc": None}, "mbpp": {"pass@1": 0.0, "zero_pass": 500}},
    }
    table = render_table(collected)
    assert "| metric | base | mix3 |" in table
    assert "| MMLU acc | — | 27.00% |" in table
    assert "| MBPP pass@1 | 0.00% | 4.27% |" in table
    assert "| MBPP zero_pass | 500 | 392 |" in table


def test_collect_results_reads_and_filters(tmp_path):
    (tmp_path / "mbpp2_t1.json").write_text(
        '{"_summary": {"pass@1": 0.1, "pass@10": 0.2}}'
    )
    (tmp_path / "mbpp2_t1_completions.json").write_text("{}")
    (tmp_path / "stray.json").write_text("{}")
    (tmp_path / "mmlu_t2.json").write_text('{"_overall": {"accuracy": 0.3}}')

    all_tags = collect_results(str(tmp_path))
    assert set(all_tags) == {"t1", "t2"}
    assert "mbpp2" in all_tags["t1"]
    assert "stray" not in str(all_tags)

    only_mmlu = collect_results(str(tmp_path), benchmarks=["mmlu"])
    assert set(only_mmlu) == {"t2"}

    only_t1 = collect_results(str(tmp_path), tags=["t1"])
    assert set(only_t1) == {"t1"}


def _fake_gpus(free):
    return lambda n, preferred=None: (list(preferred or []) + free)[:n]


def test_plan_jobs_smoke_skips_ifeval_and_pins_gpus(monkeypatch):
    monkeypatch.setattr("astrai.bench.pick_free_gpus", _fake_gpus([1, 2, 3]))
    jobs, skipped = plan_jobs(
        "./ckpt", "t1", ["mmlu", "hellaswag", "ifeval"], gpus=None, smoke=True
    )
    assert skipped == ["ifeval"]
    by_bench = {job.bench: job for job in jobs}
    assert "--subjects abstract_algebra" in by_bench["mmlu"].cmd
    assert by_bench["mmlu"].outfile == "results/mmlu_t1.json"
    assert by_bench["mmlu"].cmd.startswith("CUDA_VISIBLE_DEVICES=1 ")
    assert by_bench["hellaswag"].cmd.startswith("CUDA_VISIBLE_DEVICES=2 ")
    assert "--limit 64" in by_bench["hellaswag"].cmd


def test_plan_jobs_standard_includes_ifeval(monkeypatch):
    monkeypatch.setattr("astrai.bench.pick_free_gpus", _fake_gpus([5]))
    jobs, skipped = plan_jobs("./ckpt", "t2", ["ifeval"], gpus=None, smoke=False)
    assert skipped == []
    assert jobs[0].cmd.startswith("CUDA_VISIBLE_DEVICES=5 ")
    assert "--temperature 0.1" in jobs[0].cmd


def test_plan_jobs_raises_without_free_gpus(monkeypatch):
    monkeypatch.setattr("astrai.bench.pick_free_gpus", _fake_gpus([]))
    with pytest.raises(RuntimeError, match="free GPUs"):
        plan_jobs("./ckpt", "t1", ["mmlu"])
