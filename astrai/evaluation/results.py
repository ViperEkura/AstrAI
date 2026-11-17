"""Typed summaries and task-independent comparison tables."""

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


@dataclass
class Metric:
    value: Optional[float]
    unit: str

    def __post_init__(self):
        if self.unit not in ("ratio", "count", "number"):
            raise ValueError(f"Unsupported metric unit: {self.unit!r}")
        if self.value is None:
            return
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(self.value)
        ):
            raise ValueError("Metric value must be a finite number or None")
        if self.unit == "ratio" and not 0 <= self.value <= 1:
            raise ValueError("Ratio must be between 0 and 1")
        if self.unit == "count" and (self.value < 0 or int(self.value) != self.value):
            raise ValueError("Count must be a nonnegative integer")


@dataclass
class EvaluationResult:
    task: str
    num_samples: int
    metrics: Dict[str, Metric]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a nonempty string")
        if type(self.num_samples) is not int or self.num_samples < 0:
            raise ValueError("num_samples must be a nonnegative integer")
        if not isinstance(self.metrics, dict) or not isinstance(self.metadata, dict):
            raise ValueError("metrics and metadata must be objects")
        for name, metric in self.metrics.items():
            if not isinstance(name, str) or not name or not isinstance(metric, Metric):
                raise ValueError("metrics must map nonempty names to Metric objects")
            metric.__post_init__()
        json.dumps(self.metadata, allow_nan=False)


def save_result(path: Union[str, Path], result: EvaluationResult) -> None:
    """Validate and atomically replace a summary; never overwrite legacy data."""
    result.__post_init__()
    content = (
        json.dumps(asdict(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    path = Path(path)
    if path.exists():
        load_result(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as out:
            temp = Path(out.name)
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def load_result(path: Union[str, Path]) -> EvaluationResult:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {
            "task",
            "num_samples",
            "metrics",
            "metadata",
        }:
            raise ValueError(
                "unsupported result format; expected EvaluationResult (legacy files are not converted)"
            )
        return EvaluationResult(
            task=payload["task"],
            num_samples=payload["num_samples"],
            metrics={
                name: Metric(**metric) for name, metric in payload["metrics"].items()
            },
            metadata=payload["metadata"],
        )
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"{path}: {error}") from error


def collect_results(
    paths: Union[str, Path, Iterable[Union[str, Path]]],
    benchmarks: Optional[Iterable[str]] = None,
    tags: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, EvaluationResult]]:
    """Collect summary files; filenames supply tags, contents supply tasks."""
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    collected = {}
    for path in paths:
        path = Path(path)
        if path.stem.endswith(
            ("_completions", "_tested", "_details", ".details", ".responses")
        ):
            continue
        result = load_result(path)
        tag = path.stem.removeprefix(result.task + "_")
        if benchmarks is not None and result.task not in benchmarks:
            continue
        if tags is not None and tag not in tags:
            continue
        row = collected.setdefault(tag, {})
        if result.task in row:
            raise ValueError(f"{path}: duplicate task/tag {result.task!r}/{tag!r}")
        row[result.task] = result
    return collected


def render_table(collected: Dict[str, Dict[str, EvaluationResult]]) -> str:
    columns = {}
    for tasks in collected.values():
        for task, result in tasks.items():
            for name, metric in result.metrics.items():
                key = (task, name)
                if key in columns and columns[key] != metric.unit:
                    raise ValueError(f"Conflicting units for {task}/{name}")
                columns[key] = metric.unit
    keys = sorted(columns)

    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [
        ["Run"] + [escape(f"{task}/{name}") for task, name in keys],
        ["---"] * (len(keys) + 1),
    ]
    for tag, tasks in sorted(collected.items()):
        row = [escape(tag)]
        for task, name in keys:
            metric = tasks[task].metrics.get(name) if task in tasks else None
            if metric is None or metric.value is None:
                value = "—"
            elif metric.unit == "ratio":
                value = f"{metric.value:.2%}"
            elif metric.unit == "count":
                value = str(int(metric.value))
            else:
                value = f"{metric.value:.6g}"
            row.append(value)
        rows.append(row)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Union[str, Path], data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def check_output(path: Optional[Union[str, Path]]) -> None:
    """Refuse legacy summary paths before writing any associated details."""
    if path and Path(path).exists():
        load_result(path)


def summarize_code(
    task: str, scored: Dict, metadata: Dict[str, Any]
) -> EvaluationResult:
    problems = [v for k, v in scored.items() if not k.startswith("_")]
    metrics = {key: Metric(value, "ratio") for key, value in scored["_summary"].items()}
    metrics.update(
        {
            "zero_pass": Metric(
                sum(
                    v.get("pass@1", 0) == 0 and v.get("pass@10", 0) == 0
                    for v in problems
                ),
                "count",
            ),
            "strong_pass": Metric(
                sum((v.get("pass@1") or 0) >= 0.5 for v in problems), "count"
            ),
        }
    )
    return EvaluationResult(
        task,
        len(problems),
        metrics,
        {
            **metadata,
            "effective_counts": {
                key: sum(v.get(key) is not None for v in problems)
                for key in scored["_summary"]
            },
            "aggregation": "Mean of eligible per-problem pass@k, rounded to 4 decimals before and after aggregation",
        },
    )
