"""Task-independent scores over prepared values."""

import math
from numbers import Integral
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def exact_match(prediction: Any, reference: Any) -> bool:
    return prediction == reference


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty sequence")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Values must be finite")
    return math.fsum(values) / len(values)


def pass_at_k(n: int, correct: int, k: int) -> float:
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (n, correct, k)):
        raise ValueError("n, correct and k must be integers")
    if not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("Expected 0 <= correct <= n and 1 <= k <= n")
    if n - correct < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - correct + 1, n + 1))


def perplexity(nll_sum: float, token_count: int) -> float:
    if (
        isinstance(token_count, bool)
        or not isinstance(token_count, Integral)
        or token_count <= 0
    ):
        raise ValueError("token_count must be a positive integer")
    if not math.isfinite(nll_sum) or nll_sum < 0:
        raise ValueError("nll_sum must be finite and nonnegative")
    return math.exp(nll_sum / token_count)


def score_results(
    results: Sequence[Tuple[str, int, int]],
    k_values: Tuple[int, ...],
) -> Dict:
    """Unbiased pass@k per problem; per-problem k entries are None when
    n < k (e.g. after deduplication); the summary averages only computed ks."""
    scores: Dict[int, List[float]] = {k: [] for k in k_values}
    output: Dict = {}
    for task_id, n, passed in results:
        entry = {"task_id": task_id, "n": n, "passed": passed}
        for k in k_values:
            if k <= n:
                pk = round(pass_at_k(n, passed, k), 4)
                entry[f"pass@{k}"] = pk
                scores[k].append(pk)
            else:
                entry[f"pass@{k}"] = None
        output[str(task_id)] = entry

    summary = {}
    for k in k_values:
        vals = scores[k]
        summary[f"pass@{k}"] = round(float(np.mean(vals)), 4) if vals else None
    output["_summary"] = summary
    return output
