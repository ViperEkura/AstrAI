"""ROUGE evaluation (manual implementation, no external deps).

Computes ROUGE-1, ROUGE-2, ROUGE-L precision, recall, and F1.

Usage::

    # Batch evaluation from JSONL (each line: {"reference": ..., "candidate": ...})
    python scripts/eval/evaluate_rouge.py --data_path preds.jsonl --output results.json

    # As a library
    from scripts.eval.evaluate_rouge import compute_rouge
    scores = compute_rouge("the cat sat on the mat", "the cat sat")
"""

import argparse
import json
from collections import Counter
from typing import Dict, List


def _tokenize(text: str) -> List[str]:
    return text.split()


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def _lcs(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        xi = x[i - 1]
        dpi = dp[i]
        dpi_1 = dp[i - 1]
        for j in range(1, n + 1):
            if xi == y[j - 1]:
                dpi[j] = dpi_1[j - 1] + 1
            else:
                dpi[j] = dpi_1[j] if dpi_1[j] > dpi[j - 1] else dpi[j - 1]
    return dp[m][n]


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rouge_n(ref_tokens: List[str], cand_tokens: List[str], n: int) -> Dict[str, float]:
    ref_ngrams = _ngrams(ref_tokens, n)
    cand_ngrams = _ngrams(cand_tokens, n)

    overlap = sum((cand_ngrams & ref_ngrams).values())
    cand_total = sum(cand_ngrams.values())
    ref_total = sum(ref_ngrams.values())

    precision = overlap / cand_total if cand_total > 0 else 0.0
    recall = overlap / ref_total if ref_total > 0 else 0.0
    f1 = _f1(precision, recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def _rouge_l(ref_tokens: List[str], cand_tokens: List[str]) -> Dict[str, float]:
    lcs_len = _lcs(ref_tokens, cand_tokens)
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)

    recall = lcs_len / ref_len if ref_len > 0 else 0.0
    precision = lcs_len / cand_len if cand_len > 0 else 0.0
    f1 = _f1(precision, recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_rouge(
    reference: str, candidate: str, n: int = 2
) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-N (1..n) and ROUGE-L scores.

    Returns::

        {
            "rouge-1": {"precision": ..., "recall": ..., "f1": ...},
            "rouge-2": {"precision": ..., "recall": ..., "f1": ...},
            "rouge-l": {"precision": ..., "recall": ..., "f1": ...},
        }
    """
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)

    results = {}
    for i in range(1, n + 1):
        results[f"rouge-{i}"] = _rouge_n(ref_tokens, cand_tokens, i)
    results["rouge-l"] = _rouge_l(ref_tokens, cand_tokens)
    return results


def evaluate_file(data_path: str) -> Dict:
    with open(data_path, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    agg = {
        k: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for k in ("rouge-1", "rouge-2", "rouge-l")
    }
    per_item = []

    for item in pairs:
        ref = item["reference"]
        cand = item["candidate"]
        scores = compute_rouge(ref, cand)
        per_item.append({**item, "scores": scores})
        for k, v in scores.items():
            agg[k]["precision"] += v["precision"]
            agg[k]["recall"] += v["recall"]
            agg[k]["f1"] += v["f1"]

    n = len(pairs)
    for k in agg:
        agg[k] = {m: v / n for m, v in agg[k].items()}

    return {"num_samples": n, "aggregate": agg, "per_item": per_item}


def main():
    parser = argparse.ArgumentParser(description="ROUGE evaluation")
    parser.add_argument(
        "--data_path", required=True, help="JSONL with reference/candidate per line"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    results = evaluate_file(args.data_path)
    agg = results["aggregate"]

    print(f"Samples: {results['num_samples']}")
    print()
    for metric in ("rouge-1", "rouge-2", "rouge-l"):
        s = agg[metric]
        print(
            f"  {metric:8s}  P={s['precision']:.4f}  R={s['recall']:.4f}  F1={s['f1']:.4f}"
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
