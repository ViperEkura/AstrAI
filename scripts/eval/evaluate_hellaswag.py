"""HellaSwag evaluation via log-likelihood ranking (acc + acc_norm).

Follows the lm-evaluation-harness protocol: zero-shot, query =
``activity_label + ": " + ctx_a + " " + ctx_b.capitalize()`` with the same
text normalization, ending scored as continuation ``" " + ending``.
acc = argmax raw loglikelihood; acc_norm = argmax loglikelihood / char length.

Validation split (10,042 questions) — test labels are not public.
Data cache: ./hellaswag/val.jsonl.

Known suite-level finding (2026-09-20): every checkpoint in this repo scores
chance (~25%) here and on MMLU — verified NOT to be a scoring bug (positive
control + context-swap diagnostics); the loglikelihood path currently measures
context-integration ability, which the pretrained base lacks.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict

import tqdm
from datasets import load_dataset

from astrai.evaluation import (
    EvaluationResult,
    Metric,
    check_output,
    load_jsonl,
    load_score_model,
    loglikelihood_batched,
    save_json,
    save_result,
)

HELLASWAG_HF_DATASET = "Rowan/hellaswag"


def preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def download(data_dir: str):
    path = os.path.join(data_dir, "val.jsonl")
    if os.path.exists(path):
        return
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading HellaSwag from HuggingFace ({HELLASWAG_HF_DATASET}) ...")
    ds = load_dataset(HELLASWAG_HF_DATASET, split="validation")
    with open(path, "w", encoding="utf-8") as f:
        for item in ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  saved {len(ds)} questions to {path}")


def build_item(row: Dict):
    query = preprocess(
        row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize()
    )
    endings = [preprocess(e) for e in row["endings"]]
    gold = int(row["label"])
    return {"query": query, "endings": endings, "gold": gold}


def evaluate(model, tokenizer, items, device, batch_size, details=None):
    max_model_len = model.config.max_position_embeddings
    encoded = []
    for it in items:
        ctx_ids = tokenizer.encode(it["query"])
        encoded.append(
            (
                ctx_ids,
                [
                    tokenizer.encode(" " + e, add_special_tokens=False)
                    for e in it["endings"]
                ],
            )
        )

    correct = correct_norm = total = 0
    for start in tqdm.tqdm(
        range(0, len(items), batch_size),
        total=(len(items) + batch_size - 1) // batch_size,
        desc="HellaSwag",
    ):
        batch = encoded[start : start + batch_size]
        requests = []
        for ctx_ids, cont_ids_list in batch:
            for cont_ids in cont_ids_list:
                requests.append((ctx_ids, cont_ids))
        scores = loglikelihood_batched(model, requests, device, max_model_len)
        for bi, (ctx_ids, cont_ids_list) in enumerate(batch):
            it = items[start + bi]
            lls = scores[bi * 4 : bi * 4 + 4]
            lens = [float(len(e)) for e in it["endings"]]
            if max(range(4), key=lambda i: lls[i]) == it["gold"]:
                correct += 1
            if max(range(4), key=lambda i: lls[i] / lens[i]) == it["gold"]:
                correct_norm += 1
            if details is not None:
                details.append({"scores": lls, "lengths": lens, "gold": it["gold"]})
            total += 1
    return correct, correct_norm, total


def summarize(details, metadata):
    total = len(details)
    if not total:
        raise ValueError("No HellaSwag responses")
    correct = sum(
        max(range(4), key=lambda i: row["scores"][i]) == row["gold"] for row in details
    )
    correct_norm = sum(
        max(range(4), key=lambda i: row["scores"][i] / row["lengths"][i]) == row["gold"]
        for row in details
    )
    return EvaluationResult(
        "hellaswag",
        total,
        {
            "acc": Metric(round(correct / total, 4), "ratio"),
            "acc_norm": Metric(round(correct_norm / total, 4), "ratio"),
        },
        {
            **metadata,
            "correct": correct,
            "correct_norm": correct_norm,
            "effective_counts": {"acc": total, "acc_norm": total},
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="HellaSwag evaluation (acc + acc_norm)"
    )
    parser.add_argument("--param_path", type=str, default="./params")
    parser.add_argument("--data_path", type=str, default="./hellaswag/val.jsonl")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--limit", type=int, default=0, help="Score only first N (0 = all)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Questions per batch (4 rows each)"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--score_only", help="Rescore saved details JSON without loading a model"
    )
    args = parser.parse_args()
    check_output(args.output)
    if args.score_only:
        details = json.loads(Path(args.score_only).read_text())
        result = summarize(details, {"source": args.score_only})
        if args.output:
            save_result(args.output, result)
        print(result)
        return

    download(os.path.dirname(args.data_path) or ".")

    rows = load_jsonl(args.data_path)
    items = [build_item(r) for r in rows]
    if args.limit:
        items = items[: args.limit]

    model, tokenizer = load_score_model(args.param_path, args.device, args.dtype)
    details = []
    correct, correct_norm, total = evaluate(
        model, tokenizer, items, args.device, args.batch_size, details
    )

    print(f"\n{'=' * 60}")
    print(f"  acc:      {correct / total:.2%}  ({correct}/{total})")
    print(f"  acc_norm: {correct_norm / total:.2%}  ({correct_norm}/{total})")
    print(f"{'=' * 60}")

    if args.output:
        save_json(str(Path(args.output).with_suffix(".details.json")), details)
        save_result(args.output, summarize(details, vars(args)))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
