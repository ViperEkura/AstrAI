"""Freeze deduplicated task splits from hash-verified local public Parquet."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from astrai.tokenize import AutoTokenizer
from examples.rl_reward.data import load_splits, sha256_file
from examples.rl_reward.rewards import numeric_answer
from examples.rl_reward.run import configure_prompt

SPLIT_VERSION = "semantic-sha256-order-v1"


def normalize(rows, task, seen):
    records, duplicates = [], 0
    for row in rows:
        if task == "countdown":
            numbers, target = row["nums"], row["target"]
            if (
                not isinstance(numbers, list)
                or not 2 <= len(numbers) <= 32
                or any(type(n) is not int or not 0 < n <= 1000000 for n in numbers)
                or type(target) is not int
            ):
                raise ValueError("invalid source Countdown record")
            canonical = json.dumps([sorted(numbers), target])
            record = {"numbers": sorted(numbers), "target": target}
        else:
            question, answer = row["question"], row["answer"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError("invalid source GSM8K question")
            if not isinstance(answer, str) or numeric_answer(answer) is None:
                raise ValueError("invalid source GSM8K answer")
            canonical = " ".join(question.split())
            record = {"question": canonical, "answer": answer}
        identifier = hashlib.sha256(canonical.encode()).hexdigest()
        label = target if task == "countdown" else numeric_answer(answer)
        if identifier in seen:
            if seen[identifier] != label:
                raise ValueError("conflicting labels for a duplicate source task")
            duplicates += 1
            continue
        seen[identifier] = label
        records.append({"id": f"{task}-{identifier}", **record})
    return records, duplicates


def freeze_splits(train_rows, test_rows, task, train_size, dev_size, test_size, seed):
    if task not in {"countdown", "gsm8k"}:
        raise ValueError("unsupported task")
    if any(type(n) is not int or n < 1 for n in (train_size, dev_size, test_size)):
        raise ValueError("all split sizes must be positive integers")
    if train_size % 64:
        raise ValueError("formal train size must fill global batches of 64 prompts")
    seen = {}
    train, train_duplicates = normalize(train_rows, task, seen)
    test, test_duplicates = normalize(test_rows, task, seen)

    def order(record):
        return hashlib.sha256(
            f"{SPLIT_VERSION}:{seed}:{record['id']}".encode()
        ).digest()

    train.sort(key=order)
    test.sort(key=order)
    needed = train_size + dev_size + (test_size if task == "countdown" else 0)
    if len(train) < needed or (task == "gsm8k" and len(test) < test_size):
        raise ValueError("insufficient unique source records for frozen split sizes")
    splits = {
        "train": train[:train_size],
        "dev": train[train_size : train_size + dev_size],
        "test": train[train_size + dev_size : needed]
        if task == "countdown"
        else test[:test_size],
    }
    counts = {
        "train_source_duplicates": train_duplicates,
        "test_source_duplicates_or_train_collisions": test_duplicates,
        "unique_train_source": len(train),
        "unique_test_source": len(test),
        "unselected_unique": len(train) + len(test) - sum(map(len, splits.values())),
    }
    return splits, counts


def verified_rows(path, expected_sha256):
    if sha256_file(path) != expected_sha256:
        raise ValueError("source file hash differs from the frozen asset")
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("countdown", "gsm8k"), required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--test-source", type=Path)
    parser.add_argument("--test-sha256")
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--dev-size", type=int, required=True)
    parser.add_argument("--test-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-cap", type=int, default=1024)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.dataset_revision) != 40 or any(
        c not in "0123456789abcdef" for c in args.dataset_revision
    ):
        raise ValueError("dataset revision must be an immutable commit")
    if args.task == "gsm8k" and (args.test_source is None or args.test_sha256 is None):
        raise ValueError("GSM8K requires its original test source")
    if args.output.exists():
        raise ValueError("frozen data output already exists")
    train_rows = verified_rows(args.train_source, args.train_sha256)
    test_rows = (
        verified_rows(args.test_source, args.test_sha256) if args.test_source else []
    )
    splits, counts = freeze_splits(
        train_rows,
        test_rows,
        args.task,
        args.train_size,
        args.dev_size,
        args.test_size,
        args.seed,
    )
    args.output.mkdir(parents=True, mode=0o700)
    paths = {}
    for split, records in splits.items():
        path = args.output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )
        paths[split] = path
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    configure_prompt(tokenizer, SimpleNamespace(enable_thinking=args.enable_thinking))
    checked, _ = load_splits(paths, args.task, tokenizer, args.prompt_cap)
    manifest = {
        "state": "FROZEN_DATA_PROMPTS_VALIDATED",
        "split_version": SPLIT_VERSION,
        "dataset_repo": args.dataset_repo,
        "dataset_revision": args.dataset_revision,
        "task": args.task,
        "seed": args.seed,
        "prompt_cap": args.prompt_cap,
        "enable_thinking": args.enable_thinking,
        "counts": counts,
        "source_sha256": {"train": args.train_sha256, "test": args.test_sha256},
        "splits": {
            split: {
                "sha256": sha256_file(path),
                "records": len(checked[split]),
                "ids": [r["id"] for r in checked[split]],
            }
            for split, path in paths.items()
        },
        "tokenizer_sha256": {
            name: sha256_file(args.model_dir / name)
            for name in ("tokenizer.json", "tokenizer_config.json")
        },
    }
    (args.output / "data-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "state": manifest["state"],
                "task": args.task,
                "split_sizes": {split: len(rows) for split, rows in checked.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
