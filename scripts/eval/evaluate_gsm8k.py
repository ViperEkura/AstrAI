"""GSM8K answer accuracy: raw 5-shot or explicit chat prompting."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from astrai.evaluation import (
    EvaluationResult,
    Metric,
    check_output,
    exact_match,
    load_jsonl,
    mean,
    save_result,
)
from astrai.inference import build_engine


def extract_answer(text: str, strict: bool = True) -> Optional[str]:
    pattern = r"#### (\-?[0-9\.\,]+)" if strict else r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    match = matches[0] if strict else matches[-1]
    value = next(group for group in match.groups() if group is not None)
    return value.replace(",", "").replace("$", "").removesuffix(".")


def build_prompt(question, examples, tokenizer, prompt_format):
    if prompt_format == "raw":
        return "\n\n".join(
            [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in examples]
            + [f"Question: {question}\nAnswer:"]
        )
    turns = []
    for example in examples:
        turns.extend(
            [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ]
        )
    turns.append(
        {
            "role": "user",
            "content": question
            + "\nSolve step by step and end with #### followed by the final numeric answer.",
        }
    )
    return tokenizer.apply_chat_template(
        turns, tokenize=False, add_generation_prompt=True
    )


def score_response(record):
    target = extract_answer(record["reference"])
    if target is None:
        raise ValueError("GSM8K reference missing a #### numeric answer")
    text = re.split(r"Question:|</s>|<\|im_end\|>", record["response"], maxsplit=1)[0]
    predictions = {
        name: extract_answer(text, strict)
        for name, strict in [("strict_match", True), ("flexible_extract", False)]
    }
    return {
        **record,
        "target": target,
        "scored_text": text,
        "predictions": predictions,
        "correct": {
            key: exact_match(value, target) for key, value in predictions.items()
        },
    }


def summarize(records, metadata):
    keys = ("strict_match", "flexible_extract")
    return EvaluationResult(
        "gsm8k",
        len(records),
        {
            key: Metric(mean(row["correct"][key] for row in records), "ratio")
            for key in keys
        },
        {**metadata, "effective_counts": {key: len(records) for key in keys}},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--param_path", required=True)
    parser.add_argument("--data_dir", default="./gsm8k")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--prompt_format", choices=["raw", "chat"], default="raw")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--score_only", help="Rescore saved responses JSONL without loading a model"
    )
    args = parser.parse_args()
    check_output(args.output)
    if args.score_only:
        records = [score_response(row) for row in load_jsonl(args.score_only)]
        result = summarize(records, {"source": args.score_only})
        save_result(args.output, result)
        print({key: metric.value for key, metric in result.metrics.items()})
        return
    if (
        args.n_shot < 0
        or min(args.batch_size, args.max_tokens, args.max_seq_len) <= 0
        or (args.limit is not None and args.limit <= 0)
    ):
        parser.error("n_shot must be nonnegative; sizes and limit must be positive")
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    splits, hashes = {}, {}
    for split in ["train", "test"] if args.n_shot else ["test"]:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            dataset = load_dataset("openai/gsm8k", "main", split=split)
            temp = path.with_suffix(".tmp")
            temp.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dataset),
                encoding="utf-8",
            )
            temp.replace(path)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        splits[split] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
    problems = splits["test"][: args.limit]
    if not problems or args.n_shot > len(splits.get("train", [])):
        parser.error("Empty test split or insufficient training demonstrations")
    shot_indices = random.Random(args.seed).sample(
        range(len(splits.get("train", []))), args.n_shot
    )
    examples = [splits["train"][i] for i in shot_indices]
    golds = [extract_answer(row["answer"]) for row in problems]
    if any(answer is None for answer in golds):
        raise ValueError("GSM8K reference missing a #### numeric answer")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    engine = build_engine(
        args.param_path, max_batch_size=args.batch_size, max_seq_len=args.max_seq_len
    )
    records = []
    try:
        engine.model.eval()
        prompts = [
            build_prompt(
                row["question"], examples, engine.tokenizer, args.prompt_format
            )
            for row in problems
        ]
        for i, prompt in enumerate(prompts):
            if (
                len(engine.tokenizer.encode(prompt)) + args.max_tokens
                > args.max_seq_len
            ):
                raise ValueError(
                    f"Question {i} exceeds context budget; increase max_seq_len or reduce n_shot/max_tokens"
                )
        with output.with_suffix(".responses.jsonl").open("w", encoding="utf-8") as out:
            for start in tqdm(
                range(0, len(problems), args.batch_size), desc="GSM8K batches"
            ):
                batch = prompts[start : start + args.batch_size]
                responses = engine.generate(
                    batch,
                    stream=False,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    do_sample=False,
                    top_p=1.0,
                    top_k=0,
                    frequency_penalty=0.0,
                )
                if len(responses) != len(batch):
                    raise RuntimeError("Generation returned an incomplete batch")
                for i, response in enumerate(responses, start):
                    record = score_response(
                        {
                            "index": i,
                            "question": problems[i]["question"],
                            "reference": problems[i]["answer"],
                            "response": response,
                        }
                    )
                    records.append(record)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
    finally:
        engine.shutdown()
    result = summarize(
        records,
        {
            **vars(args),
            "dataset": "openai/gsm8k",
            "subset": "main",
            "split": "test",
            "data_sha256": hashes,
            "shot_indices": shot_indices,
            "temperature": 0.0,
            "do_sample": False,
            "note": "Seeded fixed train demonstrations; not an exact lm-eval harness reproduction.",
        },
    )
    save_result(output, result)
    print({key: metric.value for key, metric in result.metrics.items()})


if __name__ == "__main__":
    main()
