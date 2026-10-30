"""Generate IFEval answers and score every constraint with Google's verifier."""

import argparse
import dataclasses
import importlib
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from astrai.inference import build_engine


def score_responses(problems, responses, verifier):
    results, summary = {}, {}
    for mode in ("strict", "loose"):
        check = getattr(verifier, f"test_instruction_following_{mode}")
        scored = [check(problem, responses) for problem in problems]
        flags = [flag for row in scored for flag in row.follow_instruction_list]
        passed = sum(row.follow_all_instructions for row in scored)
        results[mode] = [dataclasses.asdict(row) for row in scored]
        summary[mode] = {
            "prompt_correct": passed,
            "prompt_total": len(scored),
            "prompt_accuracy": passed / len(scored),
            "instruction_correct": sum(flags),
            "instruction_total": len(flags),
            "instruction_accuracy": sum(flags) / len(flags),
        }
    return {"_summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--param_path", required=True)
    parser.add_argument("--data_path", default="./ifeval/input_data.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--verifier_path",
        help="Parent of the official instruction_following_eval package",
    )
    parser.add_argument("--dump_responses")
    parser.add_argument("--max_tokens", type=int, default=1536)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.num_samples != 1:
        parser.error("IFEval reports one response per prompt; num_samples must be 1")
    if args.verifier_path:
        sys.path.insert(0, args.verifier_path)
    verifier = importlib.import_module("instruction_following_eval.evaluation_lib")
    path = Path(args.data_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset("google/IFEval", split="train")
        path.write_text("".join(json.dumps(row) + "\n" for row in dataset))
    problems = verifier.read_prompt_list(str(path))
    if args.limit:
        problems = problems[: args.limit]
    if not problems:
        parser.error("No IFEval prompts loaded")
    # Check dependencies and unknown instruction types before GPU generation.
    verifier.test_instruction_following_strict(
        problems[0], {problems[0].prompt: "test"}
    )
    torch.manual_seed(args.seed)
    engine = build_engine(
        args.param_path, max_batch_size=args.batch_size, max_seq_len=args.max_seq_len
    )
    engine.model.eval()
    responses = {}
    dump = args.dump_responses or str(Path(args.output).with_suffix(".responses.jsonl"))
    try:
        with open(dump, "w", encoding="utf-8") as out:
            for start in tqdm(
                range(0, len(problems), args.batch_size), desc="IFEval batches"
            ):
                batch = problems[start : start + args.batch_size]
                prompts = [
                    engine.tokenizer.apply_chat_template(
                        [{"role": "user", "content": problem.prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for problem in batch
                ]
                texts = engine.generate(
                    prompts,
                    stream=False,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    frequency_penalty=0.0,
                )
                for problem, text in zip(batch, texts):
                    responses[problem.prompt] = text
                    out.write(
                        json.dumps(
                            {
                                "key": problem.key,
                                "prompt": problem.prompt,
                                "response": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                out.flush()
    finally:
        engine.shutdown()
    results = score_responses(problems, responses, verifier)
    results["_config"] = vars(args)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results["_summary"], indent=2))


if __name__ == "__main__":
    main()
