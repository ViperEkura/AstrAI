"""HumanEval benchmark — functional pipeline design.

Pipeline:
    load -> generate -> extract -> test -> score -> report

Each stage is a pure function (except GPU/CPU-bound I/O stages).
Config is a single dataclass; side effects are isolated at pipeline boundaries.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import tqdm
from datasets import load_dataset

from astrai.bench import (
    execute_one,
    load_jsonl,
    report,
    save_json,
    score_results,
)
from astrai.inference import InferenceEngine, build_engine

HUMANEVAL_HF_DATASET = "openai/openai_humaneval"


@dataclass
class EvalConfig:
    param_path: str = "./params"
    data_path: str = "./humaneval/HumanEval.jsonl"
    output: Optional[str] = None

    test_only: Optional[str] = None
    generate_only: bool = False

    num_samples: int = 200
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    batch_size: int = 32
    max_seq_len: int = 4096
    test_timeout: float = 3.0
    test_workers: int = 8
    k_values: Tuple[int, ...] = (1, 10, 100)
    problem_indices: Optional[List[int]] = None
    prompt_format: str = "completion"


def download(path: str):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading HumanEval from HuggingFace ({HUMANEVAL_HF_DATASET}) ...")
    ds = load_dataset(HUMANEVAL_HF_DATASET, split="test")
    with open(path, "w", encoding="utf-8") as f:
        for item in ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  saved {len(ds)} problems to {path}")


def extract_completions(
    raw: Sequence[str],
    entry_point: str,
) -> List[str]:
    bodies = []
    for r in raw:
        blocks = re.findall(r"```(?:python|Python|py)?\s*\n(.*?)```", r, re.S)
        code = next(
            (b for b in blocks if re.search(rf"def\s+{re.escape(entry_point)}\b", b)),
            blocks[0] if blocks else r,
        )
        # Preserve every sample, including failed extractions, in the denominator.
        bodies.append(code)
    return bodies


def generate_all(
    engine: InferenceEngine, problems: Sequence[Dict], cfg: EvalConfig
) -> List[Dict]:
    results, requests = [], []
    for index, problem in enumerate(problems):
        prompt = problem["prompt"]
        if cfg.prompt_format == "chat":
            prompt = engine.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "Complete the following Python function. Return the complete function and any necessary imports as Python code only.\n\n"
                        + prompt,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        results.append(
            dict(
                request_id=problem.get("task_id", problem.get("request_id")),
                entry_point=problem["entry_point"],
                prompt=problem["prompt"],
                test=problem["test"],
                raw_completions=[],
            )
        )
        requests.extend([(index, prompt)] * cfg.num_samples)
    for start in tqdm.tqdm(
        range(0, len(requests), cfg.batch_size), desc="Generating batches"
    ):
        batch = requests[start : start + cfg.batch_size]
        outputs = engine.generate(
            [prompt for _, prompt in batch],
            stream=False,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            frequency_penalty=0.0,
        )
        for (index, _), output in zip(batch, outputs):
            results[index]["raw_completions"].append(output)
    for result in results:
        result["completions"] = extract_completions(
            result["raw_completions"], result["entry_point"]
        )
    return results


def evaluate_completions(generated: Sequence[Dict], cfg: EvalConfig):
    results = []
    with ThreadPoolExecutor(max_workers=cfg.test_workers) as pool:
        for item in tqdm.tqdm(generated, desc="Testing"):
            codes = []
            for completion in item["completions"]:
                full_function = re.search(
                    rf"^def\s+{re.escape(item['entry_point'])}\b",
                    completion,
                    re.MULTILINE,
                )
                code = (
                    item["prompt"]
                    + ("    pass\n\n" if full_function else "")
                    + completion
                )
                code += "\n" + item["test"] + f"\ncheck({item['entry_point']})\n"
                codes.append((code, cfg.test_timeout))
            flags = list(pool.map(execute_one, codes))
            item["passed_samples"] = flags
            results.append((item["request_id"], len(codes), sum(flags)))
    return results


def run_pipeline(cfg: EvalConfig) -> Dict:
    if cfg.test_only:
        with open(cfg.test_only, encoding="utf-8") as f:
            generated = json.load(f)
    else:
        download(cfg.data_path)

        problems = load_jsonl(cfg.data_path)
        if cfg.problem_indices:
            problems = [problems[i] for i in cfg.problem_indices if i < len(problems)]

        engine = build_engine(
            cfg.param_path,
            max_batch_size=cfg.batch_size,
            max_seq_len=cfg.max_seq_len,
        )

        engine.model.eval()
        try:
            generated = generate_all(engine, problems, cfg)
        finally:
            engine.shutdown()

        if cfg.output:
            mid = cfg.output.replace(".json", "_completions.json")
            save_json(mid, generated)
            print(f"Completions saved to {mid}")

        if cfg.generate_only:
            return {}

    results = evaluate_completions(generated, cfg)
    if cfg.output:
        save_json(cfg.output.replace(".json", "_tested.json"), generated)
    scored = score_results(results, cfg.k_values)
    return scored


def parse_args(argv: Optional[List[str]] = None) -> EvalConfig:
    p = argparse.ArgumentParser(description="HumanEval benchmark")
    p.add_argument("--param_path", type=str, default="./params")
    p.add_argument("--data_path", type=str, default="./humaneval/HumanEval.jsonl")
    p.add_argument("--output", type=str, default=None)
    p.add_argument(
        "--test_only",
        type=str,
        default=None,
        help="Skip generation, test existing completions JSON",
    )
    p.add_argument(
        "--generate_only", action="store_true", help="Only generate, skip testing"
    )
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--test_workers", type=int, default=8)
    p.add_argument("--test_timeout", type=float, default=3.0)
    p.add_argument("--problems", type=int, nargs="+", default=None)
    p.add_argument(
        "--prompt_format", choices=["completion", "chat"], default="completion"
    )
    args = p.parse_args(argv)

    return EvalConfig(
        param_path=args.param_path,
        data_path=args.data_path,
        output=args.output,
        test_only=args.test_only,
        generate_only=args.generate_only,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        test_workers=args.test_workers,
        test_timeout=args.test_timeout,
        problem_indices=args.problems,
        prompt_format=args.prompt_format,
    )


def main():
    cfg = parse_args()
    scored = run_pipeline(cfg)
    report(scored)
    if cfg.output:
        save_json(cfg.output, scored)
        print(f"Results saved to {cfg.output}")


if __name__ == "__main__":
    main()
