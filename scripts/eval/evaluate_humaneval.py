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
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import tqdm
from datasets import load_dataset

from astrai.bench import (
    deduplicate,
    generate_batch,
    load_jsonl,
    report,
    save_json,
    score_results,
    test_all,
)
from astrai.inference import InferenceEngine, build_engine

HUMANEVAL_HF_DATASET = "openai/openai_humaneval"

STOP_SEQUENCES = [
    "\nclass ",
    "\ndef ",
    "\n# ",
    "\nif __name__",
    "\nprint(",
    "\n\n\n",
]


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


def trim_stop(text: str) -> str:
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text


def extract_body(code: str, entry_point: str) -> Optional[str]:
    pattern = rf"def\s+{re.escape(entry_point)}\b[^:]*:"
    match = re.search(pattern, code)
    if not match:
        return code

    lines = code[match.end() :].split("\n")
    body_lines = []
    started = False

    for line in lines:
        stripped = line.rstrip()
        if not stripped and not started:
            continue
        if not stripped and started:
            body_lines.append("")
            continue
        if not started:
            started = True
        if stripped.lstrip() == stripped and started:
            break
        body_lines.append(stripped)

    body = "\n".join(body_lines)
    return body if body.strip() else None


def extract_completions(
    raw: Sequence[str],
    entry_point: str,
) -> List[str]:
    bodies = []
    for r in raw:
        t = trim_stop(r)
        body = extract_body(t, entry_point)
        if body:
            bodies.append(body)
    return bodies


def generate_all(
    engine: InferenceEngine,
    problems: Sequence[dict],
    cfg: EvalConfig,
) -> List[dict]:
    results = []
    for problem in tqdm.tqdm(problems, desc="Generating", unit="problem"):
        raw = generate_batch(
            engine,
            problem["prompt"],
            cfg.num_samples,
            cfg.batch_size,
            cfg.max_tokens,
            cfg.temperature,
            cfg.top_p,
            cfg.top_k,
        )
        bodies = extract_completions(raw, problem["entry_point"])
        results.append(
            dict(
                task_id=problem["task_id"],
                entry_point=problem["entry_point"],
                prompt=problem["prompt"],
                test=problem["test"],
                completions=bodies,
            )
        )
    return results


def he_codes(item: dict, test_timeout: float):
    """(task_id, [(full_code, timeout), ...]) — prompt + completion + test block."""
    codes = [
        (item["prompt"] + c + "\n" + item["test"], test_timeout)
        for c in item["completions"]
    ]
    return item["task_id"], codes


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

    results = test_all(
        generated,
        lambda it: he_codes(it, cfg.test_timeout),
        cfg.test_workers,
    )
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
