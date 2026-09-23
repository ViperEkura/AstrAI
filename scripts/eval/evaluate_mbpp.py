"""MBPP benchmark — Mostly Basic Python Problems (chat zero-shot protocol).

Mirrors the HumanEval harness (generate -> extract -> execute -> pass@k) with
MBPP specifics:
  - **chat-template zero-shot** prompt: this suite's SFT models derail on the
    raw 3-shot completion transcript (completions blend shot content), so the
    task description + asserts + canonical signature go through the tokenizer's
    chat template — same convention as MMLU/IFEval in this repo. Absolute
    numbers are NOT comparable to published raw-few-shot MBPP results;
    cross-checkpoint comparisons are.
  - execution = completion + test_list asserts (challenge tests excluded)
  - pass@1 / pass@10 over n unique extracted completions

Data: `google-research-datasets/mbpp` "full" config, test split (500 problems,
task_ids 11-510). Local cache: ./mbpp/mbpp_test.jsonl.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import tqdm
from datasets import load_dataset

from astrai.bench import (
    generate_batch,
    load_jsonl,
    report,
    save_json,
    score_results,
    test_all,
)
from astrai.inference import build_engine

MBPP_HF_DATASET = "google-research-datasets/mbpp"

# Completion stops. NOTE: no "\ndef " stop here — unlike HumanEval, the MBPP
# answer *starts* with the def; clean_completion anchors to the first def.
STOP_SEQUENCES = [
    "\nclass ",
    "\n# ",
    "\nif __name__",
    "\nprint(",
    "\n\n\n",
]


@dataclass
class EvalConfig:
    param_path: str = "./params"
    data_path: str = "./mbpp/mbpp_test.jsonl"
    output: Optional[str] = None

    test_only: Optional[str] = None
    generate_only: bool = False

    num_samples: int = 20
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    batch_size: int = 32
    max_seq_len: int = 4096
    test_timeout: float = 3.0
    test_workers: int = 8
    k_values: tuple = (1, 10)
    problem_indices: Optional[List[int]] = None


def download(data_dir: str = "./mbpp"):
    test_path = os.path.join(data_dir, "mbpp_test.jsonl")
    if os.path.exists(test_path):
        return
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading MBPP from HuggingFace ({MBPP_HF_DATASET}) ...")
    test = load_dataset(MBPP_HF_DATASET, "full", split="test")
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  saved {len(test)} problems to {data_dir}")


def build_prompt(problem: dict, tokenizer) -> str:
    """Chat-wrapped task + asserts + exact signature.

    The signature line comes from the canonical solution's first def — without
    it, ~65% of failures are NameError/TypeError from the model inventing
    function names (tests use names like `remove_Occ` that the task text never
    states). With the signature this matches HumanEval semantics (signature
    given, logic is the task).
    """
    sig = next(
        (
            l
            for l in problem.get("code", "").splitlines()
            if l.lstrip().startswith("def ")
        ),
        "",
    )
    content = (
        f"{problem['text']}\n\nYour code should pass these tests:\n"
        + "\n".join(problem["test_list"])
        + (
            f"\n\nImplement a function with exactly this signature:\n{sig}"
            if sig
            else ""
        )
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def trim_stop(text: str) -> str:
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text


def clean_completion(raw: str) -> str:
    t = trim_stop(raw).strip()
    # prefer a fenced code block when present (chat answers wrap code in ```)
    blocks = re.findall(r"```(?:[a-zA-Z]*\n)?(.*?)```", t, re.S)
    if blocks:
        t = blocks[0]
    else:
        # strip a bare leading fence without a closing one
        t = re.sub(r"^```[a-zA-Z]*\s*\n?", "", t)
    # anchor to the first function definition: drops leading prose or extra
    # assert lines the model may emit before writing the solution
    idx = t.find("def ")
    if idx > 0:
        t = t[idx:]
    return t.strip()


def generate_all(engine, problems: Sequence[dict], cfg: EvalConfig) -> List[dict]:
    tokenizer = engine.tokenizer
    results = []
    for problem in tqdm.tqdm(problems, desc="Generating", unit="problem"):
        raw = generate_batch(
            engine,
            build_prompt(problem, tokenizer),
            cfg.num_samples,
            cfg.batch_size,
            cfg.max_tokens,
            cfg.temperature,
            cfg.top_p,
            cfg.top_k,
        )
        bodies = [c for c in (clean_completion(r) for r in raw) if c]
        results.append(
            dict(
                task_id=problem["task_id"],
                text=problem["text"],
                test_list=problem["test_list"],
                test_setup_code=problem.get("test_setup_code", ""),
                completions=bodies,
            )
        )
    return results


def mbpp_codes(item: dict, test_timeout: float):
    """(task_id, [(full_code, timeout), ...]) — completion + assert block."""
    tests = "\n".join(item["test_list"])
    setup = item["test_setup_code"]
    prefix = setup + "\n" if setup else ""
    codes = [(prefix + c + "\n" + tests, test_timeout) for c in item["completions"]]
    return item["task_id"], codes


def run_pipeline(cfg: EvalConfig) -> dict:
    if cfg.test_only:
        with open(cfg.test_only, encoding="utf-8") as f:
            generated = json.load(f)
    else:
        download(os.path.dirname(cfg.data_path) or ".")

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
        generated, lambda it: mbpp_codes(it, cfg.test_timeout), cfg.test_workers
    )
    scored = score_results(results, cfg.k_values)
    return scored


def parse_args(argv: Optional[List[str]] = None) -> EvalConfig:
    p = argparse.ArgumentParser(description="MBPP benchmark (chat zero-shot)")
    p.add_argument("--param_path", type=str, default="./params")
    p.add_argument("--data_path", type=str, default="./mbpp/mbpp_test.jsonl")
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
    p.add_argument("--num_samples", type=int, default=20)
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
