"""HumanEval benchmark — functional pipeline design.

Pipeline:
    load -> generate -> extract -> test -> score -> report

Each stage is a pure function (except GPU/CPU-bound I/O stages).
Config is a single dataclass; side effects are isolated at pipeline boundaries.
"""

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from math import prod
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm

from astrai.inference import InferenceEngine
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HUMANEVAL_URL = (
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
)

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
    test_timeout: float = 3.0
    test_workers: int = 8
    k_values: Tuple[int, ...] = (1, 10, 100)
    problem_indices: Optional[List[int]] = None


def download(url: str, path: str):
    if os.path.exists(path):
        return
    import gzip
    import urllib.request

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading {url} ...")
    tmp = path + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    with gzip.open(tmp, "rb") as f_in:
        with open(path, "wb") as f_out:
            f_out.write(f_in.read())
    os.remove(tmp)
    print(f"  saved to {path}")


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_engine(param_path: str, batch_size: int) -> InferenceEngine:
    model = AutoModel.from_pretrained(param_path)
    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model.to(device="cuda", dtype=torch.bfloat16)
    return InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=batch_size,
    )


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


def deduplicate(seq: Sequence[str]) -> List[str]:
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def generate_batch(
    engine: InferenceEngine,
    prompt: str,
    n: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    completions = []
    remaining = n
    while remaining > 0:
        current = min(batch_size, remaining)
        outputs = engine.generate(
            prompt=[prompt] * current,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        completions.extend(outputs if isinstance(outputs, list) else [outputs])
        remaining -= current
    return deduplicate(completions)


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


def execute_one(args: tuple) -> bool:
    full_code, entry_point, timeout = args
    try:
        r = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            timeout=timeout,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def test_one(item: dict, cfg: EvalConfig) -> Tuple[str, int, int]:
    from concurrent.futures import ProcessPoolExecutor

    task_id = item["task_id"]
    completions = item["completions"]
    codes = [
        (
            item["prompt"] + c + "\n" + item["test"],
            item["entry_point"],
            cfg.test_timeout,
        )
        for c in completions
    ]
    n = len(codes)
    passed = 0
    with ProcessPoolExecutor(max_workers=cfg.test_workers) as pool:
        for ok in pool.map(execute_one, codes):
            if ok:
                passed += 1
    return task_id, n, passed


def test_all(
    items: Sequence[dict],
    cfg: EvalConfig,
) -> Iterator[Tuple[str, int, int]]:
    for item in tqdm.tqdm(items, desc="Testing", unit="problem"):
        yield test_one(item, cfg)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def score_results(
    results: Iterator[Tuple[str, int, int]],
    k_values: Tuple[int, ...],
) -> Dict:
    # filter to k <= n (peek first result to get n)
    first = next(results)
    results = itertools.chain([first], results)
    n = first[1]
    k_values = tuple(k for k in k_values if k <= n)

    scores = {k: [] for k in k_values}
    output = {}
    for task_id, n, passed in results:
        entry = {"task_id": task_id, "n": n, "passed": passed}
        for k in k_values:
            pk = round(pass_at_k(n, passed, k), 4)
            entry[f"pass@{k}"] = pk
            scores[k].append(pk)
        output[task_id] = entry

    summary = {}
    for k in k_values:
        vals = scores[k]
        summary[f"pass@{k}"] = round(float(np.mean(vals)), 4)
    output["_summary"] = summary
    return output


def run_pipeline(cfg: EvalConfig) -> Dict:
    if cfg.test_only:
        with open(cfg.test_only, encoding="utf-8") as f:
            generated = json.load(f)
    else:
        download(HUMANEVAL_URL, cfg.data_path)

        problems = load_jsonl(cfg.data_path)
        if cfg.problem_indices:
            problems = [problems[i] for i in cfg.problem_indices if i < len(problems)]

        engine = create_engine(cfg.param_path, cfg.batch_size)

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

    results = test_all(generated, cfg)
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
    p.add_argument("--batch_size", type=int, default=32)
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
        test_workers=args.test_workers,
        test_timeout=args.test_timeout,
        problem_indices=args.problems,
    )


def report(scored: Dict):
    summary = scored.pop("_summary", {})
    print(f"\n{'=' * 60}")
    for k, v in summary.items():
        print(f"  {k}: {v:.2%}")
    print(f"{'=' * 60}")
    scored["_summary"] = summary


def main():
    cfg = parse_args()
    scored = run_pipeline(cfg)
    report(scored)
    if cfg.output:
        save_json(cfg.output, scored)
        print(f"Results saved to {cfg.output}")


if __name__ == "__main__":
    main()
