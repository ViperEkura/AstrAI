"""IFEval instruction-following evaluation benchmark.

Evaluates model responses against regex-based constraint verifiers.
Supports all IFEval constraint types except language detection.

Usage::

    python scripts/tools/evaluate_ifeval.py --param_path ./params \
        --data_path ifeval.jsonl --output results.json \
        --temperature 0.1 --max_tokens 512
"""

import argparse
import json
import os
import re
import urllib.request
from typing import Callable, Dict, List, Optional

import torch
import tqdm

from astrai.inference import InferenceEngine
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer

IFEVAL_URL = (
    "https://raw.githubusercontent.com/google-research/"
    "google-research/master/instruction_following_eval/data/input_data.jsonl"
)

CONSTRAINT_VERIFIERS: Dict[str, Callable[[str, dict], bool]] = {}


def register(instruction_id: str):
    def decorator(fn):
        CONSTRAINT_VERIFIERS[instruction_id] = fn
        return fn

    return decorator


@register("keywords:existence")
def check_keyword_existence(response: str, kwargs: dict) -> bool:
    for kw in kwargs["keywords"]:
        if not re.search(re.escape(kw), response, re.IGNORECASE):
            return False
    return True


@register("keywords:frequency")
def check_keyword_frequency(response: str, kwargs: dict) -> bool:
    keyword = kwargs["keyword"]
    frequency = kwargs.get("frequency", 1)
    relation = kwargs.get("relation", "at least")
    count = len(re.findall(re.escape(keyword), response, re.IGNORECASE))
    if relation == "less than":
        return count < frequency
    return count >= frequency


@register("keywords:forbidden_words")
def check_forbidden_words(response: str, kwargs: dict) -> bool:
    for word in kwargs["forbidden_words"]:
        if re.search(r"\b" + re.escape(word) + r"\b", response, re.IGNORECASE):
            return False
    return True


@register("keywords:letter_frequency")
def check_letter_frequency(response: str, kwargs: dict) -> bool:
    letter = kwargs["letter"].lower()
    frequency = kwargs.get("let_frequency", 1)
    relation = kwargs.get("let_relation", "at least")
    count = response.lower().count(letter)
    if relation == "less than":
        return count < frequency
    return count >= frequency


@register("detectable_content:number_placeholders")
def check_placeholders(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_placeholders", 1)
    placeholders = re.findall(r"\[.*?\]", response)
    return len(placeholders) >= num


@register("detectable_content:postscript")
def check_postscript(response: str, kwargs: dict) -> bool:
    marker = kwargs.get("postscript_marker", "P.S.")
    response_lower = response.lower()
    if marker == "P.P.S":
        return bool(re.search(r"p\.\s?p\.\s?s", response_lower))
    elif marker == "P.S.":
        return bool(re.search(r"p\.\s?s\.", response_lower))
    else:
        return bool(re.search(re.escape(marker.lower()), response_lower))


@register("detectable_format:number_bullet_lists")
def check_bullet_lists(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_bullets", 1)
    bullets = re.findall(r"^\s*\*[^\*].*$", response, re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", response, re.MULTILINE)
    return len(bullets) + len(dashes) == num


@register("detectable_format:number_highlighted_sections")
def check_highlighted_sections(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_highlights", 1)
    highlights = re.findall(r"\*[^\n\*]+\*", response)
    count = 0
    for h in highlights:
        if h.strip("*").strip():
            count += 1
    return count >= num


@register("detectable_format:multiple_sections")
def check_multiple_sections(response: str, kwargs: dict) -> bool:
    splitter = kwargs.get("section_spliter", "Section")
    num = kwargs.get("num_sections", 1)
    pattern = r"\s?" + re.escape(splitter) + r"\s?\d+\s?"
    sections = re.split(pattern, response)
    return len(sections) - 1 >= num


@register("detectable_format:title")
def check_title(response: str, kwargs: dict) -> bool:
    titles = re.findall(r"<<[^>\n]+>>", response)
    for title in titles:
        if title.strip("<>").strip():
            return True
    return False


@register("detectable_format:json_format")
def check_json_format(response: str, kwargs: dict) -> bool:
    value = response.strip()
    for prefix in ("```json", "```Json", "```JSON", "```"):
        if value.lower().startswith(prefix.lower()):
            value = value[len(prefix) :].strip()
    if value.endswith("```"):
        value = value[:-3].strip()
    try:
        json.loads(value)
        return True
    except (ValueError, json.JSONDecodeError):
        return False


@register("detectable_format:general_punctuation")
def check_general_punctuation(response: str, kwargs: dict) -> bool:
    punctuation_blacklist = kwargs.get("punctuation_blacklist", [])
    for punct in punctuation_blacklist:
        if punct in response:
            return False
    return True


@register("detectable_format:number_highlighted_words")
def check_highlighted_words(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_highlights", 1)
    highlights = re.findall(r"\*[^\s\*][^\*]*[^\s\*]\*", response)
    return len(highlights) >= num


@register("startend:end_checker")
def check_end_checker(response: str, kwargs: dict) -> bool:
    end_phrase = kwargs["end_phrase"]
    return (
        response.strip()
        .rstrip('"')
        .rstrip()
        .lower()
        .endswith(end_phrase.strip().lower())
    )


@register("startend:quotation")
def check_quotation(response: str, kwargs: dict) -> bool:
    value = response.strip()
    return value.startswith('"') and value.endswith('"')


@register("startend:start_checker")
def check_start_checker(response: str, kwargs: dict) -> bool:
    starter = kwargs["starter"]
    return bool(re.search(r"^\s*" + re.escape(starter), response, re.MULTILINE))


@register("change_case:english_capital")
def check_english_capital(response: str, kwargs: dict) -> bool:
    return response.isupper()


@register("change_case:english_lowercase")
def check_english_lowercase(response: str, kwargs: dict) -> bool:
    return response.islower()


@register("change_case:capital_word_frequency")
def check_capital_word_frequency(response: str, kwargs: dict) -> bool:
    frequency = kwargs.get("capital_frequency", 1)
    relation = kwargs.get("capital_relation", "at least")
    capital_words = re.findall(r"\b[A-Z]{2,}\b", response)
    count = len(capital_words)
    if relation == "less than":
        return count < frequency
    return count >= frequency


@register("punctuation:no_comma")
def check_no_comma(response: str, kwargs: dict) -> bool:
    return "," not in response


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def count_sentences(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return len([s for s in sentences if s.strip()])


@register("length_constraints:number_words")
def check_number_words(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_words", 100)
    relation = kwargs.get("relation", "at least")
    cnt = count_words(response)
    if relation == "less than":
        return cnt < num
    return cnt >= num


@register("length_constraints:number_sentences")
def check_number_sentences(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_sentences", 5)
    relation = kwargs.get("relation", "at least")
    cnt = count_sentences(response)
    if relation == "less than":
        return cnt < num
    return cnt >= num


@register("length_constraints:number_paragraphs")
def check_number_paragraphs(response: str, kwargs: dict) -> bool:
    num = kwargs.get("num_paragraphs", 1)
    if "***" in response:
        paragraphs = re.split(r"\s?\*\*\*\s?", response)
    else:
        paragraphs = re.split(r"\n\n+", response)
    actual = len([p for p in paragraphs if p.strip()])
    return actual == num


@register("length_constraints:nth_paragraph_first_word")
def check_nth_paragraph_first_word(response: str, kwargs: dict) -> bool:
    num_paragraphs = kwargs.get("num_paragraphs", 1)
    nth = kwargs.get("nth_paragraph", 1)
    first_word = kwargs.get("first_word", "").lower()

    paragraphs = re.split(r"\n\n+", response)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if len(paragraphs) != num_paragraphs:
        return False
    if nth > len(paragraphs):
        return False

    target = paragraphs[nth - 1]
    words = target.split()
    if not words:
        return False

    word = words[0].strip().lstrip("'\"").rstrip(".,!?:;\"'")
    return word.lower() == first_word


@register("length_constraints:nth_word_checker")
def check_nth_word(response: str, kwargs: dict) -> bool:
    nth = kwargs.get("nth_word", 1)
    target = kwargs.get("target_word", "").lower()
    words = re.findall(r"\b\w+\b", response)
    if nth > len(words):
        return False
    return words[nth - 1].lower() == target


@register("combination:repeat_prompt")
def check_repeat_prompt(response: str, kwargs: dict) -> bool:
    prompt = kwargs["prompt_to_repeat"]
    return response.strip().lower().startswith(prompt.strip().lower())


@register("combination:two_responses")
def check_two_responses(response: str, kwargs: dict) -> bool:
    parts = response.split("******")
    valid = [p for p in parts if p.strip()]
    if len(valid) != 2:
        return False
    return valid[0].strip() != valid[1].strip()


def download_ifeval(data_path: str):
    if os.path.exists(data_path):
        return
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    print(f"Downloading IFEval from {IFEVAL_URL} ...")
    tmp = data_path + ".tmp"
    urllib.request.urlretrieve(IFEVAL_URL, tmp)
    with open(tmp, "rb") as f_in:
        content = f_in.read()
    with open(data_path, "wb") as f_out:
        f_out.write(content)
    os.remove(tmp)
    print(f"  saved to {data_path}")


def load_problems(data_path: str) -> List[dict]:
    problems = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def verify_response(response: str, instruction_id: str, kwargs: dict) -> Optional[bool]:
    verifier = CONSTRAINT_VERIFIERS.get(instruction_id)
    if verifier is None:
        return None
    try:
        return verifier(response, kwargs)
    except Exception:
        return False


def generate_one(
    engine: InferenceEngine,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    output = engine.generate(
        prompt=formatted,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    if isinstance(output, list):
        return output[0]
    return output


def evaluate(
    engine: InferenceEngine,
    tokenizer: AutoTokenizer,
    problems: List[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    num_samples: int = 1,
) -> Dict:
    results = {}
    constraint_stats: Dict[str, Dict[str, int]] = {}
    total_constraints = 0
    total_passed = 0

    for problem in tqdm.tqdm(problems, desc="IFEval", unit="problem"):
        key = problem["key"]
        prompt = problem["prompt"]
        instruction_ids = problem["instruction_id_list"]
        kwargs_list = problem["kwargs"]

        samples = []
        for _ in range(num_samples):
            response = generate_one(
                engine, tokenizer, prompt, max_tokens, temperature, top_p, top_k
            )
            samples.append(response)

        constraint_results = []
        passed = 0
        verified = 0

        for idx, instruction_id in enumerate(instruction_ids):
            kwargs = kwargs_list[idx] if idx < len(kwargs_list) else {}
            best_pass = False
            for response in samples:
                result = verify_response(response, instruction_id, kwargs)
                if result is None:
                    continue
                if result:
                    best_pass = True
                    break

            verifier_exists = instruction_id in CONSTRAINT_VERIFIERS
            if verifier_exists:
                verified += 1
                if best_pass:
                    passed += 1

            constraint_results.append(
                {
                    "instruction_id": instruction_id,
                    "passed": best_pass,
                    "supported": verifier_exists,
                    "kwargs": kwargs,
                }
            )

            if verifier_exists:
                if instruction_id not in constraint_stats:
                    constraint_stats[instruction_id] = {
                        "total": 0,
                        "passed": 0,
                    }
                constraint_stats[instruction_id]["total"] += 1
                if best_pass:
                    constraint_stats[instruction_id]["passed"] += 1

        total_constraints += verified
        total_passed += passed

        accuracy = passed / verified if verified > 0 else None
        results[str(key)] = {
            "key": key,
            "prompt": prompt,
            "response": samples[0],
            "num_samples": num_samples,
            "num_constraints": len(instruction_ids),
            "num_verified": verified,
            "num_passed": passed,
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
            "constraints": constraint_results,
        }

    overall_accuracy = (
        round(total_passed / total_constraints, 4) if total_constraints > 0 else 0.0
    )

    type_summary = {}
    for inst_id, stats in sorted(constraint_stats.items()):
        type_summary[inst_id] = {
            "total": stats["total"],
            "passed": stats["passed"],
            "accuracy": round(stats["passed"] / stats["total"], 4)
            if stats["total"] > 0
            else 0.0,
        }

    unsupported_count = sum(
        1
        for p in problems
        for iid in p["instruction_id_list"]
        if iid not in CONSTRAINT_VERIFIERS
    )

    results["_summary"] = {
        "total_problems": len(problems),
        "total_constraints": total_constraints,
        "total_passed": total_passed,
        "overall_accuracy": overall_accuracy,
        "unsupported_constraints": unsupported_count,
        "supported_types": sorted(CONSTRAINT_VERIFIERS.keys()),
        "per_type_accuracy": type_summary,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="IFEval benchmark")
    parser.add_argument(
        "--param_path", type=str, default="./params", help="Model directory"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./ifeval/input_data.jsonl",
        help="IFEval JSONL file (auto-download if missing)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Max generation tokens"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples per problem (best-of-n scoring)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Inference batch size"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N problems (for quick testing)",
    )
    parser.add_argument(
        "--dump_responses",
        type=str,
        default=None,
        help="Path to dump raw model responses (JSONL)",
    )
    args = parser.parse_args()

    download_ifeval(args.data_path)
    problems = load_problems(args.data_path)
    if args.limit:
        problems = problems[: args.limit]

    print(f"Loaded {len(problems)} problems")
    print(f"Supported constraint types: {len(CONSTRAINT_VERIFIERS)}")

    model = AutoModel.from_pretrained(args.param_path)
    tokenizer = AutoTokenizer.from_pretrained(args.param_path)
    model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=args.batch_size,
    )

    results = evaluate(
        engine=engine,
        tokenizer=tokenizer,
        problems=problems,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_samples=args.num_samples,
    )

    summary = results.pop("_summary")
    print(f"\n{'=' * 60}")
    print(f"  Problems:       {summary['total_problems']}")
    print(f"  Constraints:    {summary['total_constraints']}")
    print(f"  Passed:         {summary['total_passed']}")
    print(f"  Accuracy:       {summary['overall_accuracy']:.2%}")
    print(f"  Unsupported:    {summary['unsupported_constraints']}")
    print(f"{'=' * 60}")

    print(f"\nPer-type accuracy:")
    for inst_id, stats in sorted(summary["per_type_accuracy"].items()):
        print(
            f"  {inst_id:50s}  {stats['accuracy']:.2%}  "
            f"({stats['passed']}/{stats['total']})"
        )

    if args.output:
        results["_summary"] = summary
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    if args.dump_responses:
        with open(args.dump_responses, "w", encoding="utf-8") as f:
            for k, v in results.items():
                if k.startswith("_"):
                    continue
                f.write(
                    json.dumps(
                        {
                            "key": v["key"],
                            "prompt": v["prompt"],
                            "response": v["response"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"Responses dumped to {args.dump_responses}")

    engine.shutdown()


if __name__ == "__main__":
    main()
