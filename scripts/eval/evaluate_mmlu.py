"""MMLU evaluation via log-likelihood ranking."""

import argparse
import csv
import json
import os
import shutil
import tarfile

import requests
import torch
import torch.nn.functional as F
import tqdm

from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer

MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def _download_and_extract(url: str, data_dir: str):
    tar_path = os.path.join(data_dir, "data.tar")
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading MMLU data from {url}...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with tqdm.tqdm(total=total, unit="B", unit_scale=True, desc="  Download") as bar:
        with open(tar_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    print("Extracting...")
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(data_dir)
    os.remove(tar_path)


def download_mmlu(data_dir: str):
    _download_and_extract(MMLU_URL, data_dir)
    src = os.path.join(data_dir, "data")
    if os.path.exists(src):
        for item in os.listdir(src):
            src_item = os.path.join(src, item)
            dst_item = os.path.join(data_dir, item)
            if os.path.exists(dst_item):
                if os.path.isdir(dst_item):
                    shutil.rmtree(dst_item)
                else:
                    os.remove(dst_item)
            os.rename(src_item, dst_item)
        os.rmdir(src)
    print(f"MMLU data saved to {data_dir}")


def _strip_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text


def load_csv(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            if row[0].strip().lower() == "question":
                continue
            data.append(
                {
                    "question": row[0].strip(),
                    "A": _strip_prefix(row[1].strip(), "A)"),
                    "B": _strip_prefix(row[2].strip(), "B)"),
                    "C": _strip_prefix(row[3].strip(), "C)"),
                    "D": _strip_prefix(row[4].strip(), "D)"),
                    "answer": row[5].strip(),
                }
            )
    return data


def build_prompt(question: str, choices: dict, subject: str) -> str:
    """Build the raw question prompt (without few-shot examples).

    Few-shot examples are handled by ``apply_chat`` to avoid duplication.
    """
    prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    prompt += f"Question: {question}\n"
    for k in ("A", "B", "C", "D"):
        prompt += f"{k}. {choices[k]}\n"
    prompt += "Answer:"
    return prompt


def apply_chat(
    tokenizer, raw_prompt: str, n_shot: int, dev_data: list[dict] | None
) -> str:
    """Wrap raw MMLU prompt in the model's chat template format.

    For few-shot, prepend example Q&A pairs as a second user/assistant exchange.
    """
    messages = []
    if n_shot > 0 and dev_data:
        for item in dev_data[:n_shot]:
            q = f"Question: {item['question']}\n"
            for k in ("A", "B", "C", "D"):
                q += f"{k}. {item[k]}\n"
            q += "Answer:"
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": item["answer"]})
    messages.append({"role": "user", "content": raw_prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def choice_logprob(
    model, tokenizer, context_ids: list[int], choice_letter: str, device: str
) -> float:
    choice_text = choice_letter
    choice_ids = tokenizer.encode(choice_text, add_special_tokens=False)
    input_ids = context_ids + choice_ids
    max_len = model.config.max_len
    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        input_ids = input_ids[overflow:]
        ctx_len = len(input_ids) - len(choice_ids)
    else:
        ctx_len = len(context_ids)

    input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_tensor)["logits"][0]

    score = 0.0
    for i, tid in enumerate(choice_ids):
        pos = ctx_len - 1 + i
        if pos >= len(logits):
            break
        score += F.log_softmax(logits[pos], dim=-1)[tid].item()
    return score


def evaluate_subject(
    model,
    tokenizer,
    subject: str,
    test_data: list[dict],
    dev_data: list[dict] | None,
    device: str,
    n_shot: int,
) -> tuple[float, int, int]:
    correct = 0
    total = 0
    for item in tqdm.tqdm(test_data, desc=f"{subject:40s}", leave=False):
        raw_prompt = build_prompt(item["question"], item, subject)
        context = apply_chat(tokenizer, raw_prompt, n_shot, dev_data or [])
        context_ids = tokenizer.encode(context)
        scores = {
            c: choice_logprob(model, tokenizer, context_ids, c, device)
            for c in ("A", "B", "C", "D")
        }
        if max(scores, key=scores.get) == item["answer"]:
            correct += 1
        total += 1
    return correct / total, correct, total


def main():
    parser = argparse.ArgumentParser(description="MMLU evaluation")
    parser.add_argument(
        "--param_path", type=str, default="./params", help="Model directory"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./mmlu_data", help="MMLU data directory"
    )
    parser.add_argument("--download", action="store_true", help="Download MMLU data")
    parser.add_argument(
        "--n_shot", type=int, default=5, help="Few-shot examples (0 for zero-shot)"
    )
    parser.add_argument(
        "--subjects", type=str, nargs="+", help="Specific subjects (default: all)"
    )
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() else "float32",
        help="Torch dtype",
    )
    args = parser.parse_args()

    if args.download or not os.path.exists(args.data_dir):
        download_mmlu(args.data_dir)

    model = AutoModel.from_pretrained(args.param_path)
    tokenizer = AutoTokenizer.from_pretrained(args.param_path)
    device = args.device
    dtype = getattr(torch, args.dtype)
    model.to(device=device, dtype=dtype)
    model.eval()

    subjects = args.subjects or MMLU_SUBJECTS
    results = {}
    total_correct = 0
    total_questions = 0

    for subject in subjects:
        dev_path = os.path.join(args.data_dir, "dev", f"{subject}_dev.csv")
        test_path = os.path.join(
            args.data_dir, args.split, f"{subject}_{args.split}.csv"
        )

        if not os.path.exists(test_path):
            print(f"  Skipping {subject}: test file not found")
            continue

        dev_data = load_csv(dev_path) if os.path.exists(dev_path) else None
        test_data = load_csv(test_path)

        acc, corr, tot = evaluate_subject(
            model, tokenizer, subject, test_data, dev_data, device, args.n_shot
        )
        results[subject] = {"accuracy": round(acc, 4), "correct": corr, "total": tot}
        total_correct += corr
        total_questions += tot
        print(f"  {subject:40s}  {acc:.2%}  ({corr}/{tot})")

    overall = total_correct / total_questions if total_questions else 0
    print(f"\n{'=' * 70}")
    print(f"  Overall: {overall:.2%}  ({total_correct}/{total_questions})")
    results["_overall"] = {
        "accuracy": round(overall, 4),
        "correct": total_correct,
        "total": total_questions,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
