"""MMLU evaluation via log-likelihood ranking."""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import tqdm
from datasets import load_dataset

from astrai.bench import load_score_model, loglikelihood_batched

MMLU_HF_DATASET = "cais/mmlu"
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


def _write_subject_csv(data_dir: str, split: str, subject: str, rows: List[Dict]):
    split_dir = os.path.join(data_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    path = os.path.join(split_dir, f"{subject}_{split}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def download_mmlu(data_dir: str):
    print(f"Downloading MMLU from HuggingFace ({MMLU_HF_DATASET}) ...")
    letters = ("A", "B", "C", "D")
    split_map = {"dev": "dev", "val": "validation", "test": "test"}
    for local_split, hf_split in split_map.items():
        ds = load_dataset(MMLU_HF_DATASET, "all", split=hf_split)
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for item in tqdm.tqdm(ds, desc=f"  {local_split}", leave=False):
            subject = item["subject"]
            choices = item["choices"]
            ans_letter = letters[item["answer"]]
            grouped[subject].append(
                [
                    item["question"],
                    f"A){choices[0]}",
                    f"B){choices[1]}",
                    f"C){choices[2]}",
                    f"D){choices[3]}",
                    ans_letter,
                ]
            )
        for subject, rows in grouped.items():
            _write_subject_csv(data_dir, local_split, subject, rows)
        print(f"  {local_split}: {len(ds)} items, {len(grouped)} subjects")
    print(f"MMLU data saved to {data_dir}")


def _strip_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text


def load_csv(path: str) -> List[Dict]:
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


def build_prompt(question: str, choices: Dict, subject: str) -> str:
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
    tokenizer,
    raw_prompt: str,
    n_shot: int,
    dev_data: Optional[List[Dict]],
    subject: str = "",
) -> str:
    """Wrap raw MMLU prompt in the model's chat template format.

    For few-shot, prepend example Q&A pairs as user/assistant exchanges.
    Few-shot examples use the same subject preamble as the test question to
    keep the format consistent.
    """
    messages = []
    if n_shot > 0 and dev_data:
        for item in dev_data[:n_shot]:
            q = build_prompt(item["question"], item, subject)
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": item["answer"]})
    messages.append({"role": "user", "content": raw_prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def choice_logprobs_batched(
    model,
    tokenizer,
    context_ids_list: List[List[int]],
    device: str,
    max_model_len: int,
    answer_prefix: str = "",
) -> List[Dict[str, float]]:
    """Compute log-probs for multiple questions x 4 answer letters.

    Returns a list of dicts: [{A: score, B: score, C: score, D: score}, ...]
    """
    letters = ("A", "B", "C", "D")
    choice_ids_list = [
        tokenizer.encode(answer_prefix + c, add_special_tokens=False) for c in letters
    ]
    if any(
        len(ids) + max(map(len, choice_ids_list)) > max_model_len
        for ids in context_ids_list
    ):
        raise ValueError(
            "MMLU prompt exceeds context; reduce the number of demonstrations"
        )
    if all(len(ids) == 1 for ids in choice_ids_list):
        # One shared context forward suffices for four one-token answers.
        rows = context_ids_list
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        lengths = torch.tensor(list(map(len, rows)), device=device)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row, device=device)
        positions = torch.arange(width, device=device)[None].expand(len(rows), -1)
        mask = torch.tril(torch.ones(width, width, dtype=torch.bool, device=device))[
            None, None
        ]
        mask = mask & (positions < lengths[:, None])[:, None, None, :]
        with torch.inference_mode():
            output = model(
                ids,
                position_ids=positions,
                input_mask=mask,
                skip_lm_head=True,
                return_lm_head_weight=True,
            )
            hidden = output["hidden_states"][
                torch.arange(len(rows), device=device), lengths - 1
            ]
            logits = torch.nn.functional.linear(
                hidden, output["lm_head_weight"], output.get("lm_head_bias")
            )
            scores = torch.log_softmax(logits.float(), dim=-1)[
                :, [c[0] for c in choice_ids_list]
            ].tolist()
        return [dict(zip(letters, row)) for row in scores]
    requests = []
    for ctx_ids in context_ids_list:
        for cid in choice_ids_list:
            requests.append((ctx_ids, cid))
    scores = loglikelihood_batched(model, tokenizer, requests, device, max_model_len)
    return [
        {letters[ci]: scores[qi * 4 + ci] for ci in range(4)}
        for qi in range(len(context_ids_list))
    ]


def _permute_choices(item: Dict, rng: random.Random) -> Tuple[Dict, str]:
    """Shuffle the option order of a question.

    Returns ``(permuted_item, new_answer_letter)``. The question text and
    the *content* of each choice are unchanged; only which letter (A/B/C/D)
    maps to which content is shuffled. This neutralises the model's
    positional bias (e.g. always picking B).
    """
    letters = ("A", "B", "C", "D")
    perm = list(letters)
    rng.shuffle(perm)
    permuted = {"question": item["question"]}
    for new_letter, orig_letter in zip(letters, perm):
        permuted[new_letter] = item[orig_letter]
    new_answer = letters[perm.index(item["answer"])]
    return permuted, new_answer


def evaluate_subject(
    model,
    tokenizer,
    subject: str,
    test_data: List[Dict],
    dev_data: Optional[List[Dict]],
    device: str,
    n_shot: int,
    seed: int = 0,
    batch_size: int = 16,
    details: Optional[List[Dict]] = None,
    prompt_format: str = "raw",
    max_seq_len: Optional[int] = None,
) -> Tuple[float, int, int]:
    rng = random.Random(seed) if seed >= 0 else None
    correct = 0
    total = 0

    context_ids_list = []
    answers = []
    shot_counts = []
    max_model_len = min(
        model.config.max_position_embeddings,
        max_seq_len or model.config.max_position_embeddings,
    )
    for item in test_data:
        if rng is not None:
            permuted, answer = _permute_choices(item, rng)
        else:
            permuted, answer = item, item["answer"]
        shots = min(n_shot, len(dev_data or []))
        while True:
            if prompt_format == "raw":
                context = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
                for index, row in enumerate((dev_data or [])[:shots] + [permuted]):
                    context += row["question"] + "\n"
                    context += (
                        "\n".join(f"{c}. {row[c]}" for c in ("A", "B", "C", "D"))
                        + "\nAnswer:"
                    )
                    if index < shots:
                        context += " " + row["answer"] + "\n\n"
            else:
                raw_prompt = build_prompt(permuted["question"], permuted, subject)
                context = apply_chat(
                    tokenizer, raw_prompt, shots, dev_data or [], subject
                )
            ids = tokenizer.encode(context)
            if len(ids) < max_model_len:
                break
            if shots == 0:
                raise ValueError(
                    f"Question exceeds context even without demonstrations: {subject}"
                )
            shots -= 1
        context_ids_list.append(ids)
        shot_counts.append(shots)
        answers.append(answer)

    num_batches = (len(context_ids_list) + batch_size - 1) // batch_size
    for start in tqdm.tqdm(
        range(0, len(context_ids_list), batch_size),
        total=num_batches,
        desc=f"{subject:40s}",
        leave=False,
    ):
        batch = context_ids_list[start : start + batch_size]
        batch_answers = answers[start : start + batch_size]
        scores_list = choice_logprobs_batched(
            model,
            tokenizer,
            batch,
            device,
            max_model_len,
            answer_prefix=" " if prompt_format == "raw" else "",
        )
        for scores, answer in zip(scores_list, batch_answers):
            prediction = max(scores, key=scores.get)
            if details is not None:
                details.append(
                    {
                        "subject": subject,
                        "index": total,
                        "prediction": prediction,
                        "answer": answer,
                        "scores": scores,
                        "shots": shot_counts[total],
                        "prompt_tokens": len(context_ids_list[total]),
                    }
                )
            if prediction == answer:
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
    parser.add_argument("--prompt_format", choices=["raw", "chat"], default="raw")
    parser.add_argument("--max_seq_len", type=int, default=2048)
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
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for option permutation (-1 preserves original order)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of questions per batch (4 choices each = 4*B rows)",
    )
    args = parser.parse_args()

    if args.download or not os.path.exists(args.data_dir):
        download_mmlu(args.data_dir)

    device = args.device
    model, tokenizer = load_score_model(args.param_path, device, args.dtype)

    subjects = args.subjects or MMLU_SUBJECTS
    results = {}
    total_correct = 0
    total_questions = 0
    details = []

    for subject in subjects:
        dev_path = os.path.join(args.data_dir, "dev", f"{subject}_dev.csv")
        test_path = os.path.join(
            args.data_dir, args.split, f"{subject}_{args.split}.csv"
        )

        if not os.path.exists(test_path):
            raise FileNotFoundError(test_path)

        dev_data = load_csv(dev_path) if args.n_shot else None
        test_data = load_csv(test_path)

        acc, corr, tot = evaluate_subject(
            model,
            tokenizer,
            subject,
            test_data,
            dev_data,
            device,
            args.n_shot,
            seed=args.seed,
            batch_size=args.batch_size,
            details=details,
            prompt_format=args.prompt_format,
            max_seq_len=args.max_seq_len,
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
    results["_config"] = vars(args)

    if args.output:
        with open(
            args.output.replace(".json", "_details.jsonl"), "w", encoding="utf-8"
        ) as f:
            for row in details:
                f.write(json.dumps(row) + "\n")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
