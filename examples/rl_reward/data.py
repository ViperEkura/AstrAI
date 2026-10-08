"""Load immutable JSONL splits and prepare their exact rollout prompts."""

import hashlib
import json
from pathlib import Path

from torch.utils.data import Dataset

from examples.rl_reward.rewards import numeric_answer


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits(paths, task, tokenizer, prompt_cap):
    seen_ids, seen_tasks = set(), set()
    records_by_prompt = {}
    splits = {}
    for split, path in paths.items():
        records = []
        with Path(path).open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    identifier = record["id"]
                    if not isinstance(identifier, str) or not identifier.strip():
                        raise ValueError("id must be a nonempty string")
                    if identifier in seen_ids:
                        raise ValueError("duplicate id across splits")
                    if task == "countdown":
                        numbers, target = record["numbers"], record["target"]
                        if (
                            not isinstance(numbers, list)
                            or not 2 <= len(numbers) <= 32
                            or any(
                                type(n) is not int or not 0 < n <= 1000000
                                for n in numbers
                            )
                            or type(target) is not int
                        ):
                            raise ValueError("invalid Countdown numbers or target")
                        task_key = json.dumps([sorted(numbers), target])
                        question = (
                            f"Use every number in {numbers} exactly once to make {target}. "
                            "Use only +, -, *, / and parentheses. "
                            "Return the expression inside <answer>...</answer>."
                        )
                    else:
                        question = record["question"]
                        if not isinstance(question, str) or not question.strip():
                            raise ValueError("question must be a nonempty string")
                        if numeric_answer(record["answer"]) is None:
                            raise ValueError("invalid GSM8K answer")
                        task_key = " ".join(question.split())
                        question += "\nEnd with #### followed by the numeric answer."
                    if task_key in seen_tasks:
                        raise ValueError("duplicate task across splits")
                    messages = [{"role": "user", "content": question}]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if len(tokenizer.encode(prompt)) > prompt_cap:
                        raise ValueError("prompt exceeds the frozen prompt cap")
                    if prompt in records_by_prompt:
                        raise ValueError("rendered prompt collision")
                    record = {**record, "split": split, "messages": messages}
                    record["dedup_hash"] = hashlib.sha256(task_key.encode()).hexdigest()
                    record["prompt_hash"] = hashlib.sha256(prompt.encode()).hexdigest()
                    records_by_prompt[prompt] = record
                    records.append(record)
                    seen_ids.add(identifier)
                    seen_tasks.add(task_key)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
        if not records:
            raise ValueError(f"{split} split is empty")
        splits[split] = records
    return splits, records_by_prompt


class PromptDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]["messages"]


def collate_prompts(batch):
    return {"messages": batch}
