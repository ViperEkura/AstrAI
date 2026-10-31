"""Deterministic task verifiers; generated text is never executed."""

import ast
import re
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

import torch

from astrai.trainer.rollout import BaseRewardModel

VERIFIER_VERSION = "arithmetic-ast-v1"


@dataclass(frozen=True)
class Score:
    accuracy: float
    valid_format: bool


def countdown_score(response: str, numbers: list[int], target: int) -> Score:
    """Require every supplied integer exactly once, with +, -, *, / only."""
    if len(response) > 65536:
        return Score(0.0, False)
    answers = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not answers or len(answers[-1]) > 1024:
        return Score(0.0, False)
    try:
        tree = ast.parse(answers[-1].strip(), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            raise ValueError("expression is too large")
        used = []

        def arithmetic(node, depth=0):
            if depth > 32:
                raise ValueError("expression is too deep")
            if isinstance(node, ast.Constant) and type(node.value) is int:
                if abs(node.value) > 1000000:
                    raise ValueError("integer is too large")
                used.append(node.value)
                return Fraction(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)
            ):
                value = arithmetic(node.operand, depth + 1)
                return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
            ):
                left = arithmetic(node.left, depth + 1)
                right = arithmetic(node.right, depth + 1)
                if isinstance(node.op, ast.Add):
                    value = left + right
                elif isinstance(node.op, ast.Sub):
                    value = left - right
                elif isinstance(node.op, ast.Mult):
                    value = left * right
                else:
                    value = left / right
                if (
                    max(value.numerator.bit_length(), value.denominator.bit_length())
                    > 4096
                ):
                    raise ValueError("fraction is too large")
                return value
            raise ValueError("unsupported arithmetic")

        value = arithmetic(tree.body)
        if Counter(used) != Counter(numbers):
            return Score(0.0, False)
        return Score(float(value == target), True)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return Score(0.0, False)


def numeric_answer(text: str) -> Fraction | None:
    """Parse a final #### or boxed number, retaining exact decimal values."""
    answers = re.findall(r"####\s*([^\n]+)|\\boxed\{([^{}]+)\}", text)
    if not answers:
        return None
    value = next(part for part in answers[-1] if part).strip().replace(",", "")
    value = value.removeprefix("$").removesuffix("$").strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+)?", value):
        return None
    if len(value) > 128:
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def gsm8k_score(response: str, answer: str) -> Score:
    expected = numeric_answer(answer)
    if expected is None:
        raise ValueError("dataset answer requires a final #### or boxed number")
    if len(response) > 65536:
        return Score(0.0, False)
    actual = numeric_answer(response)
    return Score(float(actual == expected), actual is not None)


class TaskReward(BaseRewardModel):
    """Bind the frozen, rendered prompt strings to their task records."""

    def __init__(self, records_by_prompt, task, monitor=None):
        if task not in {"countdown", "gsm8k"}:
            raise ValueError("unknown reward task")
        self.records_by_prompt = records_by_prompt
        self.task = task
        self.monitor = monitor

    def score(self, prompts, responses):
        started = time.perf_counter()
        if len(prompts) != len(responses) or not prompts:
            raise ValueError("reward batches require one response group per prompt")
        group_size = len(responses[0])
        if group_size == 0 or any(len(group) != group_size for group in responses):
            raise ValueError("reward batches require complete, equally sized groups")
        scores = []
        formats = []
        records = []
        for prompt, group in zip(prompts, responses):
            if prompt not in self.records_by_prompt:
                raise ValueError("generated prompt is absent from the frozen manifest")
            record = self.records_by_prompt[prompt]
            records.append(record)
            scored = [
                countdown_score(response, record["numbers"], record["target"])
                if self.task == "countdown"
                else gsm8k_score(response, record["answer"])
                for response in group
            ]
            scores.append([score.accuracy for score in scored])
            formats.append([score.valid_format for score in scored])
        rewards = torch.tensor(scores, dtype=torch.float32)
        if self.monitor is not None:
            self.monitor.record_rewards(
                records, responses, rewards, formats, time.perf_counter() - started
            )
        return rewards
