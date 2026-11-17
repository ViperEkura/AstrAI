"""Model scoring, code sampling and resource-limited execution helpers."""

import contextlib
import multiprocessing
import os
import signal
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from ctypes import c_byte
from math import ceil
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import tqdm

from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer

if TYPE_CHECKING:
    from astrai.inference import InferenceEngine

if sys.platform == "linux":
    import resource


def load_score_model(
    param_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    model = AutoModel.from_pretrained(param_path)
    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model.to(device=device, dtype=getattr(torch, dtype))
    model.eval()
    return model, tokenizer


def causal_sequence_logits(
    model: torch.nn.Module,
    rows: List[List[int]],
    device: str,
    position_ids: Optional[List[List[int]]] = None,
    group_ids: Optional[List[List[int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return logits [B, S, V] and valid-token mask [B, S].

    Ragged rows are right-padded. Attention permits only earlier/equal
    positions within the same document; optional group IDs define documents.
    Logits at position p predict the token at p + 1.
    """
    n = len(rows)
    max_len = max(len(r) for r in rows)
    ids = torch.zeros(n, max_len, dtype=torch.long, device=device)
    key_pad = torch.zeros(n, 1, 1, max_len, dtype=torch.bool, device=device)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r, dtype=torch.long, device=device)
        key_pad[i, 0, 0, : len(r)] = True

    pos = None
    if position_ids is not None:
        pos = torch.zeros(n, max_len, dtype=torch.long, device=device)
        for i, p in enumerate(position_ids):
            pos[i, : len(p)] = torch.tensor(p, dtype=torch.long, device=device)

    causal = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool, device=device))
    mask = key_pad & causal
    if group_ids is not None:
        doc = torch.full((n, max_len), -1, dtype=torch.long, device=device)
        for i, g in enumerate(group_ids):
            doc[i, : len(g)] = torch.tensor(g, dtype=torch.long, device=device)
        mask = mask & (doc[:, None, :, None] == doc[:, None, None, :])

    with torch.inference_mode():
        logits = model(ids, position_ids=pos, input_mask=mask)["logits"]
    return logits, key_pad.squeeze(1).squeeze(1)


def loglikelihood_batched(
    model: torch.nn.Module,
    requests: List[Tuple[List[int], List[int]]],
    device: str,
    max_model_len: int,
) -> List[float]:
    """Summed log-probability of each continuation given its context.

    requests: (ctx_ids, cont_ids) pairs; token sequences are concatenated as
    ``ctx_ids + cont_ids`` — callers must ensure the tokenization matches how
    the model saw such text in training (see lm-eval's _encode_pair caveat).
    """
    rows: List[List[int]] = []
    starts: List[int] = []
    for ctx_ids, cont_ids in requests:
        input_ids = ctx_ids + cont_ids
        if len(input_ids) > max_model_len:
            input_ids = input_ids[len(input_ids) - max_model_len :]
        rows.append(input_ids)
        starts.append(len(input_ids) - len(cont_ids))

    logits, _ = causal_sequence_logits(model, rows, device)

    scores = [0.0] * len(requests)
    for i, (ids, start) in enumerate(zip(rows, starts)):
        score = 0.0
        for j in range(len(ids) - start):
            pos = start - 1 + j
            if pos < 0 or pos >= logits.size(1):
                break
            score += torch.nn.functional.log_softmax(logits[i, pos].float(), dim=-1)[
                ids[start + j]
            ].item()
        scores[i] = score
    return scores


def deduplicate(seq: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(seq))


def generate_batch(
    engine: "InferenceEngine",
    prompt: str,
    n: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    """Draw n completions for one prompt, deduplicated."""
    completions: List[str] = []
    remaining = n
    while remaining > 0:
        current = min(batch_size, remaining)
        outputs = engine.generate(
            prompt=[prompt] * current,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=top_p,
            top_k=top_k,
        )
        completions.extend(outputs if isinstance(outputs, list) else [outputs])
        remaining -= current
    return deduplicate(completions)


def _execute_worker(code: str, timeout: float, directory: str, passed: c_byte) -> None:
    os.setsid()
    os.chdir(directory)
    with open("/proc/self/statm") as memory:
        baseline = int(memory.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    limits = (
        (resource.RLIMIT_AS, baseline + 512 * 1024**2),
        (resource.RLIMIT_FSIZE, 16 * 1024**2),
        (resource.RLIMIT_CPU, max(1, ceil(timeout))),
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_NOFILE, 64),
    )
    for kind, limit in limits:
        hard = resource.getrlimit(kind)[1]
        limit = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(kind, (limit, limit))
    with open(os.devnull, "w") as output:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                exec(code, {})
            except BaseException:
                return
    passed.value = True


def execute_one(args: Tuple[str, float]) -> bool:
    """Fork a worker with 512 MiB additional address space and 16 MiB/file limits."""
    full_code, timeout = args
    context = multiprocessing.get_context("fork")
    passed = context.Value("b", False, lock=False)
    with tempfile.TemporaryDirectory(prefix="eval-") as directory:
        proc = context.Process(
            target=_execute_worker, args=(full_code, timeout, directory, passed)
        )
        proc.start()
        try:
            proc.join(timeout)
            return proc.exitcode == 0 and bool(passed.value)
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if proc.is_alive():
                proc.kill()
            proc.join()
            proc.close()


def test_all(
    items: Sequence[Dict],
    codes_for: Callable[[Dict], Tuple[str, Sequence[Tuple[str, float]]]],
    test_workers: int,
    desc: str = "Testing",
) -> List[Tuple[str, int, int]]:
    """Execute generated code for each item in a process pool.

    codes_for(item) -> (task_id, [(full_code, timeout), ...]); returns
    (task_id, n, passed) per item.
    """
    results: List[Tuple[str, int, int]] = []
    with ProcessPoolExecutor(max_workers=test_workers) as pool:
        for item in tqdm.tqdm(items, desc=desc, unit="problem"):
            task_id, codes = codes_for(item)
            passed = sum(1 for ok in pool.map(execute_one, codes) if ok)
            results.append((task_id, len(codes), passed))
    return results
