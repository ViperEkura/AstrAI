import argparse
import glob
import json
import os
import statistics
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import tqdm

from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer


def _collect_input_files(input_path: str) -> List[str]:
    """Resolve *input_path* to a list of JSONL/JSON files."""
    if os.path.isdir(input_path):
        files = []
        for ext in ("*.jsonl", "*.json"):
            files.extend(
                sorted(glob.glob(os.path.join(input_path, "**", ext), recursive=True))
            )
        return files
    return sorted(glob.glob(input_path))


def _load_items(filepath: str) -> List[dict]:
    """Load JSONL or JSON (array / single dict) into a list of dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        if filepath.lower().endswith(".json"):
            data = json.load(f)
            if isinstance(data, dict):
                return [data]
            return data
        return [json.loads(line) for line in f if line.strip()]


def _encode_batch(
    tokenizer: AutoTokenizer, texts: List[str], max_length: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """Encode *texts* and return (token_ids, attention_masks).

    Each sequence is left-aligned and padded to the batch max length.
    """
    encoded = [tokenizer.encode(t)[:max_length] for t in texts]
    if not encoded:
        return [], []
    max_len = max(len(seq) for seq in encoded)
    padded_ids = []
    masks = []
    for seq in encoded:
        pad_len = max_len - len(seq)
        padded_ids.append(seq + [tokenizer.pad_id] * pad_len)
        masks.append([1] * len(seq) + [0] * pad_len)
    return padded_ids, masks


def _compute_batch(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass and return (log_probs, valid_mask) of shape [B, S-1].

    log_probs[i, j] = log P(token j+1 | tokens 0..j)
    """
    output = model(input_ids, input_mask=attention_mask)
    logits = output["logits"][:, :-1, :]  # [B, S-1, V]
    targets = input_ids[:, 1:]  # [B, S-1]
    valid = attention_mask[:, 1:].float()  # [B, S-1]

    log_probs = F.log_softmax(logits.float(), dim=-1)  # [B, S-1, V]
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # [B, S-1]

    return token_log_probs, valid


def _token_type(token_id: int, stop_ids: frozenset, decode_fn) -> str:
    """Classify a token into a coarse type for analysis.

    *stop_ids* is a pre-built set of special token IDs.
    *decode_fn* is ``tokenizer.decode`` (or a wrapper) for single-token
    decoding.
    """
    if token_id in stop_ids:
        return "special"
    decoded = decode_fn([token_id], skip_special_tokens=True)
    if any("\u4e00" <= ch <= "\u9fff" for ch in decoded):
        return "cjk"
    if any(ord(ch) > 127 for ch in decoded):
        return "non_ascii"
    return "ascii"


def _percentiles(values: List[float]) -> Dict[str, float]:
    """Compute common percentiles from a list of floats.

    Uses linear interpolation between closest ranks (same convention
    as NumPy's default).
    """
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _pct(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        k = p * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

    return {
        "p50": _pct(0.50),
        "p90": _pct(0.90),
        "p95": _pct(0.95),
        "p99": _pct(0.99),
    }


class LossAccumulator:
    """Accumulate per-token losses with optional streaming mode.

    When *stream* is True (token_level=False), losses are not kept
    in memory individually — only a running sum/count and a histogram
    (for approximate percentiles) are maintained.  When *stream* is
    False, all losses are retained for exact statistics and per-record
    output.
    """

    _HIST_BINS = 1000
    _HIST_MAX = 20.0  # clamp losses above this for histogram

    def __init__(self, stream: bool):
        self.stream = stream
        self.losses: List[float] = [] if not stream else []
        self.total: float = 0.0
        self.count: int = 0
        self.hist = torch.zeros(self._HIST_BINS, dtype=torch.long)
        # per-type losses (only populated when not streaming)
        self.by_type: Dict[str, List[float]] = {}
        self.type_total: Dict[str, float] = {}
        self.type_count: Dict[str, int] = {}

    def add(self, losses: List[float]):
        self.total += sum(losses)
        self.count += len(losses)
        if self.stream:
            clamped = [min(max(l, 0.0), self._HIST_MAX) for l in losses]
            idx = torch.tensor(clamped) / self._HIST_MAX * (self._HIST_BINS - 1)
            self.hist += torch.bincount(
                idx.long().clamp(0, self._HIST_BINS - 1),
                minlength=self._HIST_BINS,
            )
        else:
            self.losses.extend(losses)

    def add_typed(self, ttype: str, losses: List[float]):
        if not self.stream:
            self.by_type.setdefault(ttype, []).extend(losses)
        self.type_total[ttype] = self.type_total.get(ttype, 0.0) + sum(losses)
        self.type_count[ttype] = self.type_count.get(ttype, 0) + len(losses)

    def stats(self) -> Dict:
        result: Dict = {}
        if self.count == 0:
            return result
        mean_loss = self.total / self.count
        result["overall"] = {
            "num_tokens": self.count,
            "mean_loss": mean_loss,
            "ppl": float(torch.exp(torch.tensor(mean_loss))),
        }
        if self.stream:
            result["overall"].update(self._hist_percentiles())
        else:
            result["overall"]["median_loss"] = statistics.median(self.losses)
            result["overall"].update(_percentiles(self.losses))

        if self.type_count:
            result["by_token_type"] = {}
            for ttype in sorted(self.type_count.keys()):
                cnt = self.type_count[ttype]
                tmean = self.type_total[ttype] / cnt
                entry: Dict = {
                    "num_tokens": cnt,
                    "mean_loss": tmean,
                    "ppl": float(torch.exp(torch.tensor(tmean))),
                }
                if not self.stream and ttype in self.by_type:
                    entry["median_loss"] = statistics.median(self.by_type[ttype])
                    entry.update(_percentiles(self.by_type[ttype]))
                result["by_token_type"][ttype] = entry
        return result

    def _hist_percentiles(self) -> Dict[str, float]:
        """Approximate percentiles from the histogram."""
        total = self.hist.sum().item()
        if total == 0:
            return {}
        cum = torch.cumsum(self.hist.float(), dim=0)
        result = {}
        for label, p in [("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)]:
            target = p * total
            idx = int(torch.searchsorted(cum, target).item())
            idx = min(idx, self._HIST_BINS - 1)
            result[label] = (idx + 0.5) / self._HIST_BINS * self._HIST_MAX
        return result


def process_file(
    model,
    tokenizer: AutoTokenizer,
    items: List[dict],
    text_key: str,
    batch_size: int,
    max_length: int,
    token_level: bool,
    max_samples: Optional[int],
    output_file: Optional[str],
    label: str,
) -> Dict:
    """Evaluate a single dataset (list of items), return summary stats.

    If *token_level* is True and *output_file* is set, per-record token_ids
    and log_probs are written as JSONL alongside the summary.
    """
    if max_samples and len(items) > max_samples:
        import random

        items = random.sample(items, max_samples)

    texts = [item[text_key] for item in items if text_key in item]
    print(f"  [{label}] {len(texts)} samples, text_key='{text_key}'")

    acc = LossAccumulator(stream=not token_level)
    per_sample: List[dict] = []

    if token_level:
        stop_ids = frozenset(tokenizer.stop_ids)
        decode_fn = tokenizer.decode

    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm.tqdm(
        range(0, len(texts), batch_size),
        total=num_batches,
        desc=f"  {label}",
        leave=False,
    ):
        batch_texts = texts[i : i + batch_size]
        padded_ids, masks = _encode_batch(tokenizer, batch_texts, max_length)

        input_ids = torch.tensor(padded_ids, device="cuda", dtype=torch.long)
        attention_mask = torch.tensor(masks, device="cuda", dtype=torch.bool)

        token_log_probs, valid = _compute_batch(model, input_ids, attention_mask)

        for b in range(len(batch_texts)):
            seq_len = int(valid[b].sum().item())
            lps = token_log_probs[b, :seq_len].tolist()
            losses = [-lp for lp in lps]
            acc.add(losses)

            if token_level:
                # log_probs correspond to positions 1..seq_len (predicted
                # from position 0..seq_len-1), so token_ids must skip BOS
                # at position 0 to stay aligned with log_probs.
                ids = padded_ids[b][1 : seq_len + 1]
                per_sample.append(
                    {
                        "text": batch_texts[b][:200],
                        "token_ids": ids,
                        "log_probs": [round(lp, 4) for lp in lps],
                        "ppl": float(torch.exp(torch.tensor(statistics.mean(losses))))
                        if losses
                        else None,
                    }
                )
                typed_losses: Dict[str, List[float]] = {}
                for tid, loss in zip(ids, losses):
                    ttype = _token_type(tid, stop_ids, decode_fn)
                    typed_losses.setdefault(ttype, []).append(loss)
                for ttype, tl in typed_losses.items():
                    acc.add_typed(ttype, tl)

    stats = acc.stats()

    if token_level and output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in per_sample:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return stats


def print_stats(label: str, stats: Dict):
    """Pretty-print summary statistics."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    ov = stats.get("overall", {})
    if ov:
        print(f"  tokens:       {ov['num_tokens']:,}")
        print(f"  mean loss:    {ov['mean_loss']:.4f}")
        if "median_loss" in ov:
            print(f"  median loss:  {ov['median_loss']:.4f}")
        print(f"  ppl:          {ov['ppl']:.2f}")
        if "p50" in ov:
            print(
                f"  p50/p90/p95/p99: "
                f"{ov['p50']:.2f} / {ov['p90']:.2f} / {ov['p95']:.2f} / {ov['p99']:.2f}"
            )
    by_type = stats.get("by_token_type", {})
    if by_type:
        print(f"\n  by token type:")
        print(f"  {'type':<12} {'count':>8} {'mean_loss':>10} {'ppl':>8}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 10} {'-' * 8}")
        for ttype, s in by_type.items():
            print(
                f"  {ttype:<12} {s['num_tokens']:>8,} "
                f"{s['mean_loss']:>10.4f} {s['ppl']:>8.2f}"
            )


def main(
    param_path: str,
    input_path: str,
    output_dir: str,
    text_key: str,
    batch_size: int,
    max_length: int,
    token_level: bool,
    max_samples: Optional[int],
):
    print(f"Loading model from {param_path} ...")
    model = AutoModel.from_pretrained(param_path)
    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()

    input_files = _collect_input_files(input_path)
    if not input_files:
        print(f"No input files found at {input_path}")
        return

    print(f"Found {len(input_files)} file(s) to evaluate")
    os.makedirs(output_dir, exist_ok=True)

    all_stats = {}
    for filepath in input_files:
        label = os.path.splitext(os.path.basename(filepath))[0]
        items = _load_items(filepath)
        if not items:
            print(f"  [{label}] empty, skipping")
            continue

        token_output = (
            os.path.join(output_dir, f"{label}_tokens.jsonl") if token_level else None
        )

        stats = process_file(
            model=model,
            tokenizer=tokenizer,
            items=items,
            text_key=text_key,
            batch_size=batch_size,
            max_length=max_length,
            token_level=token_level,
            max_samples=max_samples,
            output_file=token_output,
            label=label,
        )
        all_stats[label] = stats
        print_stats(label, stats)

        if token_output:
            print(f"  token-level output: {token_output}")

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perplexity and token-level loss evaluation on JSONL/JSON data."
    )
    parser.add_argument(
        "--param_path", type=str, required=True, help="Path to the model directory."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input file, glob pattern, or directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output files (summary.json + per-file token JSONL).",
    )
    parser.add_argument(
        "--text_key",
        type=str,
        default="text",
        help="Key for the text field in the input data.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (tokens). Longer sequences are truncated.",
    )
    parser.add_argument(
        "--token_level",
        action="store_true",
        help="Store per-token log_probs and token type analysis. "
        "Default: off (only aggregate stats).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples per file (random subsample). Default: all.",
    )
    args = parser.parse_args()

    with torch.inference_mode():
        main(
            param_path=args.param_path,
            input_path=args.input_path,
            output_dir=args.output_dir,
            text_key=args.text_key,
            batch_size=args.batch_size,
            max_length=args.max_length,
            token_level=args.token_level,
            max_samples=args.max_samples,
        )
