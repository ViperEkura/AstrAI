"""IFD (Instruction Following Difficulty) data quality scoring.

IFD = conditional_NLL / unconditional_NLL

- Messages format: plain text concatenation (no chat template)
- Plain format: raw instr_key + resp_key fields

v2 changelog:
  - Same token set: unconditional pass prefixes resp with a plain-text sentinel
    (default ``\\n``; use ``--sentinel_text ""`` for bos/pad fallback).
    Both branches predict the identical N resp tokens.
    Single-token answers (rl=1) are now supported.
  - ctx_len tracked in output
  - skip_reason for None samples (no more silent None)
  - --per_token for per-token IFD breakdown
"""

import argparse
import glob
import json
import os
import statistics

import torch
import torch.nn.functional as F
import tqdm

from astrai.model import AutoModel
from astrai.preprocessing.packing import plan_bfd
from astrai.tokenize import AutoTokenizer


def _pack_bins(pairs, max_len):
    """BFD bin packing: pack (c+r) into bins of max total length.

    Reuses :func:`plan_bfd` so the BFD heuristic stays single-sourced.
    """
    # Treat each pair as a single sequence of length len(c)+len(r) for
    # planning purposes; plan_bfd works on pure lengths.
    fake_sequences = [[0] * (len(c) + len(r)) for c, r in pairs]
    plan = plan_bfd(fake_sequences, max_len)
    return [
        [(i, pairs[i][0], pairs[i][1]) for i in bin_indices] for bin_indices in plan
    ]


def _resolve_sentinel_ids(tokenizer, sentinel_text):
    """Tokenize the sentinel text for the unconditional pass prefix.

    Falls back to bos/pad_token_id when sentinel_text is empty or
    cannot be encoded.
    """
    if sentinel_text:
        ids = tokenizer.encode(sentinel_text, add_special_tokens=False)
        if ids:
            return ids
    for attr in ("bos_token_id", "pad_token_id", "eos_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            return [tid]
    return [0]


def _collect_input_files(input_path: str) -> list:
    """Resolve *input_path* to a list of JSONL/JSON files."""
    if os.path.isdir(input_path):
        files = []
        for ext in ("*.jsonl", "*.json"):
            files.extend(
                sorted(glob.glob(os.path.join(input_path, "**", ext), recursive=True))
            )
        return files
    return sorted(glob.glob(input_path))


def _load_items(filepath: str) -> list:
    """Load JSONL or JSON (array / single dict) into a list of dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        if filepath.lower().endswith(".json"):
            data = json.load(f)
            if isinstance(data, dict):
                return [data]
            return data
        return [json.loads(line) for line in f if line.strip()]


@torch.inference_mode()
def _score_batch(
    pairs, model, device, max_len=2048, sentinel_ids=None, per_token=False
):
    """BFD-packed IFD with text-sentinel-anchored unconditional pass.

    Conditional:   (ctx + resp[0..i-1]) → resp[i],  i = 0..N-1
    Unconditional: (<sentinel> + resp[0..i-1]) → resp[i],  i = 0..N-1

    Both branches predict the identical N response tokens.  A short
    plain-text sentinel gives the unconditional pass a prefix so that
    every response token can be predicted.  Single-token answers (rl=1)
    are supported.
    """
    if not pairs:
        return []

    if sentinel_ids is None:
        sentinel_ids = [0]

    bins = _pack_bins(pairs, max_len)
    result = [None] * len(pairs)

    # ---- conditional pass (packed, per-document position IDs) ----
    for bin_items in bins:
        seq_ids = []
        global_pos = []
        doc_ids = []
        doc_offsets = []

        for di, (orig_idx, c, r) in enumerate(bin_items):
            ctx_len = len(c)
            start = len(seq_ids)
            item_len = len(c) + len(r)
            seq_ids.extend(c)
            seq_ids.extend(r)
            end = len(seq_ids)
            global_pos.extend(range(item_len))
            doc_ids.extend([di] * item_len)
            doc_offsets.append((start, end, orig_idx, ctx_len))

        full_ids = torch.tensor([seq_ids], device=device, dtype=torch.long)
        pos_ids = torch.tensor([global_pos], device=device, dtype=torch.long)
        seq_len = len(seq_ids)
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
        doc_t = torch.tensor([doc_ids], device=device)
        doc_mask = doc_t.unsqueeze(-1) == doc_t.unsqueeze(-2)
        attn_mask = (causal & doc_mask[0]).unsqueeze(0).unsqueeze(0)
        logits_full = model(full_ids, position_ids=pos_ids, input_mask=attn_mask)[
            "logits"
        ][0]

        for start, end, orig_idx, ctx_len in doc_offsets:
            rl = end - start - ctx_len
            resp_start = start + ctx_len - 1
            resp_logits = logits_full[resp_start : end - 1]
            resp_targets = torch.tensor(
                seq_ids[start + ctx_len : end], device=device, dtype=torch.long
            )
            cond_losses = F.cross_entropy(
                resp_logits, resp_targets, reduction="none"
            ).cpu()
            result[orig_idx] = {
                "_cond_losses": cond_losses,
                "_rl": rl,
                "_ctx_len": ctx_len,
            }

    # ---- unconditional pass (sentinel-prefixed, batched 2D) ----
    valid_items = [
        (
            i,
            result[i]["_rl"],
            result[i]["_ctx_len"],
            result[i]["_cond_losses"],
            pairs[i][1],
        )
        for i in range(len(pairs))
        if result[i] is not None and "_cond_losses" in result[i]
    ]
    if not valid_items:
        return result

    valid_items.sort(key=lambda x: -x[1])
    prefix_len = len(sentinel_ids)
    max_rl = prefix_len + max(rl for _, rl, _, _, _ in valid_items)
    bsz = len(valid_items)

    u_batch = torch.zeros(bsz, max_rl, dtype=torch.long, device=device)
    for ri, (_, rl, _, _, r_ids) in enumerate(valid_items):
        u_batch[ri, :prefix_len] = torch.tensor(sentinel_ids, dtype=torch.long)
        u_batch[ri, prefix_len : prefix_len + rl] = torch.tensor(
            r_ids, dtype=torch.long
        )

    logits_resp = model(u_batch)["logits"]

    for ri, (orig_idx, rl, ctx_len, cond_losses, _) in enumerate(valid_items):
        unp_logits = logits_resp[ri, prefix_len - 1 : prefix_len - 1 + rl]
        unp_targets = u_batch[ri, prefix_len : prefix_len + rl]
        uncond_losses = F.cross_entropy(unp_logits, unp_targets, reduction="none").cpu()

        L_cond = cond_losses.mean().item()
        L_uncond = uncond_losses.mean().item()
        ifd = L_cond / L_uncond if L_uncond > 0 else None

        out = {
            "L_cond": round(L_cond, 6),
            "L_uncond": round(L_uncond, 6),
            "ifd": round(ifd, 6) if ifd is not None else None,
            "ctx_len": ctx_len,
            "resp_len": rl,
        }
        if per_token:
            per = [
                (round(c.item() / u.item(), 6) if u.item() > 0 else None)
                for c, u in zip(cond_losses, uncond_losses)
            ]
            out["ifd_per_token"] = per
        result[orig_idx] = out

    return result


def _trim(context_ids, resp_ids, max_len):
    """Truncate to fit max_len, keeping response intact if possible."""
    if len(resp_ids) > max_len // 2:
        resp_ids = resp_ids[: max_len // 2]
    full_ids = context_ids + resp_ids
    if len(full_ids) <= max_len:
        return context_ids, resp_ids
    overflow = len(full_ids) - max_len
    if overflow >= len(context_ids):
        return [], resp_ids[:max_len]
    return context_ids[overflow:], resp_ids


def process_file(
    model,
    tokenizer,
    input_file,
    output_file,
    instr_key,
    resp_key,
    max_len=2048,
    data_format="plain",
    batch_size=1,
    device=None,
    sentinel_ids=None,
    per_token=False,
    max_samples=None,
):
    """Score a single file, write per-sample JSONL, return summary stats."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if sentinel_ids is None:
        sentinel_ids = _resolve_sentinel_ids(tokenizer, "\n")

    data = _load_items(input_file)

    if max_samples and len(data) > max_samples:
        import random

        data = random.sample(data, max_samples)

    results = []
    all_ifds = []
    buffer = []

    label = os.path.splitext(os.path.basename(input_file))[0]

    for item in tqdm.tqdm(data, desc=f"  {label}", unit="sample", leave=False):
        if data_format == "messages":
            turns = []
            for i, msg in enumerate(item.get("messages", [])):
                if msg.get("role") != "assistant":
                    continue
                ctx_text = "\n\n".join(m["content"] for m in item["messages"][:i])
                ctx_ids = tokenizer.encode(ctx_text)
                resp_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
                ctx_ids, resp_ids = _trim(ctx_ids, resp_ids, max_len)
                if ctx_ids and resp_ids:
                    turns.append((ctx_ids, resp_ids))
            if not turns:
                results.append(
                    {
                        **item,
                        "ifd": None,
                        "skip_reason": "no valid assistant turns",
                        "ifd_turns": [],
                    }
                )
                continue
            buffer.append((item, turns, "messages"))
        else:
            ctx_ids = tokenizer.encode(item[instr_key], add_special_tokens=False)
            resp_ids = tokenizer.encode(item[resp_key], add_special_tokens=False)
            ctx_ids, resp_ids = _trim(ctx_ids, resp_ids, max_len)
            if not ctx_ids or not resp_ids:
                results.append(
                    {
                        **item,
                        "ifd": None,
                        "ifd_detail": {"skip_reason": "empty ctx or resp"},
                    }
                )
                continue
            buffer.append((item, [(ctx_ids, resp_ids)], "plain"))

        if len(buffer) >= batch_size:
            _flush_buffer(
                buffer,
                results,
                all_ifds,
                model,
                device,
                max_len,
                sentinel_ids,
                per_token,
            )

    if buffer:
        _flush_buffer(
            buffer, results, all_ifds, model, device, max_len, sentinel_ids, per_token
        )

    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    valid_ifd = [v for v in all_ifds if v is not None]
    stats = {
        "samples": len(data),
        "valid_ifd": len(valid_ifd),
        "skipped": len(data) - len(valid_ifd),
    }
    if valid_ifd:
        stats["mean_ifd"] = statistics.mean(valid_ifd)
        stats["median_ifd"] = statistics.median(valid_ifd)
        if len(valid_ifd) > 1:
            stats["stdev_ifd"] = statistics.stdev(valid_ifd)
        stats["min_ifd"] = min(valid_ifd)
        stats["max_ifd"] = max(valid_ifd)

        print(f"\n{'=' * 50}")
        print(f"  [{label}]")
        print(f"{'=' * 50}")
        print(f"  Samples:       {len(data)}")
        print(f"  Valid IFD:     {len(valid_ifd)}")
        print(f"  Skipped:       {len(data) - len(valid_ifd)}")
        print(f"  Mean IFD:      {statistics.mean(valid_ifd):.4f}")
        print(f"  Median IFD:    {statistics.median(valid_ifd):.4f}")
        if len(valid_ifd) > 1:
            print(f"  Stdev IFD:     {statistics.stdev(valid_ifd):.4f}")
        print(f"  Min IFD:       {min(valid_ifd):.4f}")
        print(f"  Max IFD:       {max(valid_ifd):.4f}")
        print(f"{'=' * 50}")
    print(f"  Results saved to {output_file}")
    return stats


def _flush_buffer(
    buffer, results, all_ifds, model, device, max_len, sentinel_ids, per_token
):
    all_pairs = []
    indices = []
    for item, turns, fmt in buffer:
        start = len(all_pairs)
        all_pairs.extend(turns)
        indices.append((item, turns, fmt, start, len(all_pairs)))

    raw = _score_batch(
        all_pairs,
        model,
        device,
        max_len,
        sentinel_ids=sentinel_ids,
        per_token=per_token,
    )

    for item, turns, fmt, start, end in indices:
        turn_scores = raw[start:end]
        if fmt == "messages":
            valid = [
                s for s in turn_scores if s is not None and s.get("ifd") is not None
            ]
            if not valid:
                results.append({**item, "ifd": None, "ifd_turns": turn_scores})
            else:
                avg = sum(s["ifd"] for s in valid) / len(valid)
                all_ifds.append(avg)
                results.append(
                    {
                        **item,
                        "ifd": avg,
                        "ifd_detail": valid[0] if len(valid) == 1 else None,
                        "ifd_turns": turn_scores,
                    }
                )
        else:
            score = turn_scores[0]
            all_ifds.append(score.get("ifd"))
            results.append({**item, "ifd": score.get("ifd"), "ifd_detail": score})

    buffer.clear()


def main():
    parser = argparse.ArgumentParser(
        description="Compute IFD scores for instruction-response data"
    )
    parser.add_argument("--param_path", type=str, required=True, help="Model directory")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input file, glob pattern, or directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output files (summary.json + per-file JSONL).",
    )
    parser.add_argument("--max_len", type=int, default=2048, help="Max token length")
    parser.add_argument(
        "--format",
        type=str,
        default="plain",
        choices=["plain", "messages"],
        help="Input format",
    )
    parser.add_argument(
        "--instr_key", type=str, default="instruction", help="Key for instruction field"
    )
    parser.add_argument(
        "--resp_key", type=str, default="response", help="Key for response field"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for model forward passes"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda:0)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() else "float32",
        help="Torch dtype",
    )
    parser.add_argument(
        "--sentinel_text",
        type=str,
        default="\n",
        help='Plain-text prefix for unconditional pass (default: "\\n"). Use "" for bos/pad fallback.',
    )
    parser.add_argument(
        "--per_token",
        action="store_true",
        help="Include per-token IFD breakdown in output",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples per file (random subsample). Default: all.",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    print(f"Loading model from {args.param_path} ...")
    model = AutoModel.from_pretrained(args.param_path)
    tokenizer = AutoTokenizer.from_pretrained(args.param_path)
    model.to(device=args.device, dtype=dtype)
    model.eval()

    sentinel_ids = _resolve_sentinel_ids(tokenizer, args.sentinel_text)

    input_files = _collect_input_files(args.input_path)
    if not input_files:
        print(f"No input files found at {args.input_path}")
        return

    print(f"Found {len(input_files)} file(s) to evaluate")
    os.makedirs(args.output_dir, exist_ok=True)

    all_stats = {}
    for filepath in input_files:
        label = os.path.splitext(os.path.basename(filepath))[0]
        output_file = os.path.join(args.output_dir, f"{label}_ifd.jsonl")

        stats = process_file(
            model=model,
            tokenizer=tokenizer,
            input_file=filepath,
            output_file=output_file,
            instr_key=args.instr_key,
            resp_key=args.resp_key,
            max_len=args.max_len,
            data_format=args.format,
            batch_size=args.batch_size,
            device=args.device,
            sentinel_ids=sentinel_ids,
            per_token=args.per_token,
            max_samples=args.max_samples,
        )
        all_stats[label] = stats

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
