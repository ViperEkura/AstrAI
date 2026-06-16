"""IFD (Instruction Following Difficulty) data quality scoring.

Computes IFD scores for instruction-response pairs to guide data selection.
IFD = conditional_NLL / unconditional_NLL, where:

- conditional_NLL: average CE loss on response tokens given instruction context
- unconditional_NLL: average CE loss on response tokens alone

Higher IFD (close to 1) = instruction provides less help = harder sample.
Lower IFD (close to 0) = instruction provides strong guidance = easy sample.
IFD > 1 = instruction misleads the model = likely low-quality data.

Usage::

    python scripts/eval/ifd.py --param_path ./params \
        --input data.jsonl --output data_with_ifd.jsonl \
        --instr_key instruction --resp_key response
"""

import argparse
import json

import torch
import torch.nn.functional as F
import tqdm

from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer


def compute_ifd(
    model,
    tokenizer,
    instruction: str,
    response: str,
    device: str,
    max_len: int = 2048,
) -> dict:
    instr_ids = tokenizer.encode(instruction)
    resp_ids = tokenizer.encode(response)

    if not resp_ids:
        return {
            "L_cond": None,
            "L_uncond": None,
            "ifd": None,
            "error": "empty response",
        }

    # Truncate instruction if total length exceeds max_len
    qa_len = len(instr_ids) + len(resp_ids)
    if qa_len > max_len:
        overflow = qa_len - max_len
        instr_ids = instr_ids[overflow:]

    instr_len = len(instr_ids)
    resp_len = len(resp_ids)

    # Conditional: instruction + response
    qa_ids = instr_ids + resp_ids
    qa_tensor = torch.tensor([qa_ids], device=device, dtype=torch.long)

    with torch.inference_mode():
        logits_qa = model(qa_tensor)["logits"][0]  # [qa_len, vocab]

    resp_logits = logits_qa[instr_len - 1 : -1]  # predict response tokens
    resp_targets = torch.tensor(resp_ids, device=device, dtype=torch.long)
    L_cond = F.cross_entropy(resp_logits, resp_targets, reduction="mean").item()

    # Unconditional: response alone
    resp_tensor = torch.tensor([resp_ids], device=device, dtype=torch.long)

    with torch.inference_mode():
        logits_resp = model(resp_tensor)["logits"][0]  # [resp_len, vocab]

    unp_logits = logits_resp[:-1]  # causal shift
    unp_targets = resp_tensor[0, 1:]
    L_uncond = F.cross_entropy(unp_logits, unp_targets, reduction="mean").item()

    ifd = L_cond / L_uncond if L_uncond > 0 else None

    return {
        "L_cond": round(L_cond, 6),
        "L_uncond": round(L_uncond, 6),
        "ifd": round(ifd, 6) if ifd is not None else None,
        "instr_len": instr_len,
        "resp_len": resp_len,
        "error": None,
    }


def process_file(
    param_path: str,
    input_file: str,
    output_file: str,
    instr_key: str,
    resp_key: str,
    max_len: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModel.from_pretrained(param_path)
    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model.to(device=device, dtype=dtype)
    model.eval()

    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    results = []
    ifd_values = []

    with torch.inference_mode():
        for item in tqdm.tqdm(data, desc="Computing IFD", unit="sample"):
            instruction = item[instr_key]
            response = item[resp_key]
            scores = compute_ifd(
                model, tokenizer, instruction, response, device, max_len
            )
            ifd_values.append(scores["ifd"])
            results.append({**item, "ifd": scores["ifd"], "ifd_detail": scores})

    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    valid_ifd = [v for v in ifd_values if v is not None]
    if valid_ifd:
        import statistics

        print(f"\n{'=' * 50}")
        print(f"  Samples:    {len(data)}")
        print(f"  Valid IFD:  {len(valid_ifd)}")
        print(f"  Mean IFD:   {statistics.mean(valid_ifd):.4f}")
        print(f"  Median IFD: {statistics.median(valid_ifd):.4f}")
        print(f"  Stdev IFD:  {statistics.stdev(valid_ifd):.4f}")
        print(f"  Min IFD:    {min(valid_ifd):.4f}")
        print(f"  Max IFD:    {max(valid_ifd):.4f}")
        print(f"{'=' * 50}")

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute IFD scores for instruction-response data"
    )
    parser.add_argument("--param_path", type=str, required=True, help="Model directory")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument(
        "--instr_key",
        type=str,
        default="instruction",
        help="Key for instruction field",
    )
    parser.add_argument(
        "--resp_key",
        type=str,
        default="response",
        help="Key for response field",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=2048,
        help="Max token length (instruction truncated to fit)",
    )
    args = parser.parse_args()

    process_file(
        args.param_path,
        args.input,
        args.output,
        args.instr_key,
        args.resp_key,
        args.max_len,
    )


if __name__ == "__main__":
    main()
