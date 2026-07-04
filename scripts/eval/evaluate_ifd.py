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

Disable chat template::

    python scripts/eval/ifd.py --param_path ./params \
        --input data.jsonl --output data_with_ifd.jsonl \
        --instr_key instruction --resp_key response \
        --no_chat_template
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
    use_chat_template: bool = False,
) -> dict:
    if use_chat_template:
        return _compute_ifd_with_template(
            model, tokenizer, instruction, response, device, max_len
        )
    return _compute_ifd_raw(model, tokenizer, instruction, response, device, max_len)


def _compute_ifd_raw(model, tokenizer, instruction, response, device, max_len) -> dict:
    instr_ids = tokenizer.encode(instruction, add_special_tokens=False)
    resp_ids = tokenizer.encode(response, add_special_tokens=False)

    if len(resp_ids) > max_len:
        resp_ids = resp_ids[:max_len]

    if not resp_ids:
        return {
            "L_cond": None,
            "L_uncond": None,
            "ifd": None,
            "error": "empty response",
        }

    qa_len = len(instr_ids) + len(resp_ids)
    if qa_len > max_len:
        overflow = qa_len - max_len
        if overflow >= len(instr_ids):
            resp_ids = resp_ids[:max_len]
            instr_ids = []
        else:
            instr_ids = instr_ids[overflow:]

    if not instr_ids:
        return {
            "L_cond": None,
            "L_uncond": None,
            "ifd": None,
            "error": "response too long for context",
        }

    instr_len = len(instr_ids)
    resp_len = len(resp_ids)

    qa_ids = instr_ids + resp_ids

    with torch.inference_mode():
        logits_qa = model(torch.tensor([qa_ids], device=device, dtype=torch.long))[
            "logits"
        ][0]
        logits_resp = model(torch.tensor([resp_ids], device=device, dtype=torch.long))[
            "logits"
        ][0]

    resp_logits = logits_qa[instr_len - 1 : -1]
    resp_targets = logits_resp.new_tensor(resp_ids, dtype=torch.long)
    L_cond = F.cross_entropy(resp_logits, resp_targets, reduction="mean").item()

    unp_logits = logits_resp[:-1]
    unp_targets = logits_resp.new_tensor(resp_ids[1:], dtype=torch.long)
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


def _compute_ifd_with_template(
    model, tokenizer, instruction, response, device, max_len
) -> dict:
    instr_prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    full_ids = tokenizer.encode(full_text)
    prefix_ids = tokenizer.encode(instr_prefix)
    resp_ids = tokenizer.encode(response)

    if not resp_ids:
        return {
            "L_cond": None,
            "L_uncond": None,
            "ifd": None,
            "error": "empty response",
        }

    if len(full_ids) > max_len:
        overflow = len(full_ids) - max_len
        full_ids = full_ids[overflow:]
        prefix_len = len(prefix_ids) - overflow
        prefix_len = max(0, prefix_len)
    else:
        prefix_len = len(prefix_ids)

    cond_tensor = torch.tensor([full_ids], device=device, dtype=torch.long)

    with torch.inference_mode():
        logits_qa = model(cond_tensor)["logits"][0]

    resp_start = prefix_len - 1
    resp_end = len(full_ids) - 1
    if resp_end <= resp_start:
        return {
            "L_cond": None,
            "L_uncond": None,
            "ifd": None,
            "error": "response truncated entirely",
        }

    resp_logits = logits_qa[resp_start:resp_end]
    resp_targets = torch.tensor(full_ids[prefix_len:], device=device, dtype=torch.long)
    L_cond = F.cross_entropy(resp_logits, resp_targets, reduction="mean").item()

    resp_tensor = torch.tensor([resp_ids], device=device, dtype=torch.long)

    with torch.inference_mode():
        logits_resp = model(resp_tensor)["logits"][0]

    unp_logits = logits_resp[:-1]
    unp_targets = resp_tensor[0, 1:]
    L_uncond = F.cross_entropy(unp_logits, unp_targets, reduction="mean").item()

    ifd = L_cond / L_uncond if L_uncond > 0 else None

    return {
        "L_cond": round(L_cond, 6),
        "L_uncond": round(L_uncond, 6),
        "ifd": round(ifd, 6) if ifd is not None else None,
        "instr_len": prefix_len,
        "resp_len": len(resp_ids),
        "error": None,
    }


def process_file(
    param_path: str,
    input_file: str,
    output_file: str,
    instr_key: str,
    resp_key: str,
    max_len: int = 2048,
    use_chat_template: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModel.from_pretrained(param_path)
    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model.to(device=device, dtype=dtype)
    model.eval()

    if use_chat_template and tokenizer._chat_template is None:
        raise RuntimeError(
            "--use_chat_template specified but tokenizer has no chat template. "
            "Add a chat_template to tokenizer_config.json or omit the flag."
        )

    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    results = []
    ifd_values = []

    with torch.inference_mode():
        for item in tqdm.tqdm(data, desc="Computing IFD", unit="sample"):
            instruction = item[instr_key]
            response = item[resp_key]
            scores = compute_ifd(
                model,
                tokenizer,
                instruction,
                response,
                device,
                max_len,
                use_chat_template=use_chat_template,
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
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        default=False,
        help="Disable chat template, use raw text concatenation",
    )
    args = parser.parse_args()

    process_file(
        args.param_path,
        args.input,
        args.output,
        args.instr_key,
        args.resp_key,
        args.max_len,
        use_chat_template=not args.no_chat_template,
    )


if __name__ == "__main__":
    main()
