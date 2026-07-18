import argparse
import json
import time
from typing import Optional

import torch
from tqdm import tqdm

from astrai.inference import InferenceEngine
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer


def processor(
    param_path: str,
    input_json_file: str,
    output_json_file: str,
    temperature: float,
    top_k: int,
    top_p: float,
    question_key: str,
    response_key: str,
    max_tokens: Optional[int],
    batch_size: int,
    num_samples: int = 1,
    cache_len: int = 2048,
    frequency_penalty: float = 0.0,
    rep_window: int = 64,
):
    print(f"Loading model from {param_path} ...")
    t0 = time.time()
    model = AutoModel.from_pretrained(param_path)
    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model.to(device="cuda", dtype=torch.bfloat16)
    print(f"  model loaded in {time.time() - t0:.1f}s")

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=batch_size * num_samples,
        max_seq_len=cache_len,
        max_prompt_len=cache_len,
    )

    print(f"Reading {input_json_file} ...")
    with open(input_json_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f]

    if input_data and "messages" in input_data[0]:
        prompts = [
            tokenizer.apply_chat_template(item["messages"], tokenize=False)
            for item in input_data
        ]
    else:
        prompts = [item[question_key] for item in input_data]
    print(f"  {len(prompts)} prompts loaded\n")

    if max_tokens is None:
        max_tokens = model.config.max_len

    chunk_size = max(1, batch_size)

    with open(output_json_file, "w", encoding="utf-8") as f:
        pbar = tqdm(
            total=len(prompts) * num_samples,
            unit="gen",
            desc=f" Generating ({num_samples}x/prompt)",
        )
        for chunk_start in range(0, len(prompts), chunk_size):
            chunk = prompts[chunk_start : chunk_start + chunk_size]

            if num_samples > 1:
                chunk_expanded = [p for p in chunk for _ in range(num_samples)]
                resp_chunk = engine.generate(
                    prompt=chunk_expanded,
                    stream=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    rep_window=rep_window,
                )
                resp_chunk = [
                    resp_chunk[i * num_samples : (i + 1) * num_samples]
                    for i in range(len(chunk))
                ]
            else:
                resp_chunk = engine.generate(
                    prompt=chunk,
                    stream=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    rep_window=rep_window,
                )

            for i, prompt in enumerate(chunk):
                if input_data and "messages" in input_data[0]:
                    output_item = {"response": resp_chunk[i]}
                else:
                    output_item = {
                        question_key: prompt,
                        response_key: resp_chunk[i],
                    }
                f.write(json.dumps(output_item, ensure_ascii=False) + "\n")

            pbar.update(len(chunk) * num_samples)

        pbar.close()

    elapsed = time.time() - t0
    print(
        f"\nDone! {len(prompts)} prompts x {num_samples} samples -> {output_json_file}"
    )
    print(f"Total time: {elapsed:.1f}s ({elapsed / len(prompts):.2f}s/prompt)")

    engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generation from JSONL file.")

    parser.add_argument(
        "--param_path", type=str, required=True, help="Path to the model directory."
    )
    parser.add_argument(
        "--input_json_file",
        type=str,
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output_json_file",
        type=str,
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--question_key",
        type=str,
        default="question",
        help="Key for the question in the input JSON (default: question).",
    )
    parser.add_argument(
        "--response_key",
        type=str,
        default="response",
        help="Key for the response in the output JSON (default: response).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.60,
        help="Temperature for generating responses (default: 0.60).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Top-k value for generating responses (default: 30).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p value for generating responses (default: 0.95).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generating responses (default: 1).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of responses per prompt (expands batch internally, default: 1).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: model config max_len).",
    )
    parser.add_argument(
        "--cache_len",
        type=int,
        default=2048,
        help="KV cache & prompt truncation length (default: 2048, lower = less memory).",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="Frequency penalty to reduce repetition (default: 0.0, try 0.5-1.0).",
    )
    parser.add_argument(
        "--rep_window",
        type=int,
        default=64,
        help="Window size for frequency penalty (default: 64).",
    )

    args = parser.parse_args()

    with torch.inference_mode():
        processor(**vars(args))
