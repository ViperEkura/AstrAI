import json
import time
from typing import Optional

import click
import torch
from tqdm import tqdm

from astrai.inference import build_engine


def processor(
    param_path: str,
    input_json_file: str,
    output_json_file: str,
    temperature: float,
    top_k: int,
    top_p: float,
    question_key: str,
    response_key: str,
    batch_size: int,
    num_samples: int = 1,
    max_seq_len: Optional[int] = None,
    frequency_penalty: float = 0.0,
    rep_window: int = 64,
):
    print(f"Loading model from {param_path} ...")
    t0 = time.time()
    engine = build_engine(
        param_path=param_path,
        max_batch_size=batch_size * num_samples,
        max_seq_len=max_seq_len,
    )
    tokenizer = engine.tokenizer
    print(f"  model loaded in {time.time() - t0:.1f}s")

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
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    rep_window=rep_window,
                )

            for i, prompt in enumerate(chunk):
                if input_data and "messages" in input_data[0]:
                    orig = input_data[chunk_start + i]
                    output_item = {**orig, response_key: resp_chunk[i]}
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


@click.command(name="generate", help="Batch generation from a JSONL prompt file.")
@click.option(
    "--param_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the model directory.",
)
@click.option(
    "--input_json_file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input JSONL file.",
)
@click.option(
    "--output_json_file",
    type=click.Path(),
    required=True,
    help="Path to the output JSONL file.",
)
@click.option(
    "--question_key", default="question", help="Key for the question in input JSON."
)
@click.option(
    "--response_key", default="response", help="Key for the response in output JSON."
)
@click.option("--temperature", type=float, default=0.8, help="Sampling temperature.")
@click.option("--top_k", type=int, default=50, help="Top-k filtering.")
@click.option("--top_p", type=float, default=0.95, help="Top-p filtering.")
@click.option("--batch_size", type=int, default=1, help="Batch size.")
@click.option("--num_samples", type=int, default=1, help="Responses per prompt.")
@click.option("--max_seq_len", type=int, default=2048, help="KV cache length.")
@click.option("--frequency_penalty", type=float, default=0.0, help="Frequency penalty.")
@click.option(
    "--rep_window", type=int, default=64, help="Window size for frequency penalty."
)
def generate_command(**kwargs):
    """Batch generation from a JSONL prompt file."""
    with torch.inference_mode():
        processor(**kwargs)


if __name__ == "__main__":
    generate_command()
