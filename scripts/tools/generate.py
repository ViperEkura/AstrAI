import argparse
import json
from typing import Optional

import torch

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
):
    # Load model and tokenizer
    model = AutoModel.from_pretrained(param_path)
    tokenizer = AutoTokenizer.from_pretrained(param_path)
    model.to(device="cuda", dtype=torch.bfloat16)

    # Create inference engine
    engine = InferenceEngine(
        model=model, tokenizer=tokenizer, max_batch_size=batch_size
    )

    with open(input_json_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f]

    # Check input format: chat messages or raw text
    if input_data and "messages" in input_data[0]:
        # Chat format: [{"messages": [...]}]
        prompts = [
            tokenizer.apply_chat_template(item["messages"], tokenize=False)
            for item in input_data
        ]
    else:
        # Raw text format: [{"question": "..."}]
        prompts = [item[question_key] for item in input_data]

    # Use provided max_tokens or default to model config max_len
    if max_tokens is None:
        max_tokens = model.config.max_len

    # Generate responses (batch)
    responses = engine.generate(
        prompt=prompts,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    # Write results
    with open(output_json_file, "w", encoding="utf-8") as f:
        for prompt, response in zip(prompts, responses):
            if input_data and "messages" in input_data[0]:
                output_item = {"response": response}
            else:
                output_item = {question_key: prompt, response_key: response}
            f.write(json.dumps(output_item, ensure_ascii=False) + "\n")

    # Cleanup
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
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: model config max_len).",
    )

    args = parser.parse_args()

    with torch.inference_mode():
        processor(**vars(args))
