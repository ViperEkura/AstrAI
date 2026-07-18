from argparse import ArgumentParser
from pathlib import Path

import torch

from astrai.inference import InferenceEngine
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = ArgumentParser(description="Interactive streaming chat")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=PROJECT_ROOT / "params",
        help="Path to model weights (params/ or checkpoint/epoch_N_step_M/)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling threshold",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling threshold",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.5,
        help="Penalty per occurrence for repeated tokens (0.0 disables, "
        "range -2.0~2.0, typical 0.3-1.0)",
    )
    parser.add_argument(
        "--rep_window",
        type=int,
        default=64,
        help="Number of recent prompt tokens to include in penalty history",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="Optional system prompt (default: empty, model not SFT-trained on system role)",
    )
    return parser.parse_args()


def chat():
    args = parse_args()
    model_path = args.model_path

    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device="cuda", dtype=torch.bfloat16)
    engine = InferenceEngine(model=model, tokenizer=tokenizer)

    while True:
        query = input(">> ")
        if query == "!exit":
            break

        msgs = []
        if args.system_prompt:
            msgs.append({"role": "system", "content": args.system_prompt})
        msgs.append({"role": "user", "content": query})
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

        full_response = ""
        for token in engine.generate(
            prompt=prompt,
            stream=True,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            frequency_penalty=args.frequency_penalty,
            rep_window=args.rep_window,
        ):
            print(token, end="", flush=True)
            full_response += token

        print()


if __name__ == "__main__":
    chat()
