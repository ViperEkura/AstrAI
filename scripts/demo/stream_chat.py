from argparse import ArgumentParser
from contextlib import closing
from pathlib import Path

from astrai import build_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = ArgumentParser(description="Streaming chat or raw text continuation")
    parser.add_argument("--prompt", help="Generate once; omit for interactive input")
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
        default=None,
        help="Penalty per occurrence for repeated tokens (0.0 disables, "
        "default: 0.5 for chat, 0.0 for raw continuation)",
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
    parser.add_argument(
        "--raw", action="store_true", help="Continue text without a chat template"
    )
    args = parser.parse_args()
    if args.raw and args.system_prompt:
        parser.error("--system_prompt requires chat mode")
    if args.frequency_penalty is None:
        args.frequency_penalty = 0.0 if args.raw else 0.5
    return args


def chat():
    args = parse_args()
    model_path = args.model_path

    engine = build_engine(model_path)
    tokenizer = engine.tokenizer
    msgs = []
    if args.system_prompt:
        msgs.append({"role": "system", "content": args.system_prompt})

    try:
        while True:
            try:
                query = args.prompt if args.prompt is not None else input(">> ")

                if args.raw:
                    prompt = query
                    print(prompt, end="", flush=True)
                else:
                    prompt = tokenizer.apply_chat_template(
                        msgs + [{"role": "user", "content": query}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                with closing(
                    engine.generate(
                        prompt=prompt,
                        stream=True,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        frequency_penalty=args.frequency_penalty,
                        rep_window=args.rep_window,
                    )
                ) as stream:
                    for token in stream:
                        print(token, end="", flush=True)
                print()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            except Exception as exc:
                if args.prompt is not None:
                    raise
                print(f"\n{type(exc).__name__}: {exc}")
            if args.prompt is not None:
                break
    finally:
        engine.shutdown()


if __name__ == "__main__":
    chat()
