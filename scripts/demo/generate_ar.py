from argparse import ArgumentParser
from pathlib import Path

from astrai.inference import build_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_ROOT = Path(PROJECT_ROOT, "params")


def parse_args():
    parser = ArgumentParser(
        description="Autoregressive continuation demo: continue a prompt, "
        "or drop into interactive mode without one"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prefix to continue; omit to enter interactive mode",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=PARAMETER_ROOT,
        help="Path to model weights",
    )
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    return parser.parse_args()


def run_once(engine, prompt, args):
    print(prompt, end="", flush=True)
    for token in engine.generate(
        prompt=prompt,
        stream=True,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    ):
        print(token, end="", flush=True)
    print()


def main():
    args = parse_args()
    engine = build_engine(args.model_path)

    if args.prompt is not None:
        run_once(engine, args.prompt, args)
        return

    while True:
        try:
            query = input(">> ")
        except EOFError:
            break
        if query == "!exit":
            break
        if query:
            run_once(engine, query, args)


if __name__ == "__main__":
    main()
