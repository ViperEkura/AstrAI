"""CLI: JSONL → tokenized .h5/.bin via config-driven Pipeline."""

import argparse

from astrai.config.preprocess_config import PipelineConfig
from astrai.preprocessing.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Raw JSONL → tokenized .h5/.bin via config-driven Pipeline"
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="JSONL", help="One or more JSONL files"
    )
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to pipeline config JSON"
    )
    parser.add_argument(
        "--tokenizer_path",
        default="params",
        help="Path to tokenizer directory (default: params)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of records tokenized together (default: config value)",
    )
    args = parser.parse_args()

    config = PipelineConfig.from_file(args.config)
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch_size must be at least 1")
        config.preprocessing.batch_size = args.batch_size

    Pipeline(
        config=config,
        input_paths=args.inputs,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
    ).run()


if __name__ == "__main__":
    main()
