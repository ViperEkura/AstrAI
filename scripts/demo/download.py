import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_DIR = Path(PROJECT_ROOT, "params")
DEFAULT_REPO_ID = "ViperEkura/AstrAI-V1-instruct"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download model parameters from HuggingFace"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=DEFAULT_LOCAL_DIR,
        help=f"Local directory to save model (default: {DEFAULT_LOCAL_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if files exist",
    )
    args = parser.parse_args()

    print(f"Downloading model from {args.repo_id} to {args.local_dir}")

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        force_download=args.force,
    )

    print("Download complete!")
