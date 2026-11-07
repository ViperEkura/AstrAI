import json
import os


def load_checkpoint_meta(ckpt_dir):
    """Load the first checkpoint metadata found under the given directory."""
    for root, _dirs, files in os.walk(ckpt_dir):
        if "meta.json" in files:
            with open(os.path.join(root, "meta.json")) as file:
                return json.load(file)
    raise AssertionError(f"No checkpoint meta.json found in {ckpt_dir}")
