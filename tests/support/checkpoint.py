import json
import os

import torch


def find_checkpoint_meta(ckpt_dir):
    """Walk *ckpt_dir* and return the path to the first ``meta.json`` found."""
    for root, _dirs, files in os.walk(ckpt_dir):
        if "meta.json" in files:
            return os.path.join(root, "meta.json")
    return None


def load_checkpoint_meta(ckpt_dir):
    """Find and load the first checkpoint ``meta.json`` under *ckpt_dir*."""
    meta_path = find_checkpoint_meta(ckpt_dir)
    assert meta_path is not None, f"No checkpoint meta.json found in {ckpt_dir}"
    with open(meta_path) as f:
        return json.load(f)


def load_shard_meta(out_dir):
    """Load ``meta.json`` from the default shard output directory."""
    meta_path = os.path.join(out_dir, "__default__", "shard_0000", "meta.json")
    assert os.path.exists(meta_path), f"Shard meta.json not found at {meta_path}"
    with open(meta_path) as f:
        return json.load(f)


def assert_state_dicts_equal(a, b):
    """Assert two state dicts have identical keys and equal tensor values."""
    assert set(a.keys()) == set(b.keys()), f"Key mismatch: {set(a) ^ set(b)}"
    for key in a:
        assert torch.equal(a[key], b[key]), f"Tensor mismatch at key: {key}"
