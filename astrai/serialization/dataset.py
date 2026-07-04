"""Dataset storage serialization helpers (HDF5 / memory-mapped binary)."""

import json
import os
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from torch import Tensor


def save_h5(file_path: str, file_name: str, tensor_group: Dict[str, List[Tensor]]):
    os.makedirs(file_path, exist_ok=True)
    full_file_path = os.path.join(file_path, f"{file_name}.h5")
    with h5py.File(full_file_path, "w") as f:
        for key, tensors in tensor_group.items():
            grp = f.create_group(key)
            for idx, tensor in enumerate(tensors):
                arr = tensor.cpu().numpy()
                grp.create_dataset(f"data_{idx}", data=arr)


def load_h5(file_path: str, share_memory=True) -> Dict[str, List[Tensor]]:
    tensor_group: Dict[str, List[Tensor]] = {}

    root_path = Path(file_path)
    h5_files = list(root_path.rglob("*.h5")) + list(root_path.rglob("*.hdf5"))

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            for key in f.keys():
                grp = f[key]
                dsets = []
                for dset_name in grp.keys():
                    dset = grp[dset_name]
                    tensor = torch.from_numpy(dset[:])
                    if share_memory:
                        tensor = tensor.share_memory_()
                    dsets.append(tensor)

                if tensor_group.get(key) is None:
                    tensor_group[key] = []
                tensor_group[key].extend(dsets)

    return tensor_group


def save_bin(file_path: str, tensor_group: Dict[str, List[Tensor]]):
    os.makedirs(file_path, exist_ok=True)
    meta = {}
    for key, tensors in tensor_group.items():
        cat = torch.cat(tensors, dim=0)
        meta[key] = {"shape": list(cat.shape), "dtype": str(cat.dtype).split(".")[-1]}
        np.asarray(cat.cpu().numpy()).tofile(os.path.join(file_path, f"{key}.bin"))
    with open(os.path.join(file_path, "meta.json"), "w") as f:
        json.dump(meta, f)


def load_bin(file_path: str) -> Dict[str, List[Tensor]]:
    with open(os.path.join(file_path, "meta.json"), "r") as f:
        meta = json.load(f)
    segments: Dict[str, List[Tensor]] = {}
    for key, info in meta.items():
        arr = np.memmap(
            os.path.join(file_path, f"{key}.bin"),
            dtype=info["dtype"],
            mode="r+",
            shape=tuple(info["shape"]),
        )
        segments[key] = [torch.from_numpy(arr)]
    return segments
