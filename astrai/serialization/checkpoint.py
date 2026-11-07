"""Model checkpoint serialization helpers."""

import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import safetensors.torch as st
import torch
import torch.distributed as dist

from astrai.parallel.setup import get_rank

logger = logging.getLogger(__name__)

_META_FILE = "meta.json"
_CONFIG_FILE = "config.json"
_WEIGHTS_FILE = "model.safetensors"
_MANIFEST_FILE = "manifest.json"
#: v2 dropped the per-file checksum: the atomic directory publish already
#: guarantees "visible means complete", and hashing 8.5 GB cost ~45 s per save.
_CHECKPOINT_FORMAT_VERSION = 2
_SUPPORTED_FORMAT_VERSIONS = (1, 2)


def save_safetensors(state_dict: Dict, path: Union[str, Path]):
    st.save_file(state_dict, str(path))


def _broadcast_load(loader: Callable[[], Dict], broadcast: bool) -> Dict:
    """Load on rank 0 and broadcast the object to all ranks."""
    if not broadcast or not dist.is_initialized():
        return loader()
    rank = get_rank()
    if rank == 0:
        data = loader()
    else:
        data = {}
    tmp = [data]
    dist.broadcast_object_list(tmp, src=0)
    return tmp[0]


def load_safetensors(path: Union[str, Path], broadcast: bool = False) -> Dict:
    return _broadcast_load(lambda: st.load_file(str(path)), broadcast)


def save_json(data: Dict, path: Union[str, Path]):
    with open(str(path), "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Union[str, Path], broadcast: bool = False) -> Dict:
    return _broadcast_load(lambda: json.loads(Path(path).read_text()), broadcast)


def save_torch(obj: Any, path: Union[str, Path]):
    torch.save(obj, str(path))


def load_torch(path: Union[str, Path], broadcast: bool = False) -> Any:
    if not broadcast or not dist.is_initialized():
        return torch.load(str(path), map_location="cpu", weights_only=False)

    path = Path(path)
    rank = get_rank()

    if rank == 0:
        with open(path, "rb") as f:
            raw = f.read()
        data_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        num_bytes = torch.tensor([len(raw)], dtype=torch.long)
    else:
        num_bytes = torch.tensor([0], dtype=torch.long)

    dist.broadcast(num_bytes, src=0)

    if rank != 0:
        data_tensor = torch.empty(num_bytes.item(), dtype=torch.uint8)

    dist.broadcast(data_tensor, src=0)

    buf = io.BytesIO(data_tensor.numpy().tobytes())
    return torch.load(buf, map_location="cpu", weights_only=False)


# A staging directory older than this was abandoned by a killed save.
STALE_STAGING_SECONDS = 3600.0


def cleanup_stale_staging(save_path: Path, max_age: float = STALE_STAGING_SECONDS):
    """Remove staging directories a crashed or killed save left behind."""
    now = time.time()
    for staging in save_path.parent.glob(".*.tmp-*"):
        try:
            if not staging.is_dir() or now - staging.stat().st_mtime < max_age:
                continue
            shutil.rmtree(staging, ignore_errors=True)
        except OSError:
            continue


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_file(path: Path):
    with path.open("rb") as file:
        os.fsync(file.fileno())


def _sync_directory(path: Path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _checkpoint_manifest(
    save_path: Path,
    state_dict: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    files = {}
    for path in sorted(save_path.iterdir()):
        if path.is_file() and path.name != _MANIFEST_FILE:
            files[path.name] = {"size": path.stat().st_size}
    return {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "created_at": meta["timestamp"],
        "optimizer_step": meta.get("optimizer_step"),
        "policy_version": meta.get("policy_version"),
        "tensors": sorted(state_dict),
        "files": files,
    }


def _validate_manifest(
    save_path: Path,
    verify_checksums: bool = False,
    broadcast: bool = False,
) -> Dict:
    def validate() -> Dict:
        manifest = load_json(save_path / _MANIFEST_FILE)
        version = manifest.get("format_version")
        if version not in _SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(f"Unsupported checkpoint format version: {version}")

        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("Checkpoint manifest has no file table")
        for required in (_META_FILE, _CONFIG_FILE, _WEIGHTS_FILE):
            if required not in files:
                raise ValueError(f"Checkpoint manifest is missing {required}")

        for name, descriptor in files.items():
            if Path(name).name != name or not isinstance(descriptor, dict):
                raise ValueError(f"Invalid checkpoint manifest entry: {name!r}")
            path = save_path / name
            if not path.is_file():
                raise ValueError(f"Checkpoint file is missing: {name}")
            if path.stat().st_size != descriptor.get("size"):
                raise ValueError(f"Checkpoint file size mismatch: {name}")
            if (
                verify_checksums
                and "sha256" in descriptor
                and _sha256(path) != descriptor["sha256"]
            ):
                raise ValueError(f"Checkpoint file checksum mismatch: {name}")
        return manifest

    return _broadcast_load(validate, broadcast)


def save_model(config: Dict, state_dict: Dict, save_directory: str):
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    save_json(config, save_path / _CONFIG_FILE)
    save_model_weights(state_dict, save_path)


def load_model_config(save_directory: str) -> Dict:
    return load_json(Path(save_directory) / _CONFIG_FILE)


def save_model_weights(state_dict: Dict, save_directory: Union[str, Path]) -> None:
    """Write model.safetensors only; callers own configuration and metadata."""
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    save_safetensors(state_dict, save_path / _WEIGHTS_FILE)


def load_model_weights(
    save_directory: Union[str, Path], broadcast: bool = False
) -> Dict:
    """Read model.safetensors or its indexed shards without modifying a model."""
    save_path = Path(save_directory)
    weights_file = save_path / _WEIGHTS_FILE
    if weights_file.exists():
        return _load_weight_file(weights_file, broadcast=broadcast)

    index_path = save_path / "model.safetensors.index.json"
    if index_path.exists():
        index = load_json(index_path, broadcast=broadcast)
        weight_map = index.get("weight_map", {})
        state_dict = {}
        for shard in sorted(set(weight_map.values())):
            state_dict.update(_load_weight_file(save_path / shard, broadcast=broadcast))
        return state_dict

    raise FileNotFoundError(f"No model weights found in {save_directory}")


def _load_weight_file(path: Union[str, Path], broadcast: bool = False) -> Dict:
    path = Path(path)
    if not broadcast or not dist.is_initialized():
        return load_safetensors(path)

    rank = get_rank()
    if rank == 0:
        state_dict = load_safetensors(path)
        specs = [
            (k, list(state_dict[k].shape), str(state_dict[k].dtype).split(".")[-1])
            for k in sorted(state_dict)
        ]
    else:
        state_dict = {}
        specs = []

    specs_list = [specs]
    dist.broadcast_object_list(specs_list, src=0)
    specs = specs_list[0]

    for key, shape, dtype_name in specs:
        dtype = getattr(torch, dtype_name)
        if rank != 0:
            tensor = torch.empty(shape, dtype=dtype, device="cpu")
        else:
            tensor = state_dict[key].contiguous().cpu()
        dist.broadcast(tensor, src=0)
        if rank != 0:
            state_dict[key] = tensor
    return state_dict


@dataclass
class Checkpoint:
    state_dict: Dict[str, Any] = field(default_factory=dict)
    epoch: int = 0
    consumed_samples: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def save(
        self, save_dir: str, prepare_staging: Optional[Callable[[Path], None]] = None
    ) -> bool:
        """Publish this state; ``False`` when that step is already on disk."""
        save_path = Path(save_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists() and not save_path.is_dir():
            raise FileExistsError(
                f"Checkpoint path exists and is not a directory: {save_path}"
            )
        if (save_path / _MANIFEST_FILE).exists():
            # Publication is write-once per step: the interval save and the
            # shutdown save at the same step describe the same state, so the
            # second one is a no-op instead of a re-publish.
            logger.info("Checkpoint %s is already published; keeping it", save_path)
            return False
        cleanup_stale_staging(save_path)

        staging_path = Path(
            tempfile.mkdtemp(
                prefix=f".{save_path.name}.tmp-",
                dir=save_path.parent,
            )
        )

        meta = {
            "epoch": self.epoch,
            "consumed_samples": self.consumed_samples,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **self.meta,
        }
        try:
            save_json(meta, staging_path / _META_FILE)
            save_json(self.config, staging_path / _CONFIG_FILE)
            save_model_weights(self.state_dict, staging_path)
            for key, value in self.extra.items():
                save_torch(value, staging_path / f"{key}.pt")
            if prepare_staging is not None:
                prepare_staging(staging_path)

            manifest = _checkpoint_manifest(staging_path, self.state_dict, meta)
            save_json(manifest, staging_path / _MANIFEST_FILE)
            for path in staging_path.iterdir():
                if path.is_file():
                    _sync_file(path)
            _sync_directory(staging_path)

            if save_path.exists():
                # A directory without a manifest is not a checkpoint: a crashed
                # writer stages and never publishes, so this is manual junk.
                logger.warning("Replacing %s: it carries no manifest", save_path)
                shutil.rmtree(save_path)
            os.replace(staging_path, save_path)
            _sync_directory(save_path.parent)
        except BaseException:
            shutil.rmtree(staging_path, ignore_errors=True)
            raise
        return True

    @classmethod
    def verify(cls, save_dir: str, checksums: bool = True) -> Dict[str, Any]:
        """Validate a published checkpoint without loading its weights.

        Sizes are always checked. Per-file checksums are verified only when
        the manifest carries them (v1 directories still do); v2 publishes
        without hashing, because the atomic rename already means "visible
        means complete".
        """
        save_path = Path(save_dir)
        manifest = _validate_manifest(save_path, verify_checksums=checksums)
        files = manifest.get("files") or {}
        return {
            "path": str(save_path),
            "format_version": manifest.get("format_version"),
            "files": len(files),
            "tensors": len(manifest.get("tensors") or ()),
            "checksums": (
                "verified"
                if any("sha256" in descriptor for descriptor in files.values())
                else "absent"
            ),
        }

    @classmethod
    def load(
        cls,
        save_dir: str,
        broadcast: bool = False,
        verify_checksums: bool = False,
        extra_keys: Optional[List[str]] = None,
        load_weights: bool = True,
    ) -> "Checkpoint":
        """Load selected extras; None loads all and an empty list loads none."""
        save_path = Path(save_dir)

        if (save_path / _MANIFEST_FILE).exists():
            _validate_manifest(
                save_path,
                verify_checksums=verify_checksums,
                broadcast=broadcast,
            )

        meta = load_json(save_path / _META_FILE, broadcast)
        config = load_json(save_path / _CONFIG_FILE, broadcast)
        state_dict = (
            load_model_weights(save_path, broadcast=broadcast) if load_weights else {}
        )

        extra = {}
        for f in sorted(save_path.iterdir()):
            if f.suffix == ".pt" and (extra_keys is None or f.stem in extra_keys):
                extra[f.stem] = load_torch(f, broadcast=broadcast)

        return cls(
            state_dict=state_dict,
            epoch=meta.get("epoch", 0),
            consumed_samples=meta.get("consumed_samples", 0),
            extra=extra,
            meta=meta,
            config=config,
        )

    @classmethod
    def load_any(
        cls,
        save_dir: str,
        broadcast: bool = False,
        verify_checksums: bool = False,
        extra_keys: Optional[List[str]] = None,
        load_weights: bool = True,
    ) -> Optional["Checkpoint"]:
        save_path = Path(save_dir)
        meta_path = save_path / _META_FILE
        weights_path = save_path / _WEIGHTS_FILE

        if meta_path.exists():
            return cls.load(
                save_dir,
                broadcast=broadcast,
                verify_checksums=verify_checksums,
                extra_keys=extra_keys,
                load_weights=load_weights,
            )

        weights_path = save_path / _WEIGHTS_FILE
        index_path = save_path / "model.safetensors.index.json"
        if weights_path.exists() or index_path.exists():
            state_dict = (
                load_model_weights(save_dir, broadcast=broadcast)
                if load_weights
                else {}
            )
            config = {}
            config_path = save_path / _CONFIG_FILE
            if config_path.exists():
                config = load_json(config_path, broadcast)
            return cls(state_dict=state_dict, config=config)

        return None
