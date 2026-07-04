"""Storage writer strategies for pipeline output.

The :class:`StoreWriter` abstraction decouples the pipeline from the
concrete storage format (bin / h5).  The pipeline builds a ``{key:
List[Tensor]}`` dict and delegates the write to the writer selected
by ``output.storage_format``.
"""

import logging
import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from astrai.factory import BaseFactory
from astrai.serialization import save_bin, save_h5

logger = logging.getLogger(__name__)


class StoreWriter(ABC):
    """Write pre-tokenized tensors to disk in a format-specific way."""

    @abstractmethod
    def save(
        self,
        output_dir: str,
        domain: str,
        shard_idx: int,
        tensors: Dict[str, List[torch.Tensor]],
    ) -> None: ...


class StoreWriterFactory(BaseFactory["StoreWriter"]):
    pass


@StoreWriterFactory.register("bin")
class BinWriter(StoreWriter):
    def save(self, output_dir, domain, shard_idx, tensors):
        shard_path = os.path.join(output_dir, domain, f"shard_{shard_idx:04d}")
        try:
            save_bin(shard_path, tensors)
        except Exception:
            if os.path.exists(shard_path):
                shutil.rmtree(shard_path, ignore_errors=True)
            logger.error(
                "Failed to write shard %s/%s_%04d, cleaned up partial output",
                domain,
                "shard",
                shard_idx,
                exc_info=True,
            )
            raise


@StoreWriterFactory.register("h5")
class H5Writer(StoreWriter):
    def save(self, output_dir, domain, shard_idx, tensors):
        chunk_dir = os.path.join(output_dir, domain)
        file_path = os.path.join(chunk_dir, f"data_{shard_idx:04d}.h5")
        try:
            save_h5(chunk_dir, f"data_{shard_idx:04d}", tensors)
        except Exception:
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(
                "Failed to write shard %s/data_%04d.h5, cleaned up partial output",
                domain,
                shard_idx,
                exc_info=True,
            )
            raise
