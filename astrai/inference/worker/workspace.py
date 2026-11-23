"""Pre-allocated buffers for the inference decode hot path.

Mirrors FlashInfer / SGLang's global workspace pattern: all decode buffers have stable addresses after the first bind. Each step
reads/writes fixed-address tensors with zero ``torch.empty`` calls during
the hot loop — a prerequisite for CUDA-graph capture.
"""

from itertools import accumulate
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from astrai.config.inference_config import InferenceConfig
from astrai.model.kv_cache import DecodeKVCache, KVCache, KVStorage, PrefillKVCache

_CONFIG = InferenceConfig()
MAX_SPLITS = _CONFIG.max_splits


class InferenceWorkspace:
    """Reusable fixed-shape per-step buffers for decode.

    Families of buffers, all sized to ``max_batch_size`` / ``max_seq_len``
    and sliced via views each step:

    - ``input_ids``: per-step token IDs filled from host — values are
      staged through a pinned buffer and bulk-copied into the stable
      device buffer (fixed address for CUDA-graph capture).
    - KV-cache bind metadata (request rows, page table, past lengths, query offsets), written by
      ``InferenceWorkspace.bind`` when the model runner passes this workspace.
    - ``decode_o_part`` / ``decode_ml_part``: split-KV partial result buffers
      (mirrors FlashInfer's workspace).  One global alloc, reused by every
      decode step across all layers.  Sliced views are passed to the CUDA
      attention kernel so its internal ``torch.empty`` hot-path alloc goes
      through a stable address (CUDA-graph capturable).

    No re-allocation while the server's bounds are respected.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        max_q_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_q_heads = max_q_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Invariant: all workspace buffers are plain (non-inference)
        # tensors.  The scheduler's loop thread mutates them in-place every
        # step without any ambient inference-mode context — the mode is
        # thread-local and the loop runs on its own thread.  Inference
        # identity is fixed at construction and cannot be revoked later,
        # so allocation must force the mode off: callers that build the
        # engine inside ``torch.inference_mode()`` (e.g.
        # ``scripts/tools/generate.py``) would otherwise create inference
        # tensors that reject off-thread in-place updates.
        with torch.inference_mode(False):
            # Per-step token IDs.  Values come from host Python lists every
            # step, so the device buffer is pre-allocated (stable address for
            # CUDA-graph capture) and filled via a host staging buffer.
            self.input_ids = torch.empty(
                (max_batch_size,), dtype=torch.long, device=device
            )
            self._pin = torch.empty(
                (max_batch_size,),
                dtype=torch.long,
                pin_memory=torch.cuda.is_available()
                and torch.device(device).type == "cuda",
            )

            # KV-cache bind metadata (fixed shape, written by
            # ``InferenceWorkspace.bind`` when the model runner passes this
            # workspace).  Stable addresses make the decode forward
            # CUDA-graph capturable.
            self.req_pool_indices = torch.empty(
                (max_batch_size,), dtype=torch.int32, device=device
            )
            self.cache_seqlens = torch.empty(
                max_batch_size, dtype=torch.int32, device=device
            )
            self.cu_seqlens_q = torch.empty(
                max_batch_size + 1, dtype=torch.int32, device=device
            )
            # The pool knows page size. Allocate this graph-stable table at first bind.
            self.page_table = None
            self._source_table = None
            self._page_version = -1
            self._rows = ()
            self._positions = ()
            self._decode = None

            # Split-KV partial-result buffers for decode (persistent, one
            # global alloc per process — mirrors FlashInfer's workspace
            # pattern).  Shape:
            # [max_batch_size, max_q_heads, MAX_SPLITS, head_dim] (o_part)
            # [max_batch_size, max_q_heads, MAX_SPLITS, 2]     (ml_part)
            self.decode_o_part = torch.empty(
                (max_batch_size, max_q_heads, MAX_SPLITS, head_dim),
                dtype=torch.float32,
                device=device,
            )
            self.decode_ml_part = torch.empty(
                (max_batch_size, max_q_heads, MAX_SPLITS, 2),
                dtype=torch.float32,
                device=device,
            )

            # Decode output buffer (graph-safe pre-alloc).  Shape matches the
            # decode kernel's output: [batch, q_head, head_dim].
            self.decode_out = torch.empty(
                (max_batch_size, max_q_heads, head_dim),
                dtype=dtype,
                device=device,
            )

    def bind(
        self,
        storage: KVStorage,
        table: Tensor,
        version: int,
        rows: Tuple[int, ...],
        positions: Optional[List[int]],
        start: Optional[int],
        ends: Optional[List[int]],
    ) -> KVCache:
        """Bind request pages and this step's window to fixed-address model inputs."""
        b = len(rows)
        changed = self._source_table is not table or self._rows != rows
        if self.page_table is None or self.page_table.size(1) != table.size(1):
            with torch.inference_mode(False):
                self.page_table = table.new_empty((self.max_batch_size, table.size(1)))
        if changed:
            self.req_pool_indices[:b].copy_(
                torch.as_tensor(rows, dtype=torch.int32, device=self.device)
            )
        if changed or version != self._page_version:
            torch.index_select(
                table, 0, self.req_pool_indices[:b], out=self.page_table[:b]
            )
        self._source_table, self._rows, self._page_version = table, rows, version
        if positions is not None:
            reuse = not changed and self._decode is not None
            if not reuse:
                self._decode = DecodeKVCache(
                    storage.k_buffer,
                    storage.v_buffer,
                    self.page_table[:b],
                    self.cache_seqlens[:b],
                    self.decode_o_part,
                    self.decode_ml_part,
                    self.decode_out,
                    layer_views=storage.layer_views,
                )
            past = self._decode.cache_seqlens
            if reuse and all(
                pos == prev + 1 for pos, prev in zip(positions, self._positions)
            ):
                past.add_(1)
            else:
                past.copy_(
                    torch.as_tensor(positions, dtype=torch.int32, device=self.device)
                )
            self._positions = tuple(positions)
            return self._decode
        pages, past = self.page_table[:b], self.cache_seqlens[:b]
        self._decode = None
        past.fill_(start)
        offsets = self.cu_seqlens_q[: b + 1]
        offsets.copy_(
            torch.tensor(
                [0, *accumulate(end - start for end in ends)],
                device=self.device,
                dtype=torch.int32,
            )
        )
        return PrefillKVCache(
            storage.k_buffer,
            storage.v_buffer,
            pages,
            past,
            offsets,
            layer_views=storage.layer_views,
        )

    def fill_input_ids(self, ids: "list[int]") -> Tensor:
        """Write ``ids`` into the device buffer and return ``[B]``.

        Host values are staged through a pinned buffer and copied synchronously
        into the stable device buffer.
        """
        b = len(ids)
        for i, v in enumerate(ids):
            self._pin[i] = v
        self.input_ids[:b].copy_(self._pin[:b])
        return self.input_ids[:b]

    def fill_input_ids_from_device(self, tokens: Tensor) -> Tensor:
        """Copy device-resident ``[B]`` token ids into the device buffer.

        Steady-state decode fast path: when the executor's cached task
        signature still matches, the previous step's sampled tokens map
        1:1 onto the current slots, so the ids transfer device-to-device
        instead of round-tripping through the host staging buffers.
        """
        b = tokens.size(0)
        self.input_ids[:b].copy_(tokens)
        return self.input_ids[:b]
