"""Pre-allocated buffers for the inference decode hot path.

Mirrors FlashInfer / SGLang's global workspace pattern: all per-step tensors
are allocated eagerly at init (nothing is lazy), so the decode step
reads/writes fixed-address tensors with zero ``torch.empty`` calls during
the hot loop — a prerequisite for CUDA-graph capture.
"""

import torch
from torch import Tensor

from astrai.config.inference_config import InferenceConfig

_CONFIG = InferenceConfig()
MAX_SPLITS = _CONFIG.max_splits
Q_TILE_ROWS = _CONFIG.q_tile_rows


class InferenceWorkspace:
    """Reusable fixed-shape per-step buffers for decode.

    Families of buffers, all sized to ``max_batch_size`` / ``max_seq_len``
    and sliced via views each step:

    - ``decode_mask``: a ``[B, 1, total_len]`` validity mask, the RHS
      ``arange`` pre-computed so only a single ``torch.ge(out=)`` runs per
      step.
    - ``input_ids``: per-step token IDs filled from host — values are
      staged through a pinned buffer and bulk-copied into the stable
      device buffer (fixed address for CUDA-graph capture).
    - KV-cache bind metadata (``req_pool_indices``, ``seq_lens``,
      ``kv_indptr``, ``inc``, ``out_cache_loc``), written by
      ``PagePool.bind_tasks`` when the Executor passes this workspace.
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
            # ``position_ids[:, None, None] >= arange`` RHS, reused every step.
            self.arange = torch.arange(max_seq_len, device=device)
            # Decode validity mask: [max_batch, 1, max_seq_len] bool.
            self.input_mask = torch.empty(
                (max_batch_size, 1, max_seq_len), dtype=torch.bool, device=device
            )

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
            # ``PagePool.bind_tasks`` when the Executor passes this
            # workspace).  Stable addresses make the decode forward
            # CUDA-graph capturable.
            self.req_pool_indices = torch.empty(
                (max_batch_size,), dtype=torch.int32, device=device
            )
            self.seq_lens = torch.empty(
                (max_batch_size,), dtype=torch.long, device=device
            )
            self.kv_indptr = torch.empty(
                (max_batch_size + 1,), dtype=torch.int32, device=device
            )
            self.qo_indptr = torch.empty(
                (max_batch_size + 1,), dtype=torch.int32, device=device
            )
            max_q_tiles = max_batch_size * (
                (max_seq_len + Q_TILE_ROWS - 1) // Q_TILE_ROWS
            )
            self.q_tile_to_batch = torch.empty(
                (max_q_tiles,), dtype=torch.int32, device=device
            )
            self.q_tile_to_index = torch.empty(
                (max_q_tiles,), dtype=torch.int32, device=device
            )
            self.inc = torch.arange(
                max_batch_size + 1, dtype=torch.int32, device=device
            )
            self.out_cache_loc = torch.empty(
                (max_batch_size, 1), dtype=torch.int32, device=device
            )

            # Per-step position IDs (must be at a fixed address for
            # CUDA-graph capture).
            self.position_ids = torch.empty(
                (max_batch_size,), dtype=torch.long, device=device
            )

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

    def decode_mask(self, position_ids: Tensor, total_len: int) -> Tensor:
        """Return the ``[B, 1, total_len]`` validity mask for this step.

        Written into the pre-allocated buffer via ``torch.ge(out=)`` — no
        new tensor is allocated.  ``position_ids`` is the current step's
        ``[B]`` positions; ``total_len`` must not exceed ``max_seq_len``.
        """
        b = position_ids.size(0)
        out = self.input_mask[:b, :, :total_len]
        torch.ge(position_ids[:, None, None], self.arange[:total_len], out=out)
        return out
