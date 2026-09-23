"""CUDA kernel wrappers, operator dispatch, and backend selection.

Public API:
    - ``attention``, ``apply_rotary_emb`` — op families with safe torch
      fallbacks (see ``astrai.extension.backend``)
    - ``attn_decode`` / ``attn_prefill`` / ``attn_paged_decode`` /
      ``attn_paged_prefill`` — direct attention kernel wrappers
    - ``AttentionBackend`` / ``TorchNativeBackend`` / ``CudaBackend`` /
      ``FlashAttnBackend`` — attention backend strategies
    - ``resolve`` / ``explain`` / ``op_backend`` / ``set_op`` — the shared
      operator dispatcher (see ``astrai.extension.dispatch``)
    - ``plan`` — the runtime GEMM plan (`plan.config` / `plan.configure` /
      ``plan.override`` / ``plan.probe`` / ``plan.facts`` / ``plan.tiles``);
      the flat ``set_table`` / ``set_planner`` / ``set_log`` / ``set_staging``
      / ``state`` / ``probe`` / ``facts`` / ``tile_vocabulary`` names are the
      same bindings in their raw dict/list shapes (see
      ``astrai.extension.ops.gemm``); the deprecated ``ASTR_*`` variables are
      one-time startup seeds

Layout convention: all q/k/v are ``[batch, seq_len, n_heads, head_dim]``
(blhd). Scale is always ``1/sqrt(head_dim)``. Wrapper functions call their
compiled CUDA kernels directly; fallback is the backend's responsibility.
Linear projections and dense-MLP SwiGLU run plain torch (``F.linear`` /
``Linear`` / ``MLP``); the former bf16_gemm / bf16_swiglu kernels and their
backends were removed.
"""

from astrai.extension.backend import (
    ATTN_BACKEND,
    AttentionBackend,
    AttentionBackendFactory,
    CudaBackend,
    FlashAttnBackend,
    TorchNativeBackend,
    apply_rotary_emb,
    attention,
    attn_backend,
    get_backend,
)
from astrai.extension.dispatch import (
    Axes,
    ExplicitSelectionError,
    ImplRecord,
    Resolution,
    Spec,
    axis,
    explain,
    explain_plan,
    op_backend,
    register_env_alias,
    register_family,
    resolve,
    resolve_plan,
    set_op,
    tensor_axes,
)
from astrai.extension.loader import KERNEL_NAMES, is_available
from astrai.extension.ops import (
    TensorLayout,
    attn_decode,
    attn_paged_decode,
    attn_paged_prefill,
    attn_prefill,
)
from astrai.extension.ops.gemm import (
    facts,
    probe,
    set_log,
    set_planner,
    set_staging,
    set_table,
    state,
    tile_vocabulary,
)
from astrai.extension.plan import PLANNER_MODES

__all__ = [
    "ATTN_BACKEND",
    "AttentionBackend",
    "AttentionBackendFactory",
    "CudaBackend",
    "TorchNativeBackend",
    "FlashAttnBackend",
    "TensorLayout",
    "attention",
    "attn_backend",
    "get_backend",
    "attn_decode",
    "attn_paged_decode",
    "attn_prefill",
    "attn_paged_prefill",
    "is_available",
    "KERNEL_NAMES",
    "apply_rotary_emb",
    "Axes",
    "ExplicitSelectionError",
    "ImplRecord",
    "Resolution",
    "Spec",
    "axis",
    "set_op",
    "explain",
    "explain_plan",
    "op_backend",
    "register_env_alias",
    "register_family",
    "resolve",
    "resolve_plan",
    "tensor_axes",
    "PLANNER_MODES",
    "plan",
    "facts",
    "probe",
    "set_log",
    "set_planner",
    "set_staging",
    "set_table",
    "state",
    "tile_vocabulary",
]
