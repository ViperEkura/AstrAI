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
    - ``fp8_autocast`` / ``FP8Recipe`` / ``fp8_linear_enable`` /
      ``fp8_state_dict`` — the fp8 autocast region and its checkpoint
      bridge (see ``astrai.extension.autocast``); ``quantize_weight_int8`` /
      ``quantize_act_int8`` below are the stateless int8 inference
      strategies
    - ``plan`` — the runtime GEMM plan (`plan.config` / `plan.configure` /
      ``plan.override`` / ``plan.probe`` / ``plan.facts`` / ``plan.tiles``);
      the flat ``set_table`` / ``set_planner`` / ``set_log`` / ``set_staging``
      / ``state`` / ``probe`` / ``facts`` / ``tile_vocabulary`` names are the
      same bindings in their raw dict/list shapes (see
      ``astrai.extension.kernel.gemm``); the deprecated ``ASTR_*`` variables are
      one-time startup seeds

Layout convention: all q/k/v are ``[batch, seq_len, n_heads, head_dim]``
(blhd). Scale is always ``1/sqrt(head_dim)``. Wrapper functions call their
compiled CUDA kernels directly; fallback is the backend's responsibility.
Linear projections and dense-MLP SwiGLU run plain torch (``F.linear`` /
``Linear`` / ``MLP``); the former bf16_gemm / bf16_swiglu kernels and their
backends were removed.
"""

import torch

from astrai.extension.autocast import (
    FP8Recipe,
    fp8_autocast,
    fp8_format_pair,
    fp8_linear_enable,
    fp8_linear_enabled,
    fp8_load_state_dict,
    fp8_reset,
    fp8_state_dict,
)
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
from astrai.extension.kernel import (
    TensorLayout,
    attn_decode,
    attn_paged_decode,
    attn_paged_prefill,
    attn_prefill,
    muon_ns,
)
from astrai.extension.kernel.gemm import (
    facts,
    probe,
    set_log,
    set_planner,
    set_staging,
    set_table,
    state,
    tile_vocabulary,
)
from astrai.extension.loader import KERNEL_NAMES, is_available
from astrai.extension.plan import PLANNER_MODES

# ---------------------------------------------------------------------------
# INT8: stateless symmetric inference strategies
# ---------------------------------------------------------------------------


def quantize_weight_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel int8 quantization of a linear weight.

    ``w`` is ``[N, K]`` (the nn.Linear convention); returns
    ``(w8 int8 [N, K] contiguous, scale f32 [N])`` with
    ``w ≈ w8.float() * scale[:, None]`` and ``scale = amax(N) / 127``.
    Quantization runs in float32 regardless of the source dtype.
    """
    wf = w.detach().to(torch.float32)
    scale = wf.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
    q = torch.round(wf / scale.unsqueeze(-1)).clamp_(-127, 127)
    return q.to(torch.int8).contiguous(), scale.contiguous()


def quantize_act_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row dynamic int8 quantization of activations.

    ``x`` is ``[..., K]``; returns ``(q int8 with x's shape, scale f32
    [prod(leading dims)])`` — upstream of ``quant_gemm``'s per-row a_scale
    (the scale contract: ``docs/developer/kernels/gemm.md``, "Scales").
    """
    xf = x.detach().to(torch.float32)
    x2 = xf.reshape(-1, xf.shape[-1])
    scale = x2.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
    q = torch.round(x2 / scale.unsqueeze(-1)).clamp_(-127, 127)
    return q.to(torch.int8).reshape(x.shape), scale


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
    "muon_ns",
    "is_available",
    "KERNEL_NAMES",
    "apply_rotary_emb",
    "FP8Recipe",
    "fp8_autocast",
    "fp8_format_pair",
    "fp8_linear_enable",
    "fp8_linear_enabled",
    "fp8_load_state_dict",
    "fp8_reset",
    "fp8_state_dict",
    "quantize_act_int8",
    "quantize_weight_int8",
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
