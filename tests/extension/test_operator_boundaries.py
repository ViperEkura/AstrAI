"""Reject malformed CUDA operator inputs before allocation or launch."""

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from astrai.extension.runtime.loader import get_module, is_available

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _attention_inputs():
    q = torch.zeros((1, 1, 1, 32), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    return q, k


@pytest.mark.cuda
@pytest.mark.skipif(not is_available("attention"), reason="attention kernel not built")
def test_attention_rejects_int32_stride_overflow():
    q, k = _attention_inputs()
    q = q.as_strided(q.shape, (2**31, 32, 32, 1))
    with pytest.raises(RuntimeError, match=r"q stride\(0\).*int32"):
        get_module("attention").attn_prefill(q, k, k)


@pytest.mark.cuda
@pytest.mark.skipif(not is_available("attention"), reason="attention kernel not built")
def test_attention_rejects_empty_batch_before_launch():
    q, k = _attention_inputs()
    with pytest.raises(RuntimeError, match="batch, heads, and query length"):
        get_module("attention").attn_prefill(q[:0], k[:0], k[:0])


@pytest.mark.cuda
@pytest.mark.skipif(not is_available("rotary_emb"), reason="RoPE kernel not built")
def test_rotary_empty_input_returns_without_launch():
    x = torch.empty((0, 2, 16), device="cuda", dtype=torch.bfloat16)
    freqs = torch.empty((0, 8, 2), device="cuda", dtype=torch.float32)
    out = get_module("rotary_emb").rotary_emb(x, freqs)
    assert out.shape == x.shape and out.numel() == 0


@pytest.mark.cuda
@pytest.mark.skipif(not is_available("rotary_emb"), reason="RoPE kernel not built")
def test_rotary_rejects_mismatched_token_shape():
    x = torch.empty((2, 3, 2, 16), device="cuda", dtype=torch.bfloat16)
    freqs = torch.empty((3, 2, 8, 2), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="token dimensions"):
        get_module("rotary_emb").rotary_emb(x, freqs)


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_rotary_rejects_cross_device_input():
    x = torch.empty((1, 2, 16), device="cuda:0", dtype=torch.bfloat16)
    freqs = torch.empty((1, 8, 2), device="cuda:1", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="same CUDA device"):
        get_module("rotary_emb").rotary_emb(x, freqs)


@pytest.mark.cuda
@pytest.mark.skipif(not is_available("gated_deltanet"), reason="GDN kernel not built")
def test_gdn_forward_rejects_mismatched_beta_shape():
    q = torch.empty((1, 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    g = torch.empty((1, 64, 1), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="g/beta must be"):
        get_module("gated_deltanet").gated_deltanet_fwd(q, q, q, g, g[:, :63], 1e-6, 64)


@pytest.mark.cuda
@pytest.mark.skipif(not is_available("gated_deltanet"), reason="GDN kernel not built")
def test_gdn_backward_rejects_mismatched_state_shape():
    q = torch.empty((1, 1, 64, 128), device="cuda", dtype=torch.bfloat16)
    h = torch.empty((1, 1, 2, 128, 128), device="cuda", dtype=torch.bfloat16)
    g = torch.empty((1, 1, 64), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="h must be"):
        get_module("gated_deltanet").gated_deltanet_bwd(q, q, q, h, g, q, 0.1)


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_gdn_forward_rejects_cross_device_gate():
    q = torch.empty((1, 64, 1, 128), device="cuda:0", dtype=torch.bfloat16)
    g = torch.empty((1, 64, 1), device="cuda:1", dtype=torch.float32)
    beta = torch.empty((1, 64, 1), device="cuda:0", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="same CUDA device"):
        get_module("gated_deltanet").gated_deltanet_fwd(q, q, q, g, beta, 1e-6, 64)


@pytest.mark.cuda
def test_native_launch_check_raises_instead_of_exiting(tmp_path):
    compiler = shutil.which("nvcc")
    if compiler is None:
        pytest.skip("nvcc not installed")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "launch_error.cu"
    program = tmp_path / "launch_error"
    source.write_text(
        "#include <utils/launch.cuh>\n"
        "#include <string>\n"
        "int main() {\n"
        "  try { ASTRAI_CUDA_CHECK(cudaErrorInvalidValue); }\n"
        "  catch (const astrai::cuda::Error& e) {\n"
        '    return std::string(e.what()).find("cudaErrorInvalidValue") != '
        "std::string::npos ? 0 : 2;\n"
        "  }\n"
        "  return 1;\n"
        "}\n"
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(root / "csrc/include"),
            str(source),
            "-o",
            str(program),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(program)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
