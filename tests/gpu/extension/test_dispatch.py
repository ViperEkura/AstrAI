"""Tests for the generic operator dispatcher (Spec / decision tables)."""

import importlib

import pytest
import torch

import astrai.extension.runtime.dispatch as dispatch
from astrai.extension import (
    ImplRecord,
    Spec,
    axis,
    op_backend,
    resolve,
)
from astrai.extension.backend import apply_rotary_emb
from astrai.extension.runtime.loader import is_available

attn_mod = importlib.import_module("astrai.extension.backend.attention")
rotary_mod = importlib.import_module("astrai.extension.backend.rotary")


@pytest.fixture
def toy_family():
    """A toy family: alpha (restricted), beta, and an unfaithful fast row."""
    calls = []

    def records():
        return [
            ImplRecord(
                "toy",
                "alpha",
                "alpha-obj",
                axis("dtype").in_(torch.bfloat16),
                priority=0,
            ),
            ImplRecord(
                "toy",
                "beta",
                "beta-obj",
                Spec.always(),
                priority=10,
            ),
            ImplRecord(
                "toy",
                "fp8",
                "fp8-obj",
                Spec.always(),
                priority=1,
                faithful=False,
            ),
        ]

    dispatch.register_family(
        "toy",
        lambda **kw: kw,
        records,
        lambda: ImplRecord("toy", "beta", "beta-obj", Spec.always()),
    )
    yield calls
    dispatch._FAMILIES.pop("toy", None)


class _Probe:
    def __init__(self, capable):
        self.capable = capable
        self.probed = 0

    def supports_call(self, *args, **kwargs):
        self.probed += 1
        return self.capable


_DUMMY_CACHE = object()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="rotary CUDA path needs a GPU"
)
class TestRotaryDispatch:
    def _input(self):
        torch.manual_seed(0)
        x = torch.randn(1, 5, 3, 16, device="cuda", dtype=torch.bfloat16)
        freqs = torch.randn(1, 5, 8, 2, device="cuda", dtype=torch.float32)
        return x, freqs

    def test_cuda_row_selected_under_inference_mode(self):
        x, freqs = self._input()
        with torch.inference_mode():
            resolution = resolve("rotary", x, freqs)
        expected = "cuda" if is_available("rotary_emb") else "torch"
        assert resolution.record.name == expected

    def test_grad_falls_back_to_torch(self):
        x, freqs = self._input()
        assert resolve("rotary", x, freqs).record.name == "torch"

    def test_context_switch_to_torch(self):
        x, freqs = self._input()
        with torch.inference_mode():
            with op_backend(rotary="torch"):
                out = apply_rotary_emb(x, freqs)
        assert out.shape == x.shape and out.dtype == torch.bfloat16

    def test_cuda_matches_torch_numerics(self):
        if not is_available("rotary_emb"):
            pytest.skip("rotary kernel not built")
        x, freqs = self._input()
        with torch.inference_mode():
            fast = apply_rotary_emb(x, freqs)
        slow = rotary_mod._torch_apply
        ref = slow(x, freqs)
        assert torch.allclose(fast.float(), ref.float(), atol=2e-2, rtol=1e-2)
