"""Per-operation selection precedence and capability tests."""

import importlib

import pytest
import torch

from astrai.extension import (
    op_backend,
    set_op,
    snapshot_op_selection,
    use_op_selection,
)
from astrai.extension.backend import apply_rotary_emb
from astrai.extension.backend.rotary import RotaryDispatcher, _torch_apply
from astrai.extension.runtime import context
from astrai.extension.runtime.loader import is_available

attention = importlib.import_module("astrai.extension.backend.attention")


def test_nested_choices_restore_and_operators_are_independent():
    x = torch.randn(1, 2, 2, 8)
    freqs = torch.randn(1, 2, 4, 2)
    with op_backend(rotary="torch", attention="torch_native"):
        assert RotaryDispatcher().dispatch(x, freqs) is _torch_apply
        with op_backend(attention="cuda"):
            assert RotaryDispatcher().dispatch(x, freqs) is _torch_apply
        assert (
            attention.AttentionDispatcher().dispatch(
                x, x, x, None, 0, None, False, None
            )
            is attention.torch.forward
        )


def test_snapshot_freezes_process_choice(monkeypatch):
    monkeypatch.setattr(context, "_process_choices", None)
    set_op("attention", "torch_native")
    snapshot = snapshot_op_selection()
    set_op("attention", "cuda")
    with use_op_selection(snapshot):
        assert context.op_context.snapshot().process()["attention"] == "torch_native"


def test_env_seed_accepts_only_known_operators(monkeypatch):
    monkeypatch.setenv("ASTR_OPS", "attention=torch,rotary=torch,profile=reference")
    monkeypatch.setattr(context, "_process_choices", None)
    assert context.op_context.snapshot().process() == {
        "attention": "torch",
        "rotary": "torch",
    }
    with pytest.raises(ValueError, match="unknown operator"):
        set_op("profile", "reference")


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="rotary CUDA path needs a GPU"
)
class TestRotaryDispatch:
    def _input(self):
        torch.manual_seed(0)
        x = torch.randn(1, 5, 3, 16, device="cuda", dtype=torch.bfloat16)
        freqs = torch.randn(1, 5, 8, 2, device="cuda", dtype=torch.float32)
        return x, freqs

    def test_frequency_grad_uses_torch(self):
        x, freqs = self._input()
        freqs.requires_grad_()
        assert RotaryDispatcher().dispatch(x, freqs) is _torch_apply

    def test_cuda_matches_torch_numerics(self):
        if not is_available("rotary_emb"):
            pytest.skip("rotary kernel not built")
        x, freqs = self._input()
        with torch.inference_mode():
            fast = apply_rotary_emb(x, freqs)
            ref = _torch_apply(x, freqs)
        torch.testing.assert_close(fast.float(), ref.float(), atol=2e-2, rtol=1e-2)

    @pytest.mark.parametrize("sliced", [False, True])
    def test_cuda_backward_matches_torch(self, sliced):
        if not is_available("rotary_emb"):
            pytest.skip("rotary kernel not built")
        x, freqs = self._input()
        if sliced:
            x = torch.cat((x, x), dim=-1)[..., :16]
        x = x.detach().requires_grad_()
        grad = torch.randn_like(x)
        with op_backend(rotary="cuda"):
            output = apply_rotary_emb(x, freqs)
        grad_x = torch.autograd.grad(output, x, grad)[0]
        ref_x = x.detach().float().requires_grad_()
        ref = _torch_apply(ref_x, freqs)
        ref_grad = torch.autograd.grad(ref, ref_x, grad.float())[0]
        torch.testing.assert_close(output.float(), ref.float(), atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(grad_x.float(), ref_grad, atol=2e-2, rtol=1e-2)
