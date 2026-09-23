"""Tests for the generic operator dispatcher (Spec / decision tables)."""

import importlib
from types import SimpleNamespace

import pytest
import torch

import astrai.extension.dispatch as dispatch
from astrai.extension import (
    ATTN_BACKEND,
    ExplicitSelectionError,
    ImplRecord,
    Spec,
    axis,
    explain,
    op_backend,
    resolve,
    resolve_plan,
)
from astrai.extension.backend import apply_rotary_emb

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


def test_spec_composition_and_description():
    spec = axis("dtype").in_(torch.bfloat16) & axis("grad_enabled").eq(False)
    assert spec.matches({"dtype": torch.bfloat16, "grad_enabled": False})
    assert not spec.matches({"dtype": torch.bfloat16, "grad_enabled": True})
    assert "dtype" in spec.description and "grad_enabled" in spec.description

    either = axis("fwd").none() | axis("has_cache").truthy()
    assert either.matches({"fwd": None})
    assert either.matches({"fwd": "decode", "has_cache": True})
    assert not either.matches({"fwd": "decode"})

    assert (~axis("fwd").none()).matches({"fwd": "decode"})


def test_chain_returns_first_capable(toy_family):
    assert resolve("toy", dtype=torch.bfloat16).record.obj == "alpha-obj"
    assert resolve("toy", dtype=torch.float32).record.obj == "beta-obj"


def test_unfaithful_rows_are_chain_invisible(toy_family):
    resolution = resolve("toy", dtype=torch.float32)
    assert resolution.record.obj == "beta-obj"
    with op_backend(toy="fp8"):
        assert resolve("toy", dtype=torch.float32).record.obj == "fp8-obj"


def test_explicit_selection_is_strict(toy_family):
    with pytest.raises(ExplicitSelectionError):
        resolve("toy", dtype=torch.float32, explicit="alpha")
    assert resolve("toy", dtype=torch.bfloat16, explicit="alpha").origin == "explicit"


def test_context_selection_is_strict(toy_family):
    with op_backend(toy="alpha"):
        with pytest.raises(ExplicitSelectionError):
            resolve("toy", dtype=torch.float32)
        assert resolve("toy", dtype=torch.bfloat16).origin == "context"


def test_unknown_explicit_name_raises(toy_family):
    with pytest.raises(ValueError, match="Unknown toy implementation"):
        resolve("toy", explicit="nope")
    with pytest.raises(ValueError, match="Unknown toy implementation"):
        with op_backend(toy="nope"):
            pass


class _Probe:
    def __init__(self, capable):
        self.capable = capable
        self.probed = 0

    def supports_call(self, *args, **kwargs):
        self.probed += 1
        return self.capable


def test_adhoc_instance_probed_via_supports_call(toy_family):
    probe = _Probe(capable=True)
    resolution = resolve("toy", dtype=torch.float32, explicit=probe)
    assert resolution.record.obj is probe and probe.probed == 1

    incapable = _Probe(capable=False)
    with pytest.raises(ExplicitSelectionError):
        resolve("toy", dtype=torch.float32, explicit=incapable)


def test_nested_op_backend_scopes(toy_family):
    with op_backend(toy="alpha"):
        with op_backend(toy="beta"):
            assert resolve("toy", dtype=torch.float32).origin == "context"
        assert resolve("toy", dtype=torch.bfloat16).origin == "context"


def test_set_op_entry_is_soft(toy_family):
    dispatch.set_op("toy", "alpha")
    resolution = resolve("toy", dtype=torch.float32)
    assert resolution.record.obj == "beta-obj" and resolution.origin == "chain"
    assert resolve("toy", dtype=torch.bfloat16).origin == "env"


def test_set_op_unknown_impl_ignored(toy_family):
    dispatch.set_op("toy", "missing")
    assert resolve("toy", dtype=torch.float32).record.obj == "beta-obj"


def test_set_op_profile_reference(toy_family):
    dispatch.set_op("profile", "reference")
    resolution = resolve("toy", dtype=torch.bfloat16)
    assert resolution.origin == "profile"
    dispatch.set_op("toy", "alpha")
    assert resolve("toy", dtype=torch.bfloat16).origin == "env"


def test_env_seed_and_alias(monkeypatch):
    # The one-time seed: ASTR_OPS entries, with the legacy ASTR_BACKEND
    # alias filling a family ASTR_OPS left unset.
    monkeypatch.setenv("ASTR_BACKEND", "torch_native")
    dispatch._selection = None  # re-seed on the next lookup
    assert dispatch.env_selection("attention") == "torch_native"
    monkeypatch.setenv("ASTR_OPS", "attention=cuda")
    dispatch._selection = None
    assert dispatch.env_selection("attention") == "cuda"
    assert dispatch.parse_selections("toy=alpha,profile=reference") == {
        "toy": "alpha",
        "profile": "reference",
    }


def test_context_beats_set_op(toy_family):
    dispatch.set_op("toy", "alpha")
    with op_backend(toy="beta"):
        assert resolve("toy", dtype=torch.float32).origin == "context"


def test_resolve_plan_snapshots_families(toy_family):
    plan = resolve_plan(
        {
            "toy": ((), {"dtype": torch.bfloat16}),
            "rotary": (
                (
                    SimpleNamespace(dtype=torch.bfloat16, is_cuda=True),
                    SimpleNamespace(),
                ),
                {},
            ),
        }
    )
    assert plan["toy"].record.obj == "alpha-obj"
    assert plan["rotary"].record.name in ("cuda", "torch")


def test_explain_shows_rejection_reasons(toy_family):
    text = explain("toy", dtype=torch.float32)
    assert "alpha: reject" in text and "beta: MATCH" in text
    assert "=> beta" in text


def test_explain_reports_strict_error(toy_family):
    text = explain("toy", dtype=torch.float32, explicit="alpha")
    assert "ERROR" in text


_DUMMY_CACHE = object()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_dim", [64, 96])
@pytest.mark.parametrize(
    "fwd,has_cache,ndim,has_mask",
    [
        ("decode", True, 3, False),
        ("prefill", True, 3, False),
        (None, False, 4, False),
        (None, False, 4, True),
    ],
)
def test_attention_specs_mirror_supports_call(
    dtype, head_dim, fwd, has_cache, ndim, has_mask
):
    shape = {3: (1, 2, head_dim), 4: (1, 2, 4, head_dim)}[ndim]
    q = torch.zeros(shape, dtype=dtype)
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool) if has_mask else None
    cache = _DUMMY_CACHE if has_cache else None
    ax = attn_mod._axes(q, cache, mask, False, fwd)

    cuda = attn_mod._instance(attn_mod.CudaBackend)
    assert attn_mod._SPEC_CUDA.matches(ax) == cuda.supports_call(
        q, cache, mask, False, fwd
    )

    flash = attn_mod._instance(attn_mod.FlashAttnBackend)
    assert attn_mod._SPEC_FLASH.matches(ax) == flash.supports_call(
        q, cache, mask, False, fwd
    )


def test_attention_resolution_matches_legacy_semantics():
    q = torch.zeros(1, 2, 4, 8, dtype=torch.float32)
    resolution = resolve("attention", q, None, None, True, "prefill")
    assert resolution.record.name == ATTN_BACKEND.TORCH_NATIVE.value


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
        from astrai.extension.loader import is_available

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
        from astrai.extension.loader import is_available

        if not is_available("rotary_emb"):
            pytest.skip("rotary kernel not built")
        x, freqs = self._input()
        with torch.inference_mode():
            fast = apply_rotary_emb(x, freqs)
        slow = rotary_mod._torch_apply
        ref = slow(x, freqs)
        assert torch.allclose(fast.float(), ref.float(), atol=2e-2, rtol=1e-2)
