"""Batched NS preserves independent matrix and optimizer state semantics."""

import copy

import pytest
import torch
from torch import nn

from astrai.extension.backend.newton_schulz import newton_schulz
from astrai.extension.kernel.newton_schulz import is_available
from astrai.extension.policy import newton_schulz as plan
from astrai.optim.muon_adamw import MuonAdamW

CUDA_AVAILABLE = torch.cuda.is_available() and is_available()
pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="Muon NS CUDA kernels are unavailable"
)
COEFFICIENTS = (3.4445, -4.775, 2.0315)


def _small_plans():
    major, minor = torch.cuda.get_device_capability()
    rows = []
    for m, n in [(64, 128), (64, 64)]:
        for input_layout in ["row", "column"]:
            rows.append(
                dict(
                    operation="syrk",
                    cc=major * 10 + minor,
                    rows=m,
                    cols=n,
                    input_layout=input_layout,
                    backend="cuda",
                    tile="64x64x64_W16x32_S2",
                )
            )
            for output_layout in ["row", "column"]:
                rows.append(
                    dict(
                        operation="symm",
                        cc=major * 10 + minor,
                        rows=m,
                        cols=n,
                        input_layout=input_layout,
                        output_layout=output_layout,
                        addend=True,
                        backend="cuda",
                        tile="64x64x64_W16x32_S2",
                    )
                )
    rows.append(
        dict(
            operation="syrk",
            cc=major * 10 + minor,
            rows=64,
            cols=64,
            addend=True,
            backend="cuda",
            tile="64x64x64_W16x32_S2",
        )
    )
    return [
        dict(row, batch_size=batch_size) for batch_size in [1, 2, 3, 4] for row in rows
    ]


class _Matrices(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        shapes = [(64, 128)] * 5 + [(128, 64)] * 2 + [(64, 64)]
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(shape, device="cuda", dtype=dtype))
                for shape in shapes
            ]
        )
        self.inactive = nn.Parameter(torch.randn(64, 128, device="cuda", dtype=dtype))
        self.bias = nn.Parameter(torch.randn(7, device="cuda", dtype=dtype))


def _assign_gradients(models, seed):
    torch.manual_seed(seed)
    for index in range(len(models[0].weights)):
        gradient = torch.randn_like(models[0].weights[index])
        for model in models:
            model.weights[index].grad = gradient.clone()


def _assert_momentum_equal(reference, candidate, reference_model, candidate_model):
    for reference_param, candidate_param in zip(
        reference_model.weights, candidate_model.weights
    ):
        assert torch.equal(
            reference.muon.state[reference_param]["momentum_buffer"],
            candidate.muon.state[candidate_param]["momentum_buffer"],
        )
    assert reference_model.inactive not in reference.muon.state
    assert candidate_model.inactive not in candidate.muon.state


@pytest.mark.parametrize("shape", [(64, 128), (128, 64)])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ns_batch_matches_independent_matrices_and_input_mutation(shape, steps, dtype):
    torch.manual_seed(83)
    matrix = torch.randn(3, *shape, device="cuda", dtype=dtype)
    before = matrix.clone()
    reference_inputs = [item.clone() for item in matrix]
    expected = torch.stack(
        [newton_schulz(item, COEFFICIENTS, steps) for item in reference_inputs]
    )
    with plan.override(_small_plans()):
        actual = newton_schulz(matrix, COEFFICIENTS, steps, backend="auto")
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.02)
    if dtype == torch.bfloat16:
        assert torch.equal(matrix, torch.stack(reference_inputs))
    else:
        assert torch.equal(matrix, before)


def test_ns_batch_graph_replay_uses_current_gradient():
    torch.manual_seed(89)
    gradient = torch.randn(3, 128, 64, device="cuda", dtype=torch.float32)
    with plan.override(_small_plans()):
        newton_schulz(gradient, COEFFICIENTS, 3, backend="auto")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = newton_schulz(gradient, COEFFICIENTS, 3, backend="auto")
        gradient.copy_(torch.randn_like(gradient))
        graph.replay()
        expected = newton_schulz(gradient, COEFFICIENTS, 3, backend="auto")
    assert torch.equal(output, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("steps", [0, 3])
def test_batched_muon_matches_single_matrix_state_and_skips_missing_gradients(
    dtype, nesterov, steps
):
    torch.manual_seed(97)
    reference_model = _Matrices(dtype)
    models = [reference_model, _Matrices(dtype), _Matrices(dtype)]
    initial = copy.deepcopy(reference_model.state_dict())
    for model in models[1:]:
        model.load_state_dict(initial)
    inactive = reference_model.inactive.clone()
    bias = reference_model.bias.clone()
    options = dict(lr=1e-3, weight_decay=0.1, nesterov=nesterov, ns_steps=steps)
    optimizers = [
        MuonAdamW(reference_model, **options),
        MuonAdamW(models[1], use_ns_kernels=True, ns_batch_size=1, **options),
        MuonAdamW(models[2], use_ns_kernels=True, ns_batch_size=4, **options),
    ]
    with plan.override(_small_plans()):
        for step in range(3):
            _assign_gradients(models, 101 + step)
            for optimizer in optimizers:
                optimizer.step()
            for model, optimizer in zip(models[1:], optimizers[1:]):
                _assert_momentum_equal(optimizers[0], optimizer, reference_model, model)
                for expected, actual in zip(reference_model.weights, model.weights):
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=0.003 if dtype == torch.bfloat16 else 0.00005,
                        rtol=0.0001,
                    )
                assert torch.equal(model.inactive, inactive)
                assert torch.equal(model.bias, bias)


@pytest.mark.parametrize("nesterov", [False, True])
def test_batched_muon_checkpoint_continues_parameters_and_momentum(nesterov):
    torch.manual_seed(107)
    model = _Matrices(torch.bfloat16)
    optimizer = MuonAdamW(
        model, lr=1e-3, nesterov=nesterov, use_ns_kernels=True, ns_batch_size=4
    )
    with plan.override(_small_plans()):
        for step in range(2):
            _assign_gradients([model], 109 + step)
            optimizer.step()
        saved_model = copy.deepcopy(model.state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        resumed_model = _Matrices(torch.bfloat16)
        resumed_model.load_state_dict(saved_model)
        resumed_optimizer = MuonAdamW(
            resumed_model,
            lr=1e-3,
            nesterov=nesterov,
            use_ns_kernels=True,
            ns_batch_size=4,
        )
        resumed_optimizer.load_state_dict(saved_optimizer)
        assert saved_optimizer.keys() == {"muon", "adamw"}
        for step in range(2):
            _assign_gradients([model, resumed_model], 113 + step)
            optimizer.step()
            resumed_optimizer.step()
            for expected, actual in zip(model.parameters(), resumed_model.parameters()):
                assert torch.equal(actual, expected)
            _assert_momentum_equal(optimizer, resumed_optimizer, model, resumed_model)


def test_batched_matrix_muon_graph_replay_uses_new_gradients_and_current_state():
    torch.manual_seed(127)
    captured_model = _Matrices(torch.bfloat16)
    captured_optimizer = MuonAdamW(
        captured_model, lr=1e-3, nesterov=False, use_ns_kernels=True, ns_batch_size=4
    )
    _assign_gradients([captured_model], 131)
    with plan.override(_small_plans()):
        captured_optimizer.muon.step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_optimizer.muon.step()
        reference_model = _Matrices(torch.bfloat16)
        reference_model.load_state_dict(copy.deepcopy(captured_model.state_dict()))
        reference_optimizer = MuonAdamW(
            reference_model,
            lr=1e-3,
            nesterov=False,
            use_ns_kernels=True,
            ns_batch_size=4,
        )
        reference_optimizer.load_state_dict(
            copy.deepcopy(captured_optimizer.state_dict())
        )
        torch.manual_seed(137)
        for captured_param, reference_param in zip(
            captured_model.weights, reference_model.weights
        ):
            gradient = torch.randn_like(captured_param)
            captured_param.grad.copy_(gradient)
            reference_param.grad = gradient.clone()
        graph.replay()
        reference_optimizer.muon.step()
    for expected, actual in zip(
        reference_model.parameters(), captured_model.parameters()
    ):
        assert torch.equal(actual, expected)
    _assert_momentum_equal(
        reference_optimizer, captured_optimizer, reference_model, captured_model
    )
