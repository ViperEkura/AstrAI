"""Accumulation precision, DDP parity and main-gradient consumers."""

import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from astrai.config import ComponentSpec, PrecisionConfig
from astrai.optimization.optimizer.muon.optimizer import MuonAdamW
from astrai.optimization.optimizer.precision import PrecisionOptimizer
from astrai.parallel.executor import DDPExecutor, NoneExecutor
from astrai.parallel.executor.gradients import parameter_gradient
from astrai.trainer.callbacks.metric_util import GradSNRTracker, grad_norm
from astrai.trainer.components import build_optimizer_component
from tests.support.precision import assert_precision_state


def _distributed_check(
    rank,
    world_size,
    rendezvous,
    bucket_view,
    static_graph,
    backend="gloo",
    compute_dtype=torch.bfloat16,
    gradient_dtype="float32",
):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend, rank=rank, world_size=world_size, init_method=rendezvous
    )
    try:
        executor = DDPExecutor(
            grad_accum_steps=3,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=bucket_view,
            static_graph=static_graph,
            grad_accum_dtype=gradient_dtype,
        )
        model, optimizer, _ = executor.prepare(
            lambda: (
                nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16), nn.Linear(16, 1))
                .to(device)
                .to(compute_dtype)
            ),
            lambda model: PrecisionOptimizer(
                torch.optim.SGD(model.parameters(), lr=1e-3)
            ),
        )
        for window in range(4):
            expected = [
                torch.zeros_like(p, dtype=getattr(torch, gradient_dtype))
                for p in model.parameters()
            ]
            micros = 3
            for micro in range(micros):
                torch.manual_seed(100 + window * 10 + rank * 3 + micro)
                x = torch.randn(2, 16, device=device).to(compute_dtype)
                reference = copy.deepcopy(model.module)
                reference(x).float().sum().backward()
                for total, p in zip(expected, reference.parameters()):
                    total.add_(p.grad.float())
                with executor.accumulate_explicit(model, sync=micro == micros - 1):
                    executor.backward(model(x).float().sum())
            for total, p in zip(expected, model.parameters()):
                dist.all_reduce(total)
                total.div_(world_size)
                torch.testing.assert_close(parameter_gradient(p), total, atol=0, rtol=0)
                assert p.grad is None
            optimizer.step()
            optimizer.zero_grad(set_to_none=window % 2 == 0)
            for p in model.parameters():
                gradient = parameter_gradient(p)
                assert gradient is None or torch.count_nonzero(gradient) == 0
            # All ranks must publish exactly the same compute weights.
            for p in model.parameters():
                peer = p.detach().clone()
                dist.broadcast(peer, src=0)
                torch.testing.assert_close(p, peer, atol=0, rtol=0)
        for compiled in (False, True):
            snapshot_model = (
                torch.compile(model, backend="eager") if compiled else model
            )
            with executor.checkpoint_context(
                snapshot_model, optimizer=optimizer, export_dtype="float32"
            ) as (exported, extra):
                if rank == 0:
                    masters = optimizer.master_state_dict(
                        model.module.named_parameters()
                    )
                    assert set(exported) == set(masters)
                    for name, master in masters.items():
                        torch.testing.assert_close(
                            exported[name], master.cpu(), atol=0, rtol=0
                        )
                    native = executor.model_state_dict(model)
                    for name, value in native.items():
                        torch.testing.assert_close(
                            extra["training_model"][name], value.cpu(), atol=0, rtol=0
                        )
                else:
                    assert exported is None and extra == {}
        native_executor = DDPExecutor(grad_accum_dtype="model")
        native_model, native_optimizer, _ = native_executor.prepare(
            lambda: nn.Linear(2, 2).to(device),
            lambda module: torch.optim.SGD(module.parameters(), lr=0.01),
        )
        with native_executor.checkpoint_context(
            torch.compile(native_model, backend="eager"),
            optimizer=native_optimizer,
            export_dtype="bfloat16",
        ) as (exported, extra):
            if rank == 0:
                assert set(exported) == {"weight", "bias"}
                for name, value in native_executor.model_state_dict(
                    native_model
                ).items():
                    torch.testing.assert_close(
                        exported[name], value.cpu().bfloat16(), atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        extra["training_model"][name], value.cpu(), atol=0, rtol=0
                    )
            else:
                assert exported is None and extra == {}
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("executor_type", [NoneExecutor, DDPExecutor])
def test_fp32_accumulation_preserves_cancellation_and_drives_update(executor_type):
    executor = executor_type(grad_accum_steps=3)
    model, optimizer, _ = executor.prepare(
        lambda: nn.Linear(1, 1, bias=False).bfloat16(),
        lambda model: PrecisionOptimizer(torch.optim.SGD(model.parameters(), lr=0.1)),
    )
    before = optimizer.state_dict()["master_params"][0].clone()
    for index, value in enumerate((1.0, 0.00390625, -1.0)):
        with executor.accumulate_explicit(model, sync=index == 2):
            executor.backward(
                model(torch.tensor([[value]], dtype=torch.bfloat16)).float().sum()
            )
    gradient = parameter_gradient(model.weight)
    assert gradient.dtype == torch.float32
    assert gradient.item() == 0.00390625
    assert model.weight.grad is None
    executor.scale_gradients(model, 2.0)
    norm = executor.clip_grad_norm(model, 0.001)
    assert norm == 0.0078125
    assert abs(grad_norm(model) - 0.001) < 2e-7
    tracker = GradSNRTracker()
    tracker.update(model)
    assert tracker._first[id(model.weight)].dtype == torch.float32
    wanted = before - gradient * 0.1
    optimizer.step()
    torch.testing.assert_close(
        optimizer.state_dict()["master_params"][0], wanted, atol=0, rtol=0
    )
    optimizer.zero_grad()
    assert parameter_gradient(model.weight) is None


def test_unused_parameters_and_new_windows_do_not_reuse_stale_gradients():
    executor = NoneExecutor()
    model, optimizer, _ = executor.prepare(
        lambda: nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1)).bfloat16(),
        lambda model: PrecisionOptimizer(
            torch.optim.AdamW(model.parameters(), lr=1e-3)
        ),
    )
    executor.backward(model[0](torch.ones(1, 1).bfloat16()).float().sum())
    assert parameter_gradient(model[1].weight) is None
    optimizer.step()
    optimizer.zero_grad()
    executor.backward(model[1](torch.ones(1, 1).bfloat16()).float().sum())
    assert parameter_gradient(model[0].weight) is None
    assert parameter_gradient(model[1].weight) is not None


def test_model_precision_opt_out_and_fp32_model_remain_native():
    for dtype, policy in ((torch.bfloat16, "model"), (torch.float32, "float32")):
        executor = NoneExecutor(grad_accum_dtype=policy)
        model, _, _ = executor.prepare(lambda: nn.Linear(1, 1).to(dtype))
        executor.backward(model(torch.ones(1, 1, dtype=dtype)).float().sum())
        assert model.weight.grad is not None
        assert not hasattr(model.weight, "main_grad")


@pytest.mark.parametrize(
    "bucket_view,static_graph", [(False, False), (True, False), (True, True)]
)
def test_two_rank_accumulation_and_rebuilt_buckets(tmp_path, bucket_view, static_graph):
    rendezvous = Path(tmp_path / "gloo").as_uri()
    mp.spawn(
        _distributed_check,
        args=(2, rendezvous, bucket_view, static_graph),
        nprocs=2,
        join=True,
    )


def _conditional_check(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous)
    try:
        executor = DDPExecutor(find_unused_parameters=True, bucket_cap_mb=0.00001)

        def build():
            model = nn.Linear(2, 1).bfloat16()
            model.register_parameter("unused", nn.Parameter(torch.ones(1).bfloat16()))
            return model

        model, optimizer, _ = executor.prepare(
            build,
            lambda model: PrecisionOptimizer(
                torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
            ),
        )
        raw = executor.model_for_inference(model)
        for window in range(3):
            # All ranks have the same graph output, but only rank zero uses bias.
            output = torch.nn.functional.linear(
                torch.ones(1, 2).bfloat16(),
                raw.weight,
                raw.bias if rank == 0 else None,
            )
            executor.backward(output.float().sum())
            torch.testing.assert_close(
                parameter_gradient(raw.weight), torch.ones_like(raw.weight).float()
            )
            torch.testing.assert_close(
                parameter_gradient(raw.bias), torch.full_like(raw.bias, 0.5).float()
            )
            assert parameter_gradient(raw.unused) is None
            optimizer.step()
            optimizer.zero_grad()
            assert raw.unused.item() == 1.0
            for parameter in model.parameters():
                peer = parameter.detach().clone()
                dist.broadcast(peer, src=0)
                torch.testing.assert_close(parameter, peer, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_unused_parameter_rank_agreement_and_no_spurious_decay(tmp_path):
    mp.spawn(
        _conditional_check,
        args=(Path(tmp_path / "conditional").as_uri(),),
        nprocs=2,
        join=True,
    )


def test_unaware_optimizer_is_rejected_without_leaving_stale_hooks():
    model = nn.Linear(1, 1).bfloat16()
    executor = NoneExecutor()
    with pytest.raises(TypeError, match="supporting main_grad"):
        executor.prepare(
            lambda: model, lambda model: torch.optim.SGD(model.parameters(), lr=0.1)
        )
    assert not hasattr(model.weight, "main_grad")
    model(torch.ones(1, 1).bfloat16()).float().sum().backward()
    assert model.weight.grad is not None


def test_checkpoint_continuity_includes_accumulated_fp32_updates():
    executor = NoneExecutor()

    def build():
        return executor.prepare(
            lambda: nn.Linear(1, 1, bias=False).bfloat16(),
            lambda model: PrecisionOptimizer(
                torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
            ),
        )[:2]

    model, optimizer = build()
    resumed = None
    for step in range(6):
        runs = [(model, optimizer)]
        if resumed is not None:
            runs.append(resumed)
        for current, current_optimizer in runs:
            for micro, value in enumerate((1.0, 0.00390625, -1.0)):
                with executor.accumulate_explicit(current, sync=micro == 2):
                    executor.backward(
                        current(torch.tensor([[value]]).bfloat16()).float().sum()
                    )
            executor.scale_gradients(current, 1 / 3)
            current_optimizer.step()
            current_optimizer.zero_grad()
        if resumed is not None:
            torch.testing.assert_close(
                resumed[1].state_dict()["master_params"][0],
                optimizer.state_dict()["master_params"][0],
                atol=0,
                rtol=0,
            )
        if step == 2:
            saved = copy.deepcopy(optimizer.state_dict())
            resumed = build()
            resumed[1].load_state_dict(saved)


def _buffer_sync_check(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous)
    try:
        for init_sync in (True, False):
            for forward_sync_buffers in (True, False):
                raw = nn.Linear(2, 2, bias=False).bfloat16()
                raw.weight.data.fill_(1 + rank)
                raw.register_buffer("counter", torch.tensor(float(rank)))
                executor = DDPExecutor(
                    init_sync=init_sync, forward_sync_buffers=forward_sync_buffers
                )
                model, _, _ = executor.prepare(lambda: raw)
                assert raw.counter.item() == (0 if init_sync else rank)
                assert torch.all(raw.weight == (1 if init_sync else 1 + rank))
                raw.counter.fill_(10 + rank)
                model(torch.ones(1, 2).bfloat16())
                assert raw.counter.item() == (10 if forward_sync_buffers else 10 + rank)
                executor._grad_accumulator.close()
    finally:
        dist.destroy_process_group()


def test_bf16_initial_buffer_sync_is_separate_from_forward_sync(tmp_path):
    mp.spawn(
        _buffer_sync_check,
        args=(Path(tmp_path / "buffers").as_uri(),),
        nprocs=2,
        join=True,
    )


def test_fp32_compute_bf16_main_gradients_reduce_across_ranks(tmp_path):
    mp.spawn(
        _distributed_check,
        args=(
            2,
            Path(tmp_path / "bf16_gradients").as_uri(),
            True,
            False,
            "gloo",
            torch.float32,
            "bfloat16",
        ),
        nprocs=2,
        join=True,
    )


def _owner_optimizer(
    model, kind, shard_state, cpu_offload, state_dtype, master_dtype=torch.float32
):
    if kind == "muon":
        native = MuonAdamW(model, lr=2e-5, use_ns_kernels=True, ns_batch_size=4)
    elif kind == "adamw":
        native = torch.optim.AdamW(model.parameters(), lr=2e-5)
    else:
        native = torch.optim.SGD(
            model.parameters(), lr=2e-5, momentum=0.9, weight_decay=0.1
        )
    return PrecisionOptimizer(
        native,
        shard_state=shard_state,
        cpu_offload=cpu_offload,
        state_dtype=state_dtype,
        master_dtype=master_dtype,
    )


def _owner_state_check(rank, rendezvous, directory, backend, cpu_offload):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(backend, rank=rank, world_size=2, init_method=rendezvous)
    try:
        cases = [
            (kind, state_dtype, master_dtype)
            for kind in ("sgd", "adamw", "muon")
            for state_dtype in (torch.float32, torch.bfloat16)
            for master_dtype in (torch.float32, torch.bfloat16)
        ]
        cases.append(("single", torch.float32, torch.float32))
        for case, (kind, state_dtype, master_dtype) in enumerate(cases):
            compute_dtype = (
                torch.bfloat16 if master_dtype == torch.float32 else torch.float32
            )
            torch.manual_seed(73)
            model = nn.Module()
            if kind == "single":
                model.norm = nn.Parameter(
                    torch.ones(8, device=device, dtype=compute_dtype)
                )
            else:
                model.matrix = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.randn(16, 32, device=device, dtype=compute_dtype)
                        )
                        for _ in range(4)
                    ]
                )
                model.embed = nn.Embedding(32, 16, device=device, dtype=compute_dtype)
                model.lm_head = nn.Linear(
                    16, 32, bias=False, device=device, dtype=compute_dtype
                )
                model.lm_head.weight = model.embed.weight
                model.norm = nn.Parameter(
                    torch.ones(16, device=device, dtype=compute_dtype)
                )
            model.register_buffer("counter", torch.tensor(7, device=device))
            reference = copy.deepcopy(model)
            actual = _owner_optimizer(
                model, kind, True, cpu_offload, state_dtype, master_dtype
            )
            expected = _owner_optimizer(
                reference, kind, False, False, state_dtype, master_dtype
            )
            counts = torch.tensor(len(actual._model_params), device=device)
            dist.all_reduce(counts)
            assert counts.item() == len(list(model.parameters()))
            for step in range(6):
                torch.manual_seed(130 + step)
                for index, (param, target) in enumerate(
                    zip(model.parameters(), reference.parameters())
                ):
                    gradient = torch.randn_like(param, dtype=torch.float32) * 0.1
                    if index == 0 and step % 3 == 0:
                        gradient = None
                    param.main_grad = gradient
                    target.main_grad = None if gradient is None else gradient.clone()
                actual.step()
                expected.step()
                for param, target in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(param, target, rtol=0, atol=0)
                actual.zero_grad(set_to_none=step % 2 == 0)
                expected.zero_grad(set_to_none=step % 2 == 0)
                assert all(
                    parameter_gradient(param) is None
                    or torch.count_nonzero(parameter_gradient(param)) == 0
                    for param in model.parameters()
                )
                if step == 2:
                    filename = Path(directory) / f"owner-{case}.pt"
                    with actual.checkpoint_context():
                        if rank == 0:
                            saved = copy.deepcopy(actual.state_dict())
                            torch.save(saved, filename)
                            assert_precision_state(
                                saved, expected.state_dict(), rtol=1e-6, atol=1e-7
                            )
                    with pytest.raises(RuntimeError, match="checkpoint_context"):
                        actual.state_dict()
                    dist.barrier()
                    saved = torch.load(filename, map_location="cpu", weights_only=True)
                    model = copy.deepcopy(model)
                    actual = _owner_optimizer(
                        model, kind, True, cpu_offload, state_dtype, master_dtype
                    )
                    actual.load_state_dict(saved)
            executor = DDPExecutor(grad_accum_dtype="float32")
            for export_dtype in (None, "float16", "float32"):
                with executor.checkpoint_context(
                    model, optimizer=actual, export_dtype=export_dtype
                ) as (exported, extra):
                    if rank != 0:
                        assert exported is None and extra == {}
                        continue
                    assert exported["counter"].dtype == torch.int64
                    masters = expected.master_state_dict(
                        reference.named_parameters(remove_duplicate=False)
                    )
                    for name, master in masters.items():
                        wanted = (
                            reference.state_dict()[name]
                            if export_dtype is None
                            else master.to(getattr(torch, export_dtype))
                        )
                        torch.testing.assert_close(
                            exported[name].cpu(), wanted.cpu(), rtol=0, atol=0
                        )
                    if export_dtype is not None:
                        for name, value in model.state_dict().items():
                            torch.testing.assert_close(
                                extra["training_model"][name],
                                value.cpu(),
                                rtol=0,
                                atol=0,
                            )
                    replica = copy.deepcopy(model)
                    unsharded = _owner_optimizer(
                        replica, kind, False, False, state_dtype, master_dtype
                    )
                    unsharded.load_state_dict(actual.state_dict())
                    for param, target in zip(replica.parameters(), model.parameters()):
                        torch.testing.assert_close(param, target, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "backend,cpu_offload",
    [
        ("gloo", False),
        pytest.param(
            "nccl",
            False,
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="Two GPUs required"
            ),
        ),
        pytest.param(
            "nccl",
            True,
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="Two GPUs required"
            ),
        ),
    ],
)
def test_complete_parameter_ownership_export_and_resume(tmp_path, backend, cpu_offload):
    mp.spawn(
        _owner_state_check,
        args=(Path(tmp_path / "owners").as_uri(), str(tmp_path), backend, cpu_offload),
        nprocs=2,
        join=True,
    )


def _subgroup_owner_check(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=4,
        init_method=rendezvous,
        timeout=timedelta(seconds=30),
    )
    try:
        groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        group = groups[rank // 2]
        executor = DDPExecutor(process_group=group, grad_accum_dtype="float32")
        model = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
        model.weight.data.fill_(1)
        model = executor._prepare_model(model)
        raw = executor.model_for_inference(model)
        optimizer = build_optimizer_component(
            ComponentSpec(
                type="torch.optim:SGD", params={"lr": 0.1, "shard_state": True}
            ),
            raw,
            PrecisionConfig(compute_dtype="bfloat16"),
            process_group=group,
        )
        executor.backward(
            model(torch.full((1, 2), rank // 2 + 1, dtype=torch.bfloat16)).sum()
        )
        optimizer.step()
        expected = torch.full_like(raw.weight, 1 - 0.1 * (rank // 2 + 1))
        torch.testing.assert_close(raw.weight, expected, atol=0, rtol=0)
        with executor.checkpoint_context(
            model, optimizer=optimizer, export_dtype="float32"
        ) as (state, extra):
            if rank % 2 == 0:
                torch.testing.assert_close(
                    state["weight"], torch.full((2, 2), 1 - 0.1 * (rank // 2 + 1))
                )
                saved = copy.deepcopy(optimizer.state_dict())
                assert extra["training_model"]["weight"].dtype == torch.bfloat16
            else:
                assert state is None and extra == {}
                saved = None
            data = [saved]
            dist.broadcast_object_list(data, src=rank // 2 * 2, group=group)
        optimizer.load_state_dict(data[0])
        torch.testing.assert_close(raw.weight, expected, atol=0, rtol=0)
        # Only one group saves here; peers in the other group need not enter it.
        if rank < 2:
            with executor.checkpoint_context(
                model, optimizer=optimizer, export_dtype=None
            ):
                pass
        dist.barrier()
        executor._grad_accumulator.close()
    finally:
        dist.destroy_process_group()


def test_optimizer_ownership_and_checkpoints_stay_inside_ddp_group(tmp_path):
    mp.spawn(
        _subgroup_owner_check,
        args=(Path(tmp_path / "subgroups").as_uri(),),
        nprocs=4,
        join=True,
    )
