"""AstrAI training benchmarks for single-GPU smoke and distributed comparisons.

Single-GPU modes require an idle physical GPU and limit PyTorch memory.
Distributed comparisons use --compare and --gpus with ABBA timing.
"""

import argparse
import gc
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.extension.kernel.cross_entropy import linear_cross_entropy
from astrai.model import AutoRegressiveLM
from astrai.optimization.optimizer.muon import MuonAdamW
from astrai.parallel.executor import FSDPExecutor
from astrai.trainer.callbacks.metrics import MetricCallback
from astrai.trainer.strategy.ops import _collect_moe_diagnostics

SMOKE_CASES = [
    {
        "name": "dense_fp32",
        "ffn_type": "mlp",
        "dtype": "float32",
        "batch_size": 4,
        "seq_len": 64,
        "measured_steps": 12,
    },
    {
        "name": "moe_fp32",
        "ffn_type": "moe",
        "dtype": "float32",
        "batch_size": 4,
        "seq_len": 64,
        "moe_diagnostics": True,
        "measured_steps": 12,
    },
    {
        "name": "moe_bf16_accum2",
        "ffn_type": "moe",
        "dtype": "bfloat16",
        "batch_size": 2,
        "seq_len": 64,
        "grad_accum_steps": 2,
        "moe_diagnostics": True,
        "measured_steps": 12,
        "phase_steps": 10,
    },
]


def _command(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def _gpu_info(index):
    rows = _command(
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ).splitlines()
    for row in rows:
        fields = [part.strip() for part in row.split(",")]
        if int(fields[0]) == index:
            return {
                "index": index,
                "uuid": fields[1],
                "name": fields[2],
                "total_mib": int(fields[3]),
                "used_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
            }
    raise ValueError(f"physical GPU {index} was not found")


def _require_idle_gpu(index):
    gpu = _gpu_info(index)
    try:
        apps = _command(
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ).splitlines()
    except subprocess.CalledProcessError:
        apps = []
    active = [line for line in apps if line.split(",", 1)[0].strip() == gpu["uuid"]]
    if active or gpu["used_mib"] > 1024 or gpu["utilization_percent"] > 10:
        raise RuntimeError(f"GPU {index} is not idle: {gpu}; compute apps={active}")
    return gpu


def _git_state():
    try:
        return {
            "commit": _command("git", "rev-parse", "HEAD").strip(),
            "branch": _command("git", "branch", "--show-current").strip(),
            "dirty": bool(_command("git", "status", "--porcelain").strip()),
        }
    except (OSError, subprocess.CalledProcessError):
        return None


def _summary(samples):
    ordered = sorted(samples)
    return {
        "mean_ms": round(statistics.mean(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


def _case_config(case):
    defaults = {
        "vocab_size": 4096,
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "n_activated_experts": 2,
        "batch_size": 4,
        "seq_len": 64,
        "grad_accum_steps": 1,
        "dtype": "float32",
        "ffn_type": "mlp",
        "moe_diagnostics": False,
        "warmup_steps": 3,
        "measured_steps": 12,
        "phase_steps": 4,
    }
    unknown = set(case) - set(defaults) - {"name"}
    if unknown:
        raise ValueError(f"unknown benchmark keys: {sorted(unknown)}")
    config = {**defaults, **case}
    if not config.get("name") or not isinstance(config["name"], str):
        raise ValueError("every case needs a string name")
    if config["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("dtype must be float32 or bfloat16")
    if config["ffn_type"] not in ("mlp", "moe"):
        raise ValueError("ffn_type must be mlp or moe")
    for key in (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "batch_size",
        "seq_len",
        "grad_accum_steps",
        "warmup_steps",
        "measured_steps",
        "phase_steps",
    ):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["hidden_size"] % config["num_attention_heads"]:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if config["ffn_type"] == "moe":
        if not 1 <= config["n_activated_experts"] <= config["n_routed_experts"]:
            raise ValueError("invalid MoE top-k/expert count")
    if (
        config["batch_size"] * config["seq_len"] * config["grad_accum_steps"] > 2048
        or config["hidden_size"] > 512
        or config["num_hidden_layers"] > 4
        or config["n_routed_experts"] > 8
        or config["vocab_size"] > 8192
        or config["seq_len"] > 512
    ):
        raise ValueError(f"case {config['name']} exceeds conservative safety limits")
    return config


def _run_case(case, torch):
    config = AutoRegressiveLMConfig(
        vocab_size=case["vocab_size"],
        hidden_size=case["hidden_size"],
        num_hidden_layers=case["num_hidden_layers"],
        intermediate_size=case["intermediate_size"],
        num_attention_heads=case["num_attention_heads"],
        num_key_value_heads=case["num_key_value_heads"],
        max_position_embeddings=max(512, case["seq_len"]),
        rms_norm_eps=1e-5,
        ffn_type=case["ffn_type"],
        n_routed_experts=case["n_routed_experts"],
        n_shared_experts=case["n_shared_experts"],
        n_activated_experts=case["n_activated_experts"],
    )
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[case["dtype"]]
    model = AutoRegressiveLM(config).to(device="cuda", dtype=dtype).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randint(
        case["vocab_size"], (case["batch_size"], case["seq_len"]), device="cuda"
    )
    targets = torch.randint(
        case["vocab_size"], (case["batch_size"], case["seq_len"]), device="cuda"
    )

    def step(phase_times=None):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(case["grad_accum_steps"]):
            start = time.perf_counter()
            output = model(inputs)
            loss = F.cross_entropy(
                output["logits"].float().flatten(0, 1), targets.flatten()
            )
            if "aux_loss" in output:
                loss = loss + 0.01 * output["aux_loss"]
            if phase_times is not None:
                torch.cuda.synchronize()
                phase_times["forward_loss"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            if case["moe_diagnostics"] and "router_stats" in output:
                _collect_moe_diagnostics(output["router_stats"])
            if phase_times is not None:
                torch.cuda.synchronize()
                phase_times["diagnostics"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            (loss / case["grad_accum_steps"]).backward()
            if phase_times is not None:
                torch.cuda.synchronize()
                phase_times["backward"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        optimizer.step()
        if phase_times is not None:
            torch.cuda.synchronize()
            phase_times["optimizer"].append((time.perf_counter() - start) * 1000)

    for _ in range(case["warmup_steps"]):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_samples = []
    for _ in range(case["measured_steps"]):
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        wall_samples.append((time.perf_counter() - start) * 1000)
    result = {
        "config": case,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "step_wall": _summary(wall_samples),
        "tokens_per_second": round(
            1000
            * case["batch_size"]
            * case["seq_len"]
            * case["grad_accum_steps"]
            / statistics.mean(wall_samples),
            1,
        ),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
    }
    phases = {
        name: [] for name in ("forward_loss", "diagnostics", "backward", "optimizer")
    }
    for _ in range(case["phase_steps"]):
        step(phases)
    result["sync_phase_ms"] = {
        name: _summary(values) for name, values in phases.items()
    }
    del step, model, optimizer, inputs, targets
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _run_stats(args, torch):
    """Measure metric and MoE diagnostic costs on the selected idle GPU."""

    device = "cuda"
    result = {}

    def timed(fn, repeats=30, warmup=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        ordered = sorted(samples)
        return {
            "mean_ms": round(statistics.mean(samples), 4),
            "median_ms": round(statistics.median(samples), 4),
            "p90_ms": round(ordered[int(0.9 * len(samples))], 4),
            "p95_ms": round(ordered[int(0.95 * len(samples))], 4),
        }

    def timed_abba(first, second, cycles=25):
        """Interleave alternatives to reduce clock-drift bias."""
        for _ in range(10):
            first()
            second()
        torch.cuda.synchronize()
        samples = {"one_hot": [], "scatter_add": []}
        for _ in range(cycles):
            for name, fn in (
                ("one_hot", first),
                ("scatter_add", second),
                ("scatter_add", second),
                ("one_hot", first),
            ):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
        return {
            key: {
                "median_ms": round(statistics.median(values), 4),
                "mean_ms": round(statistics.mean(values), 4),
            }
            for key, values in samples.items()
        }

    # Single-device metric overhead; distributed noise probes require DDP.
    layers = []
    for _ in range(8):
        layers.extend((nn.Linear(1024, 1024), nn.GELU()))
    model = nn.Sequential(*layers).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(16, 1024, device=device)
    ctx = SimpleNamespace(
        model=model,
        val_dataloader=None,
        optimizer_step=0,
        epoch=0,
        consumed_samples=0,
        metrics={},
        loss=0.0,
        optimizer=opt,
        dp_size=1,
    )
    default_metric = MetricCallback()

    def train_step(track):
        opt.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        if track == "default_metric":
            default_metric.before_optimizer_step(ctx)
        opt.step()

    result["train_params"] = params
    result["train_baseline"] = timed(lambda: train_step(None), repeats=20)
    result["train_default_metric"] = timed(
        lambda: train_step("default_metric"), repeats=20
    )
    del model, opt, x, ctx, default_metric, train_step
    gc.collect()
    torch.cuda.empty_cache()

    n, k, e = 4096, 2, 64
    indices = torch.randint(e, (n, k), device=device)

    def one_hot():
        return F.one_hot(indices, e).sum(dim=(0, 1)).float()

    def bincount():
        return torch.bincount(indices.reshape(-1), minlength=e).float()

    def scatter():
        out = torch.zeros(e, device=device, dtype=torch.float32)
        return out.scatter_add_(
            0, indices.reshape(-1), torch.ones(n * k, device=device)
        )

    assert torch.equal(one_hot(), bincount())
    assert torch.equal(one_hot(), scatter())
    for name, fn in (
        ("one_hot", one_hot),
        ("bincount", bincount),
        ("scatter_add", scatter),
    ):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result[name] = timed(fn, repeats=100, warmup=10)
        result[name]["peak_temporary_mib"] = round(
            (torch.cuda.max_memory_allocated() - base) / 2**20, 2
        )
    result["count_abba"] = timed_abba(one_hot, scatter)
    result["peak_process_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
    del indices, one_hot, bincount, scatter, fn
    gc.collect()
    torch.cuda.empty_cache()

    # Real AstrAI MoE forward/backward with conservative model and tokens.
    cfg = AutoRegressiveLMConfig(
        vocab_size=4096,
        hidden_size=256,
        num_hidden_layers=2,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        ffn_type="moe",
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
    )
    moe_model = AutoRegressiveLM(cfg).to(device).train()
    moe_opt = torch.optim.AdamW(moe_model.parameters(), lr=1e-4)
    input_ids = torch.randint(
        cfg.vocab_size, (args.batch_size, args.seq_len), device=device
    )
    targets = torch.randint(
        cfg.vocab_size, (args.batch_size, args.seq_len), device=device
    )
    phase_samples = {
        k: [] for k in ("forward", "diagnostics", "backward", "optimizer", "step")
    }

    def model_step(record):
        moe_opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = moe_model(input_ids)
        loss = (
            F.cross_entropy(output["logits"].float().flatten(0, 1), targets.flatten())
            + 0.01 * output["aux_loss"]
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        _collect_moe_diagnostics(output["router_stats"])
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        moe_opt.step()
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        if record:
            for key, a, b in (
                ("forward", t0, t1),
                ("diagnostics", t1, t2),
                ("backward", t2, t3),
                ("optimizer", t3, t4),
                ("step", t0, t4),
            ):
                phase_samples[key].append((b - a) * 1000)

    for _ in range(3):
        model_step(False)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps):
        model_step(True)
    result["astrai_moe_train"] = {
        "shape": f"2 layers, hidden 256, 8 experts, batch {args.batch_size} x {args.seq_len}",
        "warmup_steps": 3,
        "measured_steps": args.steps,
        "median_ms": {
            key: round(statistics.median(values), 3)
            for key, values in phase_samples.items()
        },
        "mean_ms": {
            key: round(statistics.mean(values), 3)
            for key, values in phase_samples.items()
        },
        "p95_ms": {
            key: round(sorted(values)[int(0.95 * len(values))], 3)
            for key, values in phase_samples.items()
        },
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }
    return result


def _run_muon(args, torch):
    """Measure single-stream Muon and optionally compare NS buffer reuse."""

    result = {
        "model_path": str(args.model_path),
        "optimizer": "MuonAdamW",
        "ns_steps": 5,
        "dtype": "bfloat16",
        "gradient_seed": 17,
        "warmup_steps": 3,
        "measured_steps": args.steps,
    }
    variants = [("reference", False)]
    if args.muon_reuse_ns_buffers:
        variants.append(("reuse_ns_buffers", True))
    expected = None
    for name, reuse in variants:
        torch.manual_seed(7)
        model = AutoRegressiveLM.from_pretrained(args.model_path).to(
            device="cuda", dtype=torch.bfloat16
        )
        kwargs = {"reuse_ns_buffers": True} if reuse else {}
        optimizer = MuonAdamW(model, lr=1e-5, ns_steps=5, **kwargs)
        torch.manual_seed(17)
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        for _ in range(3):
            optimizer.step()
        torch.cuda.synchronize()
        if name == "reference" and args.muon_reuse_ns_buffers:
            expected = {
                key: param.detach().cpu().clone()
                for key, param in model.named_parameters()
            }
        elif reuse:
            max_abs = 0.0
            different_elements = 0
            for key, param in model.named_parameters():
                current = param.detach().cpu()
                if not torch.equal(current, expected[key]):
                    difference = current.float() - expected[key].float()
                    max_abs = max(max_abs, difference.abs().max().item())
                    different_elements += torch.count_nonzero(difference).item()
            result["update_equivalence"] = {
                "max_abs": max_abs,
                "different_elements": different_elements,
            }
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.steps):
            start = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        result[name] = {
            "optimizer_step": _summary(samples),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
        }
        result["parameter_count"] = sum(p.numel() for p in model.parameters())
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return result


COMPARES = (
    "muon-step",
    "muon-reshard",
    "muon-train",
    "wrap",
    "ce",
    "microbatch",
    "reshard",
    "no-sync",
    "keep-unsharded",
    "ddp",
)
_OPTIMIZER_ONLY = {"muon-step", "muon-reshard"}
_REBUILD = {"wrap", "ddp"}
_KERNEL_COMPARISONS = {"muon-step", "muon-train"}
_RESHARD_COMPARISONS = {"muon-reshard", "reshard"}


def validate_train_args(args):
    if args.batch is None:
        args.batch = (
            2 if args.compare in {"muon-train", "wrap", "ce", "microbatch"} else 4
        )
    if args.accum is None:
        args.accum = (
            8 if args.compare in {"muon-train", "wrap", "ce", "microbatch"} else 4
        )
    for name in ("batch", "seq", "accum", "rounds", "ns_batch_size", "loss_chunk_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if args.compare == "microbatch" and args.accum % 2:
        raise ValueError("--accum must be even for the microbatch comparison")


def _names(args):
    if args.compare == "microbatch":
        return (
            f"batch{args.batch}_accum{args.accum}",
            f"batch{args.batch * 2}_accum{args.accum // 2}",
        )
    return {
        "muon-step": ("independent_ns", "batched_ns"),
        "muon-reshard": ("scatter", "local_shard"),
        "muon-train": ("independent_ns", "batched_ns"),
        "wrap": ("coarse", "layer"),
        "ce": ("torch_ce", "fused_ce"),
        "reshard": ("scatter", "local_shard"),
        "no-sync": ("recursive", "per_module"),
        "keep-unsharded": ("reshard_backward", "keep_unsharded"),
        "ddp": ("fsdp", "ddp"),
    }[args.compare]


def _build(args, local_rank, candidate):
    torch.manual_seed(7)
    model = (
        AutoRegressiveLM.from_pretrained(args.model)
        .to(device="cuda", dtype=torch.bfloat16)
        .train()
    )
    executor = None
    if args.compare == "ddp" and candidate:
        model = DDP(model, device_ids=[local_rank])
    else:
        layerwise = args.compare not in {
            "muon-step",
            "muon-reshard",
            "muon-train",
        } and not (args.compare == "wrap" and not candidate)
        executor = FSDPExecutor(grad_accum_steps=args.accum, layerwise=layerwise)
        model = executor._prepare_model(model)
    parameters = sum(param.numel() for param in model.parameters())
    optimizer = MuonAdamW(
        model,
        lr=1e-5,
        ns_steps=5,
        use_ns_kernels=args.compare not in _KERNEL_COMPARISONS,
        ns_batch_size=args.ns_batch_size,
    )
    if args.compare in _OPTIMIZER_ONLY:
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.ones_like(param) * 0.01
    return {
        "model": model,
        "optimizer": optimizer,
        "executor": executor,
        "modules": [m for m in model.modules() if isinstance(m, FSDPModule)],
        "parameters": parameters,
    }


@contextmanager
def _manual_no_sync(modules, *, recurse, keep_unsharded):
    if keep_unsharded:
        for module in modules:
            module.set_reshard_after_backward(False, recurse=False)
    for module in modules:
        module.set_requires_gradient_sync(False, recurse=recurse)
    try:
        yield
    finally:
        for module in modules:
            module.set_requires_gradient_sync(True, recurse=recurse)
        if keep_unsharded:
            for module in modules:
                module.set_reshard_after_backward(True, recurse=False)


def _no_sync(state, args, candidate):
    model = state["model"]
    if isinstance(model, DDP):
        return model.no_sync()
    if args.compare == "ddp":
        return state["executor"]._no_sync(model)
    return _manual_no_sync(
        state["modules"],
        recurse=not (args.compare == "no-sync" and candidate),
        keep_unsharded=args.compare == "keep-unsharded" and candidate,
    )


@contextmanager
def _muon_reshard(args, candidate):
    if args.compare not in _RESHARD_COMPARISONS:
        yield
        return
    module = importlib.import_module(
        "astrai.optimization.optimizer.muon.sharded_update"
    )
    distribute = module.distribute_tensor

    def variant(tensor, mesh, placements, **kwargs):
        return distribute(
            tensor, mesh, placements, src_data_rank=None if candidate else 0
        )

    module.distribute_tensor = variant
    try:
        yield
    finally:
        module.distribute_tensor = distribute


def _inputs(args, rank):
    torch.manual_seed(20261012 + rank)
    batch = args.batch * 2 if args.compare == "microbatch" else args.batch
    return (
        torch.randint(0, 100000, (batch, args.seq), device="cuda"),
        torch.randint(0, 100000, (batch, args.seq), device="cuda"),
    )


def _step(state, args, candidate, inputs, targets):
    model = state["model"]
    optimizer = state["optimizer"]
    with _muon_reshard(args, candidate):
        if args.compare in _KERNEL_COMPARISONS:
            optimizer.muon.use_ns_kernels = candidate
        if args.compare in _OPTIMIZER_ONLY:
            optimizer.step()
            return None
        optimizer.zero_grad(set_to_none=True)
        accum = (
            args.accum // 2
            if args.compare == "microbatch" and candidate
            else args.accum
        )
        for microbatch in range(accum):
            scope = (
                nullcontext()
                if microbatch == accum - 1
                else _no_sync(state, args, candidate)
            )
            with scope:
                if args.compare == "microbatch" and not candidate:
                    index = microbatch % 2
                    x = inputs[index * args.batch : (index + 1) * args.batch]
                    y = targets[index * args.batch : (index + 1) * args.batch]
                else:
                    x, y = inputs, targets
                if args.compare in {"muon-train", "wrap"} or (
                    args.compare == "ce" and not candidate
                ):
                    logits = model(x)["logits"]
                    loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
                else:
                    outputs = model(x, skip_lm_head=True, return_lm_head_weight=True)
                    if isinstance(model, DDP):
                        loss = (
                            linear_cross_entropy(
                                outputs["hidden_states"].flatten(0, 1),
                                outputs["lm_head_weight"],
                                y.flatten(),
                                chunk_size=args.loss_chunk_size,
                            )
                            / y.numel()
                        )
                    else:
                        loss = (
                            model.lm_head(
                                outputs["hidden_states"],
                                targets=y,
                                chunk_size=args.loss_chunk_size,
                            )
                            / y.numel()
                        )
                (loss / accum).backward()
        optimizer.step()
        return loss.detach()


def _measure(state, args, candidate, inputs, targets, name):
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss = _step(state, args, candidate, inputs, targets)
    torch.cuda.synchronize()
    elapsed = torch.tensor([1000 * (time.perf_counter() - start)], device="cuda")
    peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**20], device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    peaks = [torch.empty_like(peak) for _ in range(dist.get_world_size())]
    dist.all_gather(peaks, peak)
    peak_by_rank = [round(value.item(), 1) for value in peaks]
    return {
        "variant": name,
        "ms": round(elapsed.item(), 3),
        "peak_mib": max(peak_by_rank),
        "peak_mib_by_rank": peak_by_rank,
        "loss": None if loss is None else float(loss),
    }


def _release(state):
    state.clear()
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


def run_train(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        names = _names(args)
        inputs, targets = (
            (None, None) if args.compare in _OPTIMIZER_ONLY else _inputs(args, rank)
        )
        records = []
        parameters = None
        if args.compare in _REBUILD:
            for _ in range(args.rounds):
                for candidate in (False, True, True, False):
                    state = _build(args, local_rank, candidate)
                    parameters = state["parameters"]
                    name = names[int(candidate)]
                    for _ in range(args.warmup):
                        _step(state, args, candidate, inputs, targets)
                    record = _measure(state, args, candidate, inputs, targets, name)
                    if rank == 0:
                        records.append(record)
                        print(json.dumps(record), flush=True)
                    _release(state)
        else:
            state = _build(args, local_rank, False)
            parameters = state["parameters"]
            for _ in range(args.warmup):
                for candidate in (False, True):
                    _step(state, args, candidate, inputs, targets)
            for _ in range(args.rounds):
                for candidate in (False, True, True, False):
                    name = names[int(candidate)]
                    record = _measure(state, args, candidate, inputs, targets, name)
                    if rank == 0:
                        records.append(record)
                        print(json.dumps(record), flush=True)
        if rank == 0:
            tokens = (
                None
                if args.compare in _OPTIMIZER_ONLY
                else args.batch * args.seq * args.accum * dist.get_world_size()
            )
            samples = {
                name: [r["ms"] for r in records if r["variant"] == name]
                for name in names
            }
            peaks = {
                name: [r["peak_mib"] for r in records if r["variant"] == name]
                for name in names
            }
            result = {
                "benchmark": "train",
                "args": {
                    key: getattr(args, key)
                    for key in (
                        "compare",
                        "model",
                        "output",
                        "gpus",
                        "batch",
                        "seq",
                        "accum",
                        "rounds",
                        "warmup",
                        "ns_batch_size",
                        "loss_chunk_size",
                    )
                },
                "model_parameters": parameters,
                "world_size": dist.get_world_size(),
                "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "precision": "bf16",
                "input_tokens_per_step": tokens,
                "supervised_tokens_per_step": tokens,
                "records": records,
                "samples_ms": samples,
                "peak_allocated_mib": peaks,
                "median_ms": {
                    name: statistics.median(values) for name, values in samples.items()
                },
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


def _run_distributed_cli(args, parser):
    if args.mode != "train":
        parser.error("--compare applies only to train mode")
    try:
        validate_train_args(args)
        if args.worker:
            run_train(args)
            return
        if not args.gpus:
            raise ValueError("--gpus is required for a distributed train benchmark")
        gpus = [item.strip() for item in args.gpus.split(",")]
        if len(gpus) < 2 or len(set(gpus)) != len(gpus):
            raise ValueError("--gpus needs at least two distinct GPU indices")
        if any(not item.isdecimal() for item in gpus):
            raise ValueError("--gpus must contain nonnegative integer indices")
    except ValueError as exc:
        parser.error(str(exc))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(gpus)}",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker",
    ]
    if args.dry_run:
        print(json.dumps({"gpus": gpus, "command": command}, indent=2))
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    subprocess.run(command, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "stats", "muon"), default="train")
    parser.add_argument("--batch-size", type=int, default=4, help="stats mode")
    parser.add_argument("--seq-len", type=int, default=64, help="stats mode")
    parser.add_argument("--steps", type=int, default=10, help="stats or muon mode")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/AstrAI-V1-base"),
        help="muon mode: local pretrained model",
    )
    parser.add_argument(
        "--muon-reuse-ns-buffers",
        action="store_true",
        help="muon mode: compare optional NS scratch-buffer reuse",
    )
    parser.add_argument(
        "--config", type=Path, help="optional JSON case matrix for train mode"
    )
    parser.add_argument("--gpu", type=int, help="physical nvidia-smi GPU index")
    parser.add_argument("--memory-fraction", type=float, default=0.15)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compare", choices=COMPARES)
    parser.add_argument("--gpus", help="physical GPU indices, e.g. 4,5,6,7")
    parser.add_argument("--model", default="models/astrai_seq_0831")
    parser.add_argument("--output")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--accum", type=int)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--ns-batch-size", type=int, default=4)
    parser.add_argument("--loss-chunk-size", type=int, default=512)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.compare:
        if not args.output:
            parser.error("--output is required with --compare")
        _run_distributed_cli(args, parser)
        return
    if args.gpu is None or args.json_out is None:
        parser.error("--gpu and --json-out are required without --compare")
    if not 0 < args.memory_fraction <= 0.5:
        parser.error("memory-fraction must be in (0, 0.5]")
    if args.mode == "stats":
        if args.config:
            parser.error("--config applies only to train mode")
        if args.batch_size < 1 or not 1 <= args.seq_len <= 256 or args.steps < 1:
            parser.error("batch-size/steps must be positive and seq-len must be 1..256")
        if args.batch_size * args.seq_len > 2048:
            parser.error("stats mode is limited to 2048 tokens per step")
        cases = None
    elif args.mode == "muon":
        if args.config:
            parser.error("--config applies only to train mode")
        if args.steps < 1:
            parser.error("muon mode requires steps >= 1")
        cases = None
    else:
        document = (
            json.loads(args.config.read_text(encoding="utf-8"))
            if args.config
            else {"cases": SMOKE_CASES}
        )
        if not isinstance(document, dict) or not isinstance(
            document.get("cases"), list
        ):
            parser.error("config must contain a cases list")
        cases = [_case_config(case) for case in document["cases"]]
        if not cases or len({case["name"] for case in cases}) != len(cases):
            parser.error("cases must be nonempty with unique names")
    if args.dry_run:
        dry_run = {"mode": args.mode, "gpu": args.gpu}
        if cases is not None:
            dry_run["case_source"] = (
                str(args.config) if args.config else "built_in_smoke"
            )
            dry_run["cases"] = cases
        else:
            if args.mode == "muon":
                dry_run.update(
                    model_path=str(args.model_path),
                    muon_reuse_ns_buffers=args.muon_reuse_ns_buffers,
                    steps=args.steps,
                )
            else:
                dry_run.update(
                    batch_size=args.batch_size, seq_len=args.seq_len, steps=args.steps
                )
        print(json.dumps(dry_run, indent=2))
        return
    gpu = _require_idle_gpu(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(7)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(),
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "memory_fraction_limit": args.memory_fraction,
        "seed": 7,
        "mode": args.mode,
    }
    if args.mode == "stats":
        result["stats"] = _run_stats(args, torch)
    elif args.mode == "muon":
        result["muon"] = _run_muon(args, torch)
    else:
        result["case_source"] = str(args.config) if args.config else "built_in_smoke"
        result["cases"] = [_run_case(case, torch) for case in cases]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
