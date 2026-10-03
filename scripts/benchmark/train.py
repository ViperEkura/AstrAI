"""Bounded, configurable single-GPU AstrAI training and statistics benchmark.

Use a free physical GPU only. The script refuses a selected GPU with another
compute process and limits PyTorch to a fraction of its memory.
"""

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

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
        "grad_snr_interval": 10,
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
        "grad_snr_interval": 0,
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
    if (
        not isinstance(config["grad_snr_interval"], int)
        or config["grad_snr_interval"] < 0
    ):
        raise ValueError("grad_snr_interval must be nonnegative")
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
    import torch.nn.functional as F

    from astrai.config.model_config import AutoRegressiveLMConfig
    from astrai.model import AutoRegressiveLM
    from astrai.trainer.metric_util import GradSNRTracker
    from astrai.trainer.strategy import _collect_moe_diagnostics

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
    tracker = GradSNRTracker() if case["grad_snr_interval"] else None
    step_index = 0
    last_snr = None

    def step(phase_times=None):
        nonlocal step_index, last_snr
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
        if tracker is not None and step_index % case["grad_snr_interval"] == 0:
            tracker.update(model, step_span=case["grad_snr_interval"])
            last_snr = tracker.snr
        if phase_times is not None:
            torch.cuda.synchronize()
            phase_times["snr"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        optimizer.step()
        if phase_times is not None:
            torch.cuda.synchronize()
            phase_times["optimizer"].append((time.perf_counter() - start) * 1000)
        step_index += 1

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
        "last_grad_snr_db": last_snr,
    }
    phases = {
        name: []
        for name in ("forward_loss", "diagnostics", "backward", "snr", "optimizer")
    }
    for _ in range(case["phase_steps"]):
        step(phases)
    result["sync_phase_ms"] = {
        name: _summary(values) for name, values in phases.items()
    }
    del step, model, optimizer, inputs, targets, tracker
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _run_stats(args, torch):
    """Measure SNR and MoE diagnostic costs on the selected idle GPU."""
    from types import SimpleNamespace

    import torch.nn as nn
    import torch.nn.functional as F

    from astrai.config.model_config import AutoRegressiveLMConfig
    from astrai.model import AutoRegressiveLM
    from astrai.trainer.metric_util import GradSNRTracker
    from astrai.trainer.strategy import _collect_moe_diagnostics
    from astrai.trainer.train_callback import MetricCallback

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

    def timed_wall(fn, repeats=20):
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000)
        return round(statistics.median(samples), 4)

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

    # 8.4 million trainable weights, ~32 MiB per full-size FP32 moment.
    layers = []
    for _ in range(8):
        layers.extend((nn.Linear(1024, 1024), nn.GELU()))
    model = nn.Sequential(*layers).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(16, 1024, device=device)
    tracker = GradSNRTracker()
    ctx = SimpleNamespace(
        model=model,
        grad_snr_tracker=tracker,
        val_dataloader=None,
        optimizer_step=0,
        epoch=0,
        consumed_samples=0,
        metrics={},
        loss=0.0,
        optimizer=opt,
        dp_size=1,
    )
    default_metric = MetricCallback(ckpt_dir=None, save_interval=100)

    def train_step(track):
        opt.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        if track == "direct_snr":
            tracker.update(model)
        elif track == "default_metric":
            default_metric.before_optimizer_step(ctx)
        opt.step()

    result["train_params"] = params
    result["train_without_snr"] = timed(lambda: train_step(None), repeats=20)
    result["train_default_metric"] = timed(
        lambda: train_step("default_metric"), repeats=20
    )
    assert not tracker._first, "default metric unexpectedly allocated SNR state"
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    result["train_with_snr"] = timed(lambda: train_step("direct_snr"), repeats=20)
    torch.cuda.synchronize()
    result["snr_extra_allocated_mib"] = round(
        (torch.cuda.memory_allocated() - allocated_before) / 2**20, 2
    )
    result["snr_db"] = round(tracker.snr, 4)
    result["snr_read_wall_ms"] = timed_wall(lambda: tracker.snr)

    def sampled_metric_step(callback, sampled_ctx):
        opt.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        sampled_ctx.optimizer_step += 1
        callback.before_optimizer_step(sampled_ctx)
        opt.step()

    for interval in (1, 10):
        sample_ctx = SimpleNamespace(**vars(ctx))
        sample_ctx.grad_snr_tracker = GradSNRTracker()
        sample_ctx.grad_snr_value = None
        callback = MetricCallback(
            ckpt_dir=None,
            save_interval=100,
            metrics=["grad_snr"],
            grad_snr_interval=interval,
        )
        result[f"train_snr_interval_{interval}"] = timed(
            lambda: sampled_metric_step(callback, sample_ctx), repeats=20
        )
        assert sample_ctx.grad_snr_value is not None
    del model, opt, x, tracker, ctx, sample_ctx, callback, default_metric
    del train_step, sampled_metric_step
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
    """Measure Muon variants with matched inputs and symmetric run order."""
    from astrai.extension.kernel.muon_ns import is_available
    from astrai.model import AutoRegressiveLM
    from astrai.optim.muon_adamw import MuonAdamW

    kernel_available = is_available()
    if args.muon_fused_ns and not kernel_available:
        raise RuntimeError("--muon-fused-ns requires the muon_ns CUDA extension")
    result = {
        "model_path": str(args.model_path),
        "optimizer": "MuonAdamW",
        "ns_steps": 5,
        "dtype": "bfloat16",
        "gradient_seed": 17,
        "warmup_steps": 3,
        "measured_steps": args.steps,
        "parity_after_steps": 3 + args.steps,
        "muon_ns_available": kernel_available,
    }
    variants = [("reference", False, False)]
    if args.muon_reuse_ns_buffers:
        variants.append(("reuse_ns_buffers", True, False))
    if args.muon_fused_ns:
        variants.append(("fused_ns", False, True))
    # Symmetric order reduces sensitivity to the GPU clock drifting during a run.
    run_order = variants + list(reversed(variants)) if len(variants) > 1 else variants
    result["runs_per_variant"] = 2 if len(variants) > 1 else 1
    result["run_order"] = [name for name, _, _ in run_order]
    measurements = {name: [] for name, _, _ in variants}
    peak_allocated = {name: 0.0 for name, _, _ in variants}
    peak_reserved = {name: 0.0 for name, _, _ in variants}
    parity = {}
    expected = None
    for name, reuse, fused in run_order:
        torch.manual_seed(7)
        model = AutoRegressiveLM.from_pretrained(args.model_path).to(
            device="cuda", dtype=torch.bfloat16
        )
        kwargs = {}
        if reuse:
            kwargs["reuse_ns_buffers"] = True
        if fused:
            kwargs["fused_ns"] = True
        optimizer = MuonAdamW(model, lr=1e-5, ns_steps=5, **kwargs)
        torch.manual_seed(17)
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        for _ in range(3):
            optimizer.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            measurements[name].append((time.perf_counter() - start) * 1000)
        peak_allocated[name] = max(
            peak_allocated[name], torch.cuda.max_memory_allocated() / 2**20
        )
        peak_reserved[name] = max(
            peak_reserved[name], torch.cuda.max_memory_reserved() / 2**20
        )
        if name == "reference" and expected is None and len(variants) > 1:
            expected = {
                key: param.detach().cpu().clone()
                for key, param in model.named_parameters()
            }
        elif expected is not None:
            max_abs = 0.0
            different_elements = 0
            for key, param in model.named_parameters():
                current = param.detach().cpu()
                if not torch.equal(current, expected[key]):
                    difference = current.float() - expected[key].float()
                    max_abs = max(max_abs, difference.abs().max().item())
                    different_elements += torch.count_nonzero(difference).item()
            previous = parity.get(name, {"max_abs": 0.0, "different_elements": 0})
            parity[name] = {
                "max_abs": max(max_abs, previous["max_abs"]),
                "different_elements": max(
                    different_elements, previous["different_elements"]
                ),
            }
        result["parameter_count"] = sum(p.numel() for p in model.parameters())
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    for name, _, _ in variants:
        result[name] = {
            "optimizer_step": _summary(measurements[name]),
            "sample_count": len(measurements[name]),
            "peak_allocated_mib": round(peak_allocated[name], 2),
            "peak_reserved_mib": round(peak_reserved[name], 2),
        }
        if name in parity:
            result[name]["parameter_parity"] = {
                **parity[name],
                "passed": parity[name]["different_elements"] == 0,
            }
    return result


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
        "--muon-fused-ns",
        action="store_true",
        help="muon mode: compare the fused CUDA NS operator",
    )
    parser.add_argument(
        "--config", type=Path, help="optional JSON case matrix for train mode"
    )
    parser.add_argument(
        "--gpu", type=int, required=True, help="physical nvidia-smi GPU index"
    )
    parser.add_argument("--memory-fraction", type=float, default=0.15)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
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
                    muon_fused_ns=args.muon_fused_ns,
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
    import torch

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
