"""Paired CE memory/throughput benchmark against remote-main Torch semantics.

Example (lock and restore GPU clocks outside this process):
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark/cross_entropy.py \
    --mode train --batch-sizes 1 2 --warmup 20 --steps 100 --rounds 3

Synthetic full-vocabulary tokens, identical model/data/optimizer per variant.
Times include zero_grad, forward/loss, backward and optimizer; exclude data IO.
Chunked CE recomputes vocabulary tiles in backward; use the complete step
for the speed gate.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from astrai.extension.kernel.cross_entropy import (
    cross_entropy,
    is_available,
    linear_cross_entropy,
)
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.optim.muon_adamw import MuonAdamW
from astrai.trainer.strategy import SEQStrategy


def stats(values):
    ordered = sorted(values)
    return dict(
        mean=statistics.mean(values),
        median=statistics.median(values),
        p95=ordered[math.ceil(len(ordered) * 0.95) - 1],
    )


def parity(model, reference):
    maximum, different, total, sq_error, sq_ref = 0.0, 0, 0, 0.0, 0.0
    worst = ""
    for name, param in model.named_parameters():
        expected = reference[name].to(param.device)
        delta = param.detach().float() - expected.float()
        error = delta.abs().max().item()
        if error > maximum:
            maximum, worst = error, name
        different += (param.detach() != expected).sum().item()
        total += param.numel()
        sq_error += delta.square().sum().item()
        sq_ref += expected.float().square().sum().item()
    return dict(
        max_abs=maximum,
        different_elements=different,
        elements=total,
        relative_l2=math.sqrt(sq_error / max(sq_ref, 1e-30)),
        worst_parameter=worst,
    )


def capacity(args, config):
    """Find the largest completed micro-batch, including optimizer state."""
    records = []

    def attempt(batch_size, variant):
        torch.manual_seed(7)
        model = (
            AutoRegressiveLM.from_pretrained(args.model_path)
            .to(device="cuda", dtype=torch.bfloat16)
            .train()
        )
        optimizer = MuonAdamW(model, lr=1e-5, ns_steps=5)
        chunk = int(variant[6:]) if variant.startswith("linear") else 512
        backend = "cuda_linear_ce" if variant.startswith("linear") else variant
        strategy = SEQStrategy(
            model, "cuda", loss_backend=backend, loss_chunk_size=chunk
        )
        data = dict(
            input_ids=torch.randint(
                config["vocab_size"], (batch_size, args.seq_len), device="cuda"
            ),
            target_ids=torch.randint(
                config["vocab_size"], (batch_size, args.seq_len), device="cuda"
            ),
        )
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.warmup + args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = strategy.compute_loss(data)
            loss.backward()
            optimizer.step()
            del loss
        torch.cuda.synchronize()
        return dict(
            peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20,
        )

    for variant in args.variants:
        for batch_size in sorted(set(args.batch_sizes)):
            gc.collect()
            torch.cuda.empty_cache()
            record = dict(variant=variant, batch=batch_size)
            try:
                record.update(attempt(batch_size, variant))
                record["completed"] = True
            except torch.cuda.OutOfMemoryError as error:
                record.update(completed=False, error=str(error))
            records.append(record)
            print(json.dumps(record), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    dict(
                        mode="capacity",
                        gpu=torch.cuda.get_device_name(),
                        seq_len=args.seq_len,
                        dtype="bfloat16",
                        memory_fraction=args.memory_fraction,
                        steps=args.warmup + args.steps,
                        model=args.model_path,
                        records=records,
                    ),
                    indent=2,
                )
                + "\n"
            )
            if not record["completed"]:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["head", "train", "capacity"], default="train"
    )
    parser.add_argument("--model-path", default="models/AstrAI-V1-base")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--memory-fraction", type=float, default=0.95)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="separate numerical audit; changes the training kernels",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "torch",
            "cuda_ce",
            "linear128",
            "linear256",
            "linear512",
            "linear1024",
        ],
    )
    parser.add_argument("--out", type=Path, default=Path("results/ce-main.json"))
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.rounds, args.seq_len, *args.batch_sizes) < 1:
        parser.error("counts must be positive")
    if args.variants[0] != "torch" or set(args.variants) - {
        "torch",
        "cuda_ce",
        "linear128",
        "linear256",
        "linear512",
        "linear1024",
    }:
        parser.error("variants must start with torch and use documented backends")
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    if not is_available():
        raise RuntimeError(
            "build cross_entropy before benchmarking; fallback is not a measurement"
        )
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory-fraction must be in (0, 1]")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(7)
    config = json.loads((Path(args.model_path) / "config.json").read_text())
    if args.mode == "capacity":
        capacity(args, config)
        return
    records = []
    report = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        config=config,
        git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                "csrc/cross_entropy.cu",
                "astrai/extension/kernel/cross_entropy.py",
                "astrai/model/autoregressive_lm.py",
                "astrai/trainer/strategy.py",
                "scripts/benchmark/cross_entropy.py",
            )
        },
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        clocks=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,clocks.sm,clocks.mem,power.limit",
                "--format=csv",
            ],
            text=True,
        ),
        dtype="bfloat16",
        lr=1e-5,
        ns_steps=5,
        reuse_ns_buffers=False,
        model_seed=7,
        data_seed=11,
        records=records,
        timing="synchronized wall time; phases use CUDA events; synthetic data already on GPU",
    )

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    model = None
    initial = None
    if args.mode == "train":
        model = (
            AutoRegressiveLM.from_pretrained(args.model_path)
            .to(dtype=torch.bfloat16)
            .train()
        )
        initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        model.cuda()
    for batch_size in args.batch_sizes:
        generator = torch.Generator(device="cuda").manual_seed(11)
        shape = (args.warmup + args.steps, batch_size, args.seq_len)
        inputs = torch.randint(
            config["vocab_size"], shape, device="cuda", generator=generator
        )
        targets = torch.randint(
            config["vocab_size"], shape, device="cuda", generator=generator
        )
        reference = None
        for round_id in range(args.rounds):
            order = (
                args.variants if round_id % 2 == 0 else list(reversed(args.variants))
            )
            for variant in order:
                gc.collect()
                torch.cuda.empty_cache()
                torch.manual_seed(7)
                print(
                    f"START batch={batch_size} round={round_id} variant={variant}",
                    flush=True,
                )
                chunk = int(variant[6:]) if variant.startswith("linear") else 256
                if args.mode == "train":
                    model.load_state_dict(initial)
                    optimizer = MuonAdamW(model, lr=1e-5, ns_steps=5)
                    backend = (
                        "cuda_linear_ce" if variant.startswith("linear") else variant
                    )
                    strategy = SEQStrategy(
                        model, "cuda", loss_backend=backend, loss_chunk_size=chunk
                    )
                else:
                    x = torch.randn(
                        batch_size * args.seq_len,
                        config["hidden_size"],
                        device="cuda",
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    )
                    w = (
                        torch.randn(
                            config["vocab_size"],
                            config["hidden_size"],
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        * 0.02
                    ).requires_grad_()
                times = {
                    k: []
                    for k in ("step_ms", "forward_ms", "backward_ms", "optimizer_ms")
                }
                peak_allocated = peak_reserved = 0
                losses = []
                for index in range(args.warmup + args.steps):
                    if index == args.warmup:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if args.mode == "train":
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        x.grad = w.grad = None
                    events[0].record()
                    if args.mode == "train":
                        data = strategy.prepare_batch(
                            dict(input_ids=inputs[index], target_ids=targets[index])
                        )
                        result = strategy.forward_tokens(data)
                        token_loss = strategy.reduce_loss(result, data)
                        loss = token_loss.mean()
                        # Match the trainer lifetime: release ForwardResult before backward.
                        del result, token_loss
                    else:
                        y = targets[index].flatten()
                        if variant.startswith("linear"):
                            loss = (
                                linear_cross_entropy(x, w, y, chunk_size=chunk)
                                / y.numel()
                            )
                        elif variant == "cuda_ce":
                            loss = cross_entropy(F.linear(x, w), y) / y.numel()
                        else:
                            loss = F.cross_entropy(F.linear(x, w).float(), y)
                    events[1].record()
                    loss.backward()
                    events[2].record()
                    if args.mode == "train":
                        optimizer.step()
                    events[3].record()
                    torch.cuda.synchronize()
                    wall = (time.perf_counter() - start) * 1000
                    if index >= args.warmup:
                        times["step_ms"].append(wall)
                        for name, a, b in (
                            ("forward_ms", 0, 1),
                            ("backward_ms", 1, 2),
                            ("optimizer_ms", 2, 3),
                        ):
                            times[name].append(events[a].elapsed_time(events[b]))
                        losses.append(loss.item())
                    del loss
                peak_allocated = torch.cuda.max_memory_allocated() / 2**20
                peak_reserved = torch.cuda.max_memory_reserved() / 2**20
                record = dict(
                    batch=batch_size,
                    round=round_id,
                    variant=variant,
                    **{k: stats(v) for k, v in times.items()},
                    peak_allocated_MiB=peak_allocated,
                    peak_reserved_MiB=peak_reserved,
                    samples=times,
                    loss_first=losses[0],
                    loss_last=losses[-1],
                )
                record["tokens_per_second"] = (
                    batch_size * args.seq_len / (record["step_ms"]["median"] / 1000)
                )
                if args.mode == "train":
                    if reference is None:
                        reference = {
                            n: p.detach().cpu().clone()
                            for n, p in model.named_parameters()
                        }
                    record["parameter_parity"] = parity(model, reference)
                    del optimizer, strategy
                    model.zero_grad(set_to_none=True)
                else:
                    del x, w
                records.append(record)
                save()
                print(
                    json.dumps({k: v for k, v in record.items() if k != "samples"}),
                    flush=True,
                )
        del inputs, targets, reference
    for batch_size in args.batch_sizes:
        base = [
            r for r in records if r["batch"] == batch_size and r["variant"] == "torch"
        ]
        base_median = statistics.median(r["step_ms"]["median"] for r in base)
        base_p95 = statistics.median(r["step_ms"]["p95"] for r in base)
        base_mem = max(r["peak_allocated_MiB"] for r in base)
        for r in records:
            if r["batch"] == batch_size:
                r["median_ratio"] = r["step_ms"]["median"] / base_median
                r["p95_ratio"] = r["step_ms"]["p95"] / base_p95
                r["saved_MiB"] = base_mem - r["peak_allocated_MiB"]
                r["performance_gate"] = (
                    r["median_ratio"] <= 1.01
                    and r["p95_ratio"] <= 1.02
                    and r["saved_MiB"] > 0
                )
    save()


if __name__ == "__main__":
    main()
