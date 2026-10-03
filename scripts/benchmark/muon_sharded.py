"""Compare full-gather Muon with the experimental sharded CUDA path."""

import argparse
import gc
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.optim._muon import _zeropower_via_newtonschulz

from astrai.extension.kernel.muon_sharded import is_available, steps_


def _summary(samples):
    values = sorted(samples)
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(values[math.ceil(0.95 * len(values)) - 1], 3),
    }


def _worker(rank, world_size, init_file, args):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        mesh = DeviceMesh("cuda", list(range(world_size)))
        coefficients = (3.4445, -4.775, 2.0315)
        run_order = ["reference", "sharded", "sharded", "reference"]
        samples = {name: [] for name in ("reference", "sharded")}
        peaks = {name: 0.0 for name in samples}
        parity = {"max_abs": 0.0, "different_elements": 0}
        expected = None
        for name in run_order:
            torch.manual_seed(7)
            matrices = []
            for _ in range(args.matrices):
                full_param = torch.randn(
                    (args.rows, args.cols), device=f"cuda:{rank}", dtype=torch.bfloat16
                )
                full_grad = torch.randn_like(full_param)
                full_buffer = torch.zeros_like(full_param)
                matrices.append(
                    (
                        distribute_tensor(full_param, mesh, [Shard(0)]),
                        distribute_tensor(full_grad, mesh, [Shard(0)]),
                        distribute_tensor(full_buffer, mesh, [Shard(0)]),
                    )
                )

            def optimizer_step(name=name, matrices=matrices):
                if name == "sharded":
                    steps_(
                        [
                            (param, grad, buffer, 1e-3)
                            for param, grad, buffer in matrices
                        ],
                        lr=1e-3,
                        weight_decay=0.1,
                        momentum=0.95,
                        nesterov=True,
                        ns_coefficients=coefficients,
                        ns_steps=5,
                        eps=1e-7,
                    )
                else:
                    for param, grad, buffer in matrices:
                        buffer.lerp_(grad, 0.05)
                        update = grad.lerp(buffer, 0.95)
                        full = update.full_tensor()
                        ortho = _zeropower_via_newtonschulz(full, coefficients, 5, 1e-7)
                        param.mul_(1 - 1e-3 * 0.1)
                        param.add_(
                            distribute_tensor(ortho, mesh, [Shard(0)]), alpha=-1e-3
                        )

            for _ in range(args.warmup):
                optimizer_step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            local_times = []
            for _ in range(args.steps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                optimizer_step()
                torch.cuda.synchronize()
                local_times.append((time.perf_counter() - start) * 1000)
            rank_times = torch.tensor(local_times, device=f"cuda:{rank}")
            dist.all_reduce(rank_times, op=dist.ReduceOp.MAX)
            if rank == 0:
                samples[name].extend(rank_times.cpu().tolist())
            peak = torch.tensor(
                torch.cuda.max_memory_allocated() / 2**20, device=f"cuda:{rank}"
            )
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                peaks[name] = max(peaks[name], peak.item())
            current = [param.to_local().detach().clone() for param, _, _ in matrices]
            if name == "reference" and expected is None:
                expected = current
            elif expected is not None:
                differences = [
                    (value.float() - target.float()).abs()
                    for value, target in zip(current, expected)
                ]
                maximum = torch.stack([value.max() for value in differences]).max()
                count = sum(torch.count_nonzero(value) for value in differences)
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
                if rank == 0:
                    parity["max_abs"] = max(parity["max_abs"], maximum.item())
                    parity["different_elements"] = max(
                        parity["different_elements"], count.item()
                    )
            del optimizer_step, full_param, full_grad, full_buffer
            matrices.clear()
            gc.collect()
            torch.cuda.empty_cache()
        if rank == 0:
            output = {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "shape": [args.rows, args.cols],
                "matrices": args.matrices,
                "world_size": world_size,
                "dtype": "bfloat16",
                "warmup": args.warmup,
                "measured_steps": args.steps,
                "run_order": run_order,
                "reference": {
                    "time": _summary(samples["reference"]),
                    "peak_allocated_mib": round(peaks["reference"], 2),
                },
                "sharded": {
                    "time": _summary(samples["sharded"]),
                    "peak_allocated_mib": round(peaks["sharded"], 2),
                },
                "parameter_parity": parity,
            }
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(output, indent=2) + "\n")
            print(json.dumps(output, indent=2))
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--matrices", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if not is_available():
        parser.error("muon_sharded extension is unavailable")
    if not 2 <= args.world_size <= torch.cuda.device_count():
        parser.error("world-size must be between 2 and the CUDA device count")
    if args.rows < 4 * args.cols or args.cols < 1 or args.rows < args.world_size:
        parser.error("requires a nonempty dim-0 shard with rows >= 4 * cols")
    if args.matrices < 1:
        parser.error("matrices must be positive")
    if args.warmup < 1 or args.steps < 1:
        parser.error("warmup and steps must be positive")
    with tempfile.TemporaryDirectory() as temporary:
        mp.spawn(
            _worker,
            args=(args.world_size, str(Path(temporary) / "init"), args),
            nprocs=args.world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
