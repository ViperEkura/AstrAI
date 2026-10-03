"""Benchmark DDP-replicated full-matrix Muon ownership with torchrun."""

import argparse
import contextlib
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from astrai.extension.kernel.cross_entropy import cross_entropy
from astrai.model import AutoRegressiveLM
from astrai.optim.muon_adamw import MuonAdamW
from scripts.benchmark.train_mfu import PhaseTimer, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/AstrAI-V1-base")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=0,
        help="Enable complete DDP training when positive; zero benchmarks optimizer only",
    )
    parser.add_argument(
        "--effective-batch", type=int, default=8, help="Global batch across all ranks"
    )
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument(
        "--json-out", type=Path, default=Path("results/muon-replicated.json")
    )
    args = parser.parse_args()
    if min(args.warmup, args.steps) < 1:
        parser.error("warmup and steps must be positive")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    if (
        args.seq_len < 0
        or args.micro_batch < 1
        or args.effective_batch < 1
        or (args.seq_len and args.effective_batch % (world * args.micro_batch))
    ):
        parser.error(
            "global batch must be divisible by world size times micro-batch; seq-len must be nonnegative"
        )
    accumulation = args.effective_batch // (world * args.micro_batch)
    baseline = None
    records = []

    def run(variant):
        torch.manual_seed(7)
        model = AutoRegressiveLM.from_pretrained(args.model_path).to(
            device="cuda", dtype=torch.bfloat16
        )
        optimizer = MuonAdamW(
            model,
            lr=1e-5,
            fused_ns=True,
            muon_process_group=dist.group.WORLD if variant == "owner" else None,
        )
        timer = None
        hooks = []
        for name, sub in (("muon", optimizer.muon), ("adamw", optimizer.adamw)):
            hooks.append(
                sub.register_step_pre_hook(
                    lambda *_, name=name: (
                        timer.begin(name) if timer is not None else None
                    )
                )
            )
            hooks.append(
                sub.register_step_post_hook(
                    lambda *_, name=name: timer.end(name) if timer is not None else None
                )
            )
        if args.seq_len:
            run_model = DDP(
                model,
                device_ids=[torch.cuda.current_device()],
                gradient_as_bucket_view=True,
            )
            generator = torch.Generator().manual_seed(11 + rank)
            shape = (
                args.warmup + args.steps,
                accumulation,
                args.micro_batch,
                args.seq_len,
            )
            inputs = torch.randint(0, 4096, shape, generator=generator).cuda()
            targets = torch.randint(0, 4096, shape, generator=generator).cuda()
        else:
            torch.manual_seed(11)
            for param in model.parameters():
                param.grad = torch.randn_like(param)

        def step(index):
            loss_sum = torch.zeros((), device="cuda")
            if args.seq_len:
                optimizer.zero_grad(set_to_none=True)
                for micro in range(accumulation):
                    scope = (
                        contextlib.nullcontext()
                        if micro == accumulation - 1
                        else run_model.no_sync()
                    )
                    with scope:
                        if timer is not None:
                            timer.begin("forward")
                        logits = run_model(inputs[index, micro])["logits"]
                        loss = (
                            cross_entropy(
                                logits.flatten(0, 1), targets[index, micro].flatten()
                            )
                            / accumulation
                        )
                        if timer is not None:
                            timer.end("forward")
                            timer.begin("backward")
                        loss.backward()
                        if timer is not None:
                            timer.end("backward")
                        loss_sum += loss.detach()
            optimizer.step()
            return loss_sum

        for index in range(args.warmup):
            step(index)
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples, losses, phase_samples = [], [], {}
        for index in range(args.warmup, args.warmup + args.steps):
            timer = PhaseTimer()
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss = step(index)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
            losses.append(loss.item())
            for name, elapsed in timer.values().items():
                phase_samples.setdefault(name, []).append(elapsed)
        timer = None
        times = torch.tensor(samples, device="cuda")
        dist.all_reduce(times, op=dist.ReduceOp.MAX)
        result = {
            "variant": variant,
            "raw_ms": times.tolist(),
            "median_ms": statistics.median(times.tolist()),
            "full_step": summarize(times.tolist()),
            "phases_rank0": {
                name: summarize(values) for name, values in phase_samples.items()
            },
            "losses_rank0": losses,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        }
        if args.seq_len:
            result["tokens_per_second"] = (
                args.effective_batch * args.seq_len * 1000 / result["median_ms"]
            )
        for hook in hooks:
            hook.remove()
        params = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
        }
        return result, params

    try:
        for variant in ("local", "owner", "owner", "local"):
            record, params = run(variant)
            if baseline is None:
                baseline = params
            else:
                max_abs, different = 0.0, 0
                for name, value in params.items():
                    difference = value.float() - baseline[name].float()
                    max_abs = max(max_abs, difference.abs().max().item())
                    different += torch.count_nonzero(difference).item()
                parity = torch.tensor(
                    [max_abs, different], device="cuda", dtype=torch.float64
                )
                dist.all_reduce(parity, op=dist.ReduceOp.MAX)
                record["parity"] = {
                    "max_abs": parity[0].item(),
                    "different_elements": int(parity[1].item()),
                }
            records.append(record)
            if rank == 0:
                print(json.dumps(record), flush=True)
            del params
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
        if rank == 0:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(
                    {
                        "world_size": world,
                        "gpu": torch.cuda.get_device_name(),
                        "model": args.model_path,
                        "warmup": args.warmup,
                        "seq_len": args.seq_len,
                        "effective_batch": args.effective_batch,
                        "micro_batch": args.micro_batch,
                        "steps": args.steps,
                        "records": records,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
