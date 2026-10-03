"""Compare training configurations at fixed tokens per optimizer update."""

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from astrai.extension.kernel.cross_entropy import cross_entropy
from astrai.model import AutoRegressiveLM
from astrai.model.components.linear import Linear
from astrai.optim import muon_adamw as muon_impl
from astrai.optim.muon_adamw import MuonAdamW


def summarize(values: List[float]) -> Dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


class PhaseTimer:
    def __init__(self):
        self.pairs = {}
        self.active = {}

    def begin(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.active[name] = event

    def end(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.pairs.setdefault(name, []).append((self.active.pop(name), event))

    def values(self):
        return {
            name: sum(start.elapsed_time(end) for start, end in pairs)
            for name, pairs in self.pairs.items()
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/AstrAI-V1-base")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--effective-batch", type=int, default=8)
    parser.add_argument("--micro-batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--compile-modes", nargs="+", default=["eager"])
    parser.add_argument(
        "--loss-modes", choices=["torch", "fused"], nargs="+", default=["torch"]
    )
    parser.add_argument(
        "--variants", nargs="+", help="Override sweep with MICRO:COMPILE:LOSS entries"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--profile-ns", action="store_true")
    parser.add_argument("--peak-bf16-tflops", type=float)
    parser.add_argument("--json-out", type=Path, default=Path("results/train-mfu.json"))
    args = parser.parse_args()
    if (
        min(
            args.seq_len,
            args.effective_batch,
            args.warmup,
            args.steps,
            *args.micro_batches,
        )
        < 1
    ):
        parser.error("shape, warmup, and steps must be positive")
    if not args.variants and any(
        args.effective_batch % micro for micro in args.micro_batches
    ):
        parser.error("each micro-batch must divide effective-batch")
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory-fraction must be in (0, 1]")
    if args.peak_bf16_tflops is not None and args.peak_bf16_tflops <= 0:
        parser.error("peak-bf16-tflops must be positive")
    allowed = {"eager", "default", "reduce-overhead", "max-autotune-no-cudagraphs"}
    if set(args.compile_modes) - allowed:
        parser.error(f"compile-modes must be drawn from {sorted(allowed)}")

    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    generator = torch.Generator().manual_seed(11)
    shape = (args.warmup + args.steps + 1, args.effective_batch, args.seq_len)
    inputs = torch.randint(0, 4096, shape, generator=generator).cuda()
    targets = torch.randint(0, 4096, shape, generator=generator).cuda()
    variants = [
        (micro, mode, loss_mode)
        for loss_mode in args.loss_modes
        for mode in args.compile_modes
        for micro in args.micro_batches
    ]
    if args.variants:
        try:
            variants = [
                (int(micro), mode, loss)
                for micro, mode, loss in (value.split(":") for value in args.variants)
            ]
        except ValueError:
            parser.error("variants must be MICRO:COMPILE:LOSS")
        if any(
            micro < 1
            or args.effective_batch % micro
            or mode not in allowed
            or loss not in {"torch", "fused"}
            for micro, mode, loss in variants
        ):
            parser.error("invalid micro-batch, compile mode, or loss mode in variants")
    run_order = variants + list(reversed(variants))
    records = []
    baseline_params = None
    baseline_losses = None
    original_ns = muon_impl.muon_ns
    tokens_per_update = args.effective_batch * args.seq_len

    def save():
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "model": args.model_path,
                    "dtype": "bfloat16",
                    "seq_len": args.seq_len,
                    "effective_batch": args.effective_batch,
                    "tokens_per_update": tokens_per_update,
                    "warmup": args.warmup,
                    "steps": args.steps,
                    "model_seed": 7,
                    "data_seed": 11,
                    "run_order": run_order,
                    "peak_bf16_tflops": args.peak_bf16_tflops,
                    "mfu_note": "Estimated dense-model FLOPs only; excludes NS and optimizer FLOPs. MFU is null unless a hardware peak is supplied.",
                    "records": records,
                },
                indent=2,
            )
            + "\n"
        )

    for micro, mode, loss_mode in run_order:
        print(
            f"START micro={micro} accum={args.effective_batch // micro} compile={mode} loss={loss_mode}",
            flush=True,
        )
        torch.manual_seed(7)
        model = (
            AutoRegressiveLM.from_pretrained(args.model_path)
            .to(device="cuda", dtype=torch.bfloat16)
            .train()
        )
        optimizer = MuonAdamW(model, lr=1e-5, ns_steps=5, fused_ns=True)
        # This proxy counts all dense linear weights, including lm_head, and
        # causal attention's quadratic matmuls. Embedding lookups are excluded.
        linear_params = sum(
            module.weight.numel()
            for module in model.modules()
            if isinstance(module, (torch.nn.Linear, Linear))
        )
        config = model.config
        flop_per_token = (
            6 * linear_params
            + 6 * config.num_hidden_layers * args.seq_len * config.hidden_size
        )
        run_model = model if mode == "eager" else torch.compile(model, mode=mode)
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
        del sub

        def step(index):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for start in range(0, args.effective_batch, micro):
                if timer is not None:
                    timer.begin("forward")
                output = run_model(inputs[index, start : start + micro])
                logits = output["logits"].flatten(0, 1)
                labels = targets[index, start : start + micro].flatten()
                loss = (
                    cross_entropy(logits, labels)
                    if loss_mode == "fused"
                    else F.cross_entropy(logits.float(), labels)
                )
                loss = loss * (micro / args.effective_batch)
                if timer is not None:
                    timer.end("forward")
                    timer.begin("backward")
                loss.backward()
                if timer is not None:
                    timer.end("backward")
                losses.append(loss.detach())
                del output, loss, logits, labels
            optimizer.step()
            return torch.stack(losses).sum()

        record = {
            "micro_batch": micro,
            "grad_accum_steps": args.effective_batch // micro,
            "compile_mode": mode,
            "loss_mode": loss_mode,
        }
        try:
            for index in range(args.warmup):
                step(index)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            wall_samples, loss_samples = [], []
            phase_samples = {
                key: [] for key in ("forward", "backward", "muon", "adamw")
            }
            for index in range(args.warmup, args.warmup + args.steps):
                timer = PhaseTimer()
                torch.cuda.synchronize()
                started = time.perf_counter()
                loss = step(index)
                torch.cuda.synchronize()
                wall_samples.append((time.perf_counter() - started) * 1000)
                loss_samples.append(loss.item())
                for name, elapsed in timer.values().items():
                    phase_samples[name].append(elapsed)
            timer = None
            full_step = summarize(wall_samples)
            tokens_per_second = tokens_per_update * 1000 / full_step["median_ms"]
            record.update(
                {
                    "status": "ok",
                    "full_step": full_step,
                    "phases": {
                        key: summarize(value) for key, value in phase_samples.items()
                    },
                    "tokens_per_second": tokens_per_second,
                    "estimated_model_tflops": tokens_per_second * flop_per_token / 1e12,
                    "estimated_mfu": tokens_per_second
                    * flop_per_token
                    / 1e12
                    / args.peak_bf16_tflops
                    if args.peak_bf16_tflops
                    else None,
                    "linear_parameters": linear_params,
                    "estimated_flops_per_token": flop_per_token,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "raw_ms": wall_samples,
                    "losses": loss_samples,
                }
            )
            if baseline_params is None:
                baseline_params = {
                    name: param.detach().cpu().clone()
                    for name, param in model.named_parameters()
                }
                baseline_losses = loss_samples
            max_abs, different, total = 0.0, 0, 0
            for name, param in model.named_parameters():
                actual = param.detach().cpu()
                difference = actual.float() - baseline_params[name].float()
                max_abs = max(max_abs, difference.abs().max().item())
                different += torch.count_nonzero(difference).item()
                total += param.numel()
            del param
            record["parity"] = {
                "max_abs": max_abs,
                "different_elements": different,
                "total_elements": total,
                "max_loss_abs": max(
                    abs(a - b) for a, b in zip(loss_samples, baseline_losses)
                ),
            }
            if args.profile_ns:
                ns_timer = PhaseTimer()

                def profile_ns(grad, *ns_args):
                    name = f"{grad.size(0)}x{grad.size(1)}"
                    ns_timer.begin(name)
                    result = original_ns(grad, *ns_args)
                    ns_timer.end(name)
                    return result

                muon_impl.muon_ns = profile_ns
                step(args.warmup + args.steps)
                torch.cuda.synchronize()
                muon_impl.muon_ns = original_ns
                record["separate_ns_profile_ms"] = ns_timer.values()
                record["separate_ns_profile_total_ms"] = sum(ns_timer.values().values())
        except torch.OutOfMemoryError as error:
            record.update({"status": "oom", "error": str(error)})
        finally:
            muon_impl.muon_ns = original_ns
            timer = None
            for hook in hooks:
                hook.remove()
            del step, run_model, model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
        records.append(record)
        save()
        print(
            json.dumps(
                {
                    key: record[key]
                    for key in (
                        "micro_batch",
                        "compile_mode",
                        "loss_mode",
                        "status",
                        "full_step",
                        "tokens_per_second",
                        "peak_allocated_mib",
                        "parity",
                    )
                    if key in record
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
