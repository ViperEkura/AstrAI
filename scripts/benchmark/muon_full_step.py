"""Measure full AstrAI 1B Muon training steps with matched data and ABBA order."""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from astrai.model import AutoRegressiveLM
from astrai.optim.muon_adamw import MuonAdamW

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-path", default="models/AstrAI-V1-base")
parser.add_argument("--seq-len", type=int, default=256)
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--memory-fraction", type=float, default=0.5)
parser.add_argument(
    "--json-out", type=Path, default=Path("results/muon-full-step.json")
)
args = parser.parse_args()
if min(args.seq_len, args.warmup, args.steps) < 1 or not 0 < args.memory_fraction <= 1:
    parser.error(
        "seq-len, warmup and steps must be positive; memory-fraction must be in (0, 1]"
    )

WARMUP = args.warmup
MEASURED = args.steps
SEQ = args.seq_len


def summarize(samples):
    samples = sorted(samples)
    return {
        "median_ms": round(statistics.median(samples), 3),
        "mean_ms": round(statistics.mean(samples), 3),
        "p95_ms": round(samples[-1], 3),
    }


def step(model, optimizer, inputs, targets):
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)
    loss = F.cross_entropy(output["logits"].float().flatten(0, 1), targets.flatten())
    loss.backward()
    optimizer.step()
    return loss


torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
torch.manual_seed(11)
inputs = torch.randint(0, 4096, (1, SEQ), device="cuda")
targets = torch.randint(0, 4096, (1, SEQ), device="cuda")
timings = {"reference": [], "fused_ns": []}
phases = {
    "reference": {"forward": [], "backward": [], "optimizer": []},
    "fused_ns": {"forward": [], "backward": [], "optimizer": []},
}
baseline = None
parity = {"max_abs": 0.0, "different_elements": 0}
blocks = []
peaks = {"reference": 0.0, "fused_ns": 0.0}

for variant in ("reference", "fused_ns", "fused_ns", "reference"):
    torch.manual_seed(7)
    model = (
        AutoRegressiveLM.from_pretrained(args.model_path)
        .to(device="cuda", dtype=torch.bfloat16)
        .train()
    )
    optimizer = MuonAdamW(model, lr=1e-5, ns_steps=5, fused_ns=variant == "fused_ns")
    for _ in range(WARMUP):
        step(model, optimizer, inputs, targets)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(MEASURED):
        start, forward, backward, finish = (
            torch.cuda.Event(enable_timing=True) for _ in range(4)
        )
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        start.record()
        output = model(inputs)
        loss = F.cross_entropy(
            output["logits"].float().flatten(0, 1), targets.flatten()
        )
        forward.record()
        loss.backward()
        backward.record()
        optimizer.step()
        finish.record()
        torch.cuda.synchronize()
        timings[variant].append((time.perf_counter() - wall_start) * 1000)
        phases[variant]["forward"].append(start.elapsed_time(forward))
        phases[variant]["backward"].append(forward.elapsed_time(backward))
        phases[variant]["optimizer"].append(backward.elapsed_time(finish))
    current = timings[variant][-MEASURED:]
    blocks.append(
        {"variant": variant, "full_step": summarize(current), "raw_ms": current}
    )
    peaks[variant] = max(peaks[variant], torch.cuda.max_memory_allocated() / 2**20)
    print(variant, summarize(current), flush=True)
    if baseline is None:
        baseline = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
        }
    elif variant == "fused_ns":
        run_different = 0
        for name, param in model.named_parameters():
            current = param.detach().cpu()
            difference = current.float() - baseline[name].float()
            parity["max_abs"] = max(parity["max_abs"], difference.abs().max().item())
            run_different += torch.count_nonzero(difference).item()
        parity["different_elements"] = max(parity["different_elements"], run_different)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

result = {
    "gpu": torch.cuda.get_device_name(),
    "model": args.model_path,
    "dtype": "bfloat16",
    "batch": 1,
    "seq": SEQ,
    "memory_fraction": args.memory_fraction,
    "model_seed": 7,
    "data_seed": 11,
    "warmup": WARMUP,
    "measured_per_run": MEASURED,
    "parity_after_steps": WARMUP + MEASURED,
    "run_order": ["reference", "fused_ns", "fused_ns", "reference"],
    "blocks": blocks,
    "reference": {
        "full_step": summarize(timings["reference"]),
        "phases": {key: summarize(value) for key, value in phases["reference"].items()},
        "peak_allocated_mib": round(peaks["reference"], 2),
    },
    "fused_ns": {
        "full_step": summarize(timings["fused_ns"]),
        "phases": {key: summarize(value) for key, value in phases["fused_ns"].items()},
        "peak_allocated_mib": round(peaks["fused_ns"], 2),
    },
    "parity": parity,
}
args.json_out.parent.mkdir(parents=True, exist_ok=True)
with args.json_out.open("w", encoding="utf-8") as file:
    json.dump(result, file, indent=2)
print(json.dumps(result, indent=2), flush=True)
