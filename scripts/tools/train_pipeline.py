"""Run dense SEQ pretraining with one pipeline stage per rank.

Example (24 layers, eight GPUs, one million input tokens per update):

    torchrun --standalone --nproc-per-node=8 -m scripts.tools.train_pipeline \
        --model MODEL --data DATA --steps 5 --pp-size 2 --split-at 12 \
        --checkpoint-counts 6 0 --global-batch 64 --seq-len 16384

The reader drops incomplete global batches. --output exports model weights;
optimizer state and the data cursor are local to the run.
"""

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader

from astrai.config import ComponentSpec
from astrai.dataset import DatasetFactory, GlobalBatchSampler
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel.pipeline import DecoderStage, PipelineUpdate
from astrai.trainer.components import build_optimizer_component


def train(args):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    pp = args.pp_size
    if pp < 2 or world % pp or min(args.steps, args.microbatch, args.seq_len) < 1:
        raise ValueError(
            "PP must divide world size and batch dimensions must be positive"
        )
    dp = world // pp
    if args.global_batch % (dp * args.microbatch):
        raise ValueError("global batch must be divisible by DP times microbatch size")
    count = args.global_batch // (dp * args.microbatch)
    if count < pp:
        raise ValueError("1F1B requires at least one microbatch per pipeline stage")
    mesh = init_device_mesh("cuda", (dp, pp), mesh_dim_names=("dp", "pp"))
    rank = mesh["pp"].get_local_rank()
    replica = mesh["dp"].get_local_rank()
    torch.manual_seed(args.seed + replica)
    lm = AutoRegressiveLM.from_pretrained(args.model)
    depth = len(lm.model.layers)
    cuts = (
        args.split_at
        if args.split_at is not None
        else [i * depth // pp for i in range(1, pp)]
    )
    if len(cuts) != pp - 1:
        raise ValueError("--split-at needs pp_size - 1 layer indices")
    recompute = (
        args.checkpoint_counts if args.checkpoint_counts is not None else [0] * pp
    )
    if len(recompute) != pp:
        raise ValueError("--checkpoint-counts needs a layer count per stage")
    module = DecoderStage(lm, [0, *cuts, depth], rank, recompute[rank])
    del lm
    module.to(device=device, dtype=torch.bfloat16)
    update = PipelineUpdate(module, mesh, args.microbatch, args.seq_len, count)
    opt = build_optimizer_component(
        ComponentSpec(
            type="muon_adamw",
            params=dict(lr=args.lr, use_ns_kernels=True, shard_state=True),
        ),
        module,
        process_group=mesh["dp"].get_group(),
    )
    data = DatasetFactory.load("seq", str(args.data), window_size=args.seq_len)
    sampler = GlobalBatchSampler(
        data, dp, replica, args.microbatch, count, args.seed, "drop"
    )
    if len(sampler) < count:
        raise ValueError("dataset must contain a complete global optimizer batch")
    loader = DataLoader(data, batch_sampler=sampler)
    step = 0
    while step < args.steps:
        batches = iter(loader)
        for _ in sampler.windows():
            dist.barrier()
            start = time.perf_counter()
            batch = list(itertools.islice(batches, count))
            inputs = torch.cat([b["input_ids"] for b in batch]).to(device)
            targets = torch.cat([b["target_ids"] for b in batch]).to(device)
            opt.zero_grad()
            loss = update.backward(inputs, targets)
            norm = update.clip_grad_norm(1.0)
            opt.step()
            torch.cuda.synchronize(device)
            stats = torch.tensor(
                [
                    time.perf_counter() - start,
                    torch.cuda.max_memory_allocated(device) / 2**30,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            step += 1
            if dist.get_rank() == 0:
                seconds, gib = stats.tolist()
                print(
                    json.dumps(
                        dict(
                            step=step,
                            loss=loss.item(),
                            grad_norm=norm.item(),
                            seconds=seconds,
                            input_tokens_s=args.global_batch * args.seq_len / seconds,
                            peak_gib=gib,
                        )
                    ),
                    flush=True,
                )
            if step == args.steps:
                break
        sampler.epoch += 1
    if args.output is not None:
        # Mark the export complete by publishing its index after all shards exist.
        if dist.get_rank() == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        if replica == 0:
            shard = f"model-{rank + 1:05d}-of-{pp:05d}.safetensors"
            state = {k: v.cpu().contiguous() for k, v in module.state_dict().items()}
            save_file(state, str(args.output / shard))
            manifests = [None] * pp
            dist.all_gather_object(
                manifests, dict.fromkeys(state, shard), group=mesh["pp"].get_group()
            )
            if rank == 0:
                (args.output / "config.json").write_text(
                    json.dumps(module.model.config.to_dict())
                )
                mapping = {k: v for part in manifests for k, v in part.items()}
                (args.output / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": mapping})
                )
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="AstrAI pretrained model directory"
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="SEQ dataset directory"
    )
    parser.add_argument(
        "--output", type=Path, help="export final model weights to a new directory"
    )
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument(
        "--split-at",
        type=int,
        nargs="+",
        help="zero-based first layer of each subsequent stage",
    )
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--checkpoint-counts",
        type=int,
        nargs="+",
        help="number of leading blocks to activation-checkpoint in each stage",
    )
    train(parser.parse_args())
