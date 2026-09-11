"""Run a bounded DDP parity check for rank-local training telemetry.

Example:
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python \
        scripts/eval/training_telemetry_ddp_e2e.py \
        --world-size 4 --output-dir /tmp/astrai-telemetry-e2e
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import Dataset

from astrai.config import AutoRegressiveLMConfig, TrainConfig
from astrai.model.transformer import AutoRegressiveLM
from astrai.parallel.setup import find_free_port, get_rank
from astrai.serialization import Checkpoint
from astrai.trainer import Trainer

_MODEL_SEED = 20260904
_DATASET_SEED = 1907
_TRACE_DIR_ENV = "ASTRAI_TELEMETRY_E2E_TRACE_DIR"


class DeterministicTokenDataset(Dataset):
    def __init__(self, length: int, sequence_length: int, vocab_size: int) -> None:
        self.length = length
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(_DATASET_SEED + index)
        input_ids = torch.randint(
            1,
            self.vocab_size,
            (self.sequence_length,),
            generator=generator,
        )
        target_ids = torch.roll(input_ids, shifts=-1)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": torch.ones(self.sequence_length, dtype=torch.bool),
        }


def _configure_trace_logging() -> None:
    trace_dir = os.environ.get(_TRACE_DIR_ENV)
    if not trace_dir:
        return
    # LocalStrategy passes rank directly to init_process_group instead of
    # exporting RANK. Query the initialized process group so each spawned
    # worker writes its own evidence file under both local and torchrun launch.
    rank = get_rank()
    path = Path(trace_dir) / f"rank-{rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_logger = logging.getLogger("astrai.trainer.training_telemetry")
    if any(
        getattr(handler, "baseFilename", None) == str(path)
        for handler in telemetry_logger.handlers
    ):
        return
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    telemetry_logger.addHandler(handler)
    telemetry_logger.setLevel(logging.INFO)


def build_model() -> nn.Module:
    _configure_trace_logging()
    torch.manual_seed(_MODEL_SEED)
    config = AutoRegressiveLMConfig(
        vocab_size=512,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
        num_hidden_layers=2,
        rms_norm_eps=1e-5,
    )
    return AutoRegressiveLM(config)


def build_optimizer(model: nn.Module) -> Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)


def build_scheduler(optimizer: Optimizer) -> LRScheduler:
    return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


def _run_training(
    run_dir: Path,
    *,
    telemetry_enabled: bool,
    world_size: int,
    steps: int,
    batch_per_device: int,
    sequence_length: int,
    device_type: str,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = run_dir / "traces"
    os.environ[_TRACE_DIR_ENV] = str(trace_dir)
    dataset = DeterministicTokenDataset(
        length=world_size * steps * batch_per_device,
        sequence_length=sequence_length,
        vocab_size=512,
    )
    config = TrainConfig(
        model_fn=build_model,
        strategy="seq",
        dataset=dataset,
        optimizer_fn=build_optimizer,
        scheduler_fn=build_scheduler,
        n_epoch=1,
        batch_per_device=batch_per_device,
        grad_accum_steps=1,
        ckpt_dir=str(run_dir / "checkpoints"),
        ckpt_interval=steps,
        random_seed=_DATASET_SEED,
        nprocs=world_size,
        backend="nccl" if device_type == "cuda" else "gloo",
        parallel_mode="ddp" if world_size > 1 else "none",
        start_method="spawn",
        device_type=device_type,
        master_port=find_free_port(),
        training_telemetry_enabled=telemetry_enabled,
    )
    Trainer(config).train()
    return run_dir / "checkpoints" / f"epoch_0_step_{steps}"


def _state_digest(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _compare_states(
    baseline: dict[str, torch.Tensor], observed: dict[str, torch.Tensor]
) -> tuple[bool, float]:
    if baseline.keys() != observed.keys():
        return False, float("inf")
    exact = True
    max_abs_diff = 0.0
    for name, value in baseline.items():
        left = value.detach().cpu()
        right = observed[name].detach().cpu()
        exact = exact and torch.equal(left, right)
        max_abs_diff = max(
            max_abs_diff,
            float((left.float() - right.float()).abs().max().item()),
        )
    return exact, max_abs_diff


def _load_traces(trace_dir: Path, world_size: int) -> dict[int, list[dict[str, Any]]]:
    traces: dict[int, list[dict[str, Any]]] = {}
    for rank in range(world_size):
        path = trace_dir / f"rank-{rank}.jsonl"
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            prefix = "training_telemetry "
            if line.startswith(prefix):
                records.append(json.loads(line.removeprefix(prefix)))
        traces[rank] = records
    return traces


def _validate_traces(
    traces: dict[int, list[dict[str, Any]]],
    *,
    steps: int,
    batch_per_device: int,
    sequence_length: int,
    require_hbm: bool,
) -> None:
    expected_tokens = batch_per_device * sequence_length
    work_ids = set()
    for rank, records in traces.items():
        if len(records) != steps:
            raise AssertionError(
                f"rank {rank} emitted {len(records)} traces, expected {steps}"
            )
        for record in records:
            work_item = record["work_item"]
            work_ids.add(work_item["work_id"])
            assert record["status"] == "ok"
            if require_hbm:
                assert record["peak_hbm_bytes"] > 0
            else:
                assert record["peak_hbm_bytes"] == 0
            assert record["inflight_tokens_at_end"] == 0
            assert work_item["rank"] == rank
            assert work_item["cost"]["tokens"] == expected_tokens
            assert work_item["estimation_error"] is None
    if len(work_ids) != len(traces) * steps:
        raise AssertionError("training work IDs are not globally unique")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-per-device", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--device-type", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.world_size < 1:
        raise ValueError("world-size must be positive")
    if args.device_type == "cuda" and torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"requested {args.world_size} GPUs, only {torch.cuda.device_count()} visible"
        )
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    baseline_dir = args.output_dir / "baseline"
    telemetry_dir = args.output_dir / "telemetry"
    baseline_checkpoint = _run_training(
        baseline_dir,
        telemetry_enabled=False,
        world_size=args.world_size,
        steps=args.steps,
        batch_per_device=args.batch_per_device,
        sequence_length=args.sequence_length,
        device_type=args.device_type,
    )
    telemetry_checkpoint = _run_training(
        telemetry_dir,
        telemetry_enabled=True,
        world_size=args.world_size,
        steps=args.steps,
        batch_per_device=args.batch_per_device,
        sequence_length=args.sequence_length,
        device_type=args.device_type,
    )

    baseline_state = Checkpoint.load(
        str(baseline_checkpoint), verify_checksums=True
    ).state_dict
    telemetry_state = Checkpoint.load(
        str(telemetry_checkpoint), verify_checksums=True
    ).state_dict
    exact, max_abs_diff = _compare_states(baseline_state, telemetry_state)
    traces = _load_traces(telemetry_dir / "traces", args.world_size)
    _validate_traces(
        traces,
        steps=args.steps,
        batch_per_device=args.batch_per_device,
        sequence_length=args.sequence_length,
        require_hbm=args.device_type == "cuda",
    )
    if not exact:
        raise AssertionError(
            f"telemetry changed trained parameters (max_abs_diff={max_abs_diff})"
        )

    peak_hbm_by_rank = {
        str(rank): max(record["peak_hbm_bytes"] for record in records)
        for rank, records in traces.items()
    }
    summary = {
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_type": args.device_type,
        "world_size": args.world_size,
        "steps_per_rank": args.steps,
        "batch_per_device": args.batch_per_device,
        "sequence_length": args.sequence_length,
        "traces_per_rank": {
            str(rank): len(records) for rank, records in traces.items()
        },
        "peak_hbm_bytes_by_rank": peak_hbm_by_rank,
        "parameters_exactly_equal": exact,
        "max_parameter_abs_diff": max_abs_diff,
        "baseline_parameter_sha256": _state_digest(baseline_state),
        "telemetry_parameter_sha256": _state_digest(telemetry_state),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
