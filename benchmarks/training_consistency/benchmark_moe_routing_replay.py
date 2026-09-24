"""Verify MoE dispatch replay across DDP online-training boundaries."""

import argparse
import hashlib
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from astrai.model.components.mlp import DeepSeekMoE
from astrai.trainer.train_callback import GradientCheckpointingCallback


def tensor_digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


class RouteRecorder:
    def __init__(self, moe: DeepSeekMoE):
        self.moe = moe
        self.records = []
        self.had_instance_override = "_select_experts" in moe.__dict__
        self.instance_override = moe.__dict__.get("_select_experts")
        self.original = moe._select_experts
        moe._select_experts = self._record

    def _record(self, router_logits: torch.Tensor):
        router_probs, topk_weights, topk_indices = self.original(router_logits)
        canonical_indices = torch.sort(topk_indices.detach(), dim=-1).values
        self.records.append(
            {
                "expert_indices_sha256": tensor_digest(canonical_indices),
                "router_logits_sha256": tensor_digest(router_logits.float()),
                "router_probs_sha256": tensor_digest(router_probs),
                "expert_token_counts": torch.bincount(
                    topk_indices.detach().reshape(-1),
                    minlength=self.moe.n_routed_experts,
                )
                .cpu()
                .tolist(),
            }
        )
        return router_probs, topk_weights, topk_indices

    def clear(self) -> None:
        self.records.clear()

    def close(self) -> None:
        if self.had_instance_override:
            self.moe._select_experts = self.instance_override
        else:
            del self.moe.__dict__["_select_experts"]


def assert_across_ranks(value: str, world_size: int, label: str) -> None:
    gathered = [None] * world_size
    dist.all_gather_object(gathered, value)
    if len(set(gathered)) != 1:
        raise AssertionError(f"{label} differs across ranks: {gathered}")


def make_moe(experts: int, top_k: int, device: torch.device) -> DeepSeekMoE:
    moe = DeepSeekMoE(
        dim=64,
        dim_ffn=128,
        n_routed_experts=experts,
        n_shared_experts=1,
        n_activated_experts=top_k,
    ).to(device=device, dtype=torch.bfloat16)
    moe.apply(
        lambda module: (
            module.reset_parameters() if hasattr(module, "reset_parameters") else None
        )
    )
    with torch.no_grad():
        moe.router.weight.mul_(0.001)
    return moe


def capture_forward(
    model: DeepSeekMoE,
    recorder: RouteRecorder,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    recorder.clear()
    output = model(hidden_states)
    if len(recorder.records) != 1:
        raise AssertionError(
            f"expected one routing decision, got {len(recorder.records)}"
        )
    return output["hidden_states"], recorder.records[0]


def run_case(
    *,
    experts: int,
    top_k: int,
    tokens: int,
    trials: int,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict:
    torch.manual_seed(3407)
    moe = make_moe(experts, top_k, device)
    recorder = RouteRecorder(moe)
    wrapped = DDP(
        moe,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=True,
    )
    latencies_ms = []
    mismatch_total = 0
    recompute_forward_total = 0
    expert_counts = None
    route_digest = None
    logits_digest = None
    probs_digest = None

    for trial in range(trials):
        generator = torch.Generator(device=device).manual_seed(3407 + trial)
        hidden_states = torch.randn(
            tokens,
            64,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        started = time.perf_counter_ns()

        moe.eval()
        with torch.no_grad():
            rollout_output, rollout_record = capture_forward(
                moe, recorder, hidden_states
            )
        reference_digest = rollout_record["expert_indices_sha256"]

        moe.train()
        wrapped.zero_grad(set_to_none=True)
        recorder.clear()
        train_input = hidden_states.clone().requires_grad_(True)
        train_result = wrapped(train_input)
        train_loss = train_result["hidden_states"].float().square().mean()
        if train_result["aux_loss"] is not None:
            train_loss = train_loss + 0.01 * train_result["aux_loss"]
        train_loss.backward()
        if len(recorder.records) != 1:
            raise AssertionError(
                f"expected one training route, got {len(recorder.records)}"
            )
        training_record = recorder.records[0]

        wrapped.zero_grad(set_to_none=True)
        checkpointing = GradientCheckpointingCallback(modules=[DeepSeekMoE])
        checkpointing._enable(moe)
        recorder.clear()
        recompute_input = hidden_states.clone().requires_grad_(True)
        recompute_result = wrapped(recompute_input)
        recompute_loss = recompute_result["hidden_states"].float().square().mean()
        if recompute_result["aux_loss"] is not None:
            recompute_loss = recompute_loss + 0.01 * recompute_result["aux_loss"]
        recompute_loss.backward()
        checkpointing._disable(moe)
        if len(recorder.records) < 2:
            raise AssertionError("checkpointed backward did not replay MoE routing")
        recompute_records = list(recorder.records)
        recompute_forward_total += len(recompute_records)

        state = {key: value.detach().clone() for key, value in moe.state_dict().items()}
        resumed = make_moe(experts, top_k, device)
        resumed.load_state_dict(state)
        resumed_recorder = RouteRecorder(resumed)
        resumed.eval()
        with torch.no_grad():
            resumed_output, resumed_record = capture_forward(
                resumed, resumed_recorder, hidden_states
            )
        resumed_recorder.close()

        records = [training_record, *recompute_records, resumed_record]
        mismatch_total += sum(
            record["expert_indices_sha256"] != reference_digest for record in records
        )
        if not torch.equal(resumed_output, rollout_output):
            mismatch_total += 1

        assert_across_ranks(reference_digest, world_size, "expert route")
        assert_across_ranks(
            rollout_record["router_logits_sha256"], world_size, "router logits"
        )
        assert_across_ranks(
            rollout_record["router_probs_sha256"], world_size, "router probabilities"
        )

        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        maximum_elapsed = torch.tensor(elapsed_ms, device=device)
        dist.all_reduce(maximum_elapsed, op=dist.ReduceOp.MAX)
        if rank == 0:
            latencies_ms.append(maximum_elapsed.item())
        expert_counts = rollout_record["expert_token_counts"]
        route_digest = reference_digest
        logits_digest = rollout_record["router_logits_sha256"]
        probs_digest = rollout_record["router_probs_sha256"]

    mismatch_tensor = torch.tensor(mismatch_total, device=device, dtype=torch.int64)
    dist.all_reduce(mismatch_tensor, op=dist.ReduceOp.SUM)
    if mismatch_tensor.item() != 0:
        raise AssertionError(f"observed {mismatch_tensor.item()} routing mismatches")
    recorder.close()

    if rank != 0:
        return {}
    return {
        "tokens": tokens,
        "experts": experts,
        "top_k": top_k,
        "trials": trials,
        "mismatch_total": mismatch_tensor.item(),
        "route_sha256": route_digest,
        "router_logits_sha256": logits_digest,
        "router_probs_sha256": probs_digest,
        "expert_token_counts": expert_counts,
        "checkpoint_route_observations": recompute_forward_total,
        "trial_ms_p50": statistics.median(latencies_ms),
        "trial_ms_p95": percentile(latencies_ms, 0.95),
        "allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
        "reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("trials must be positive")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    cases = []
    for tokens, experts, top_k in ((128, 8, 2), (512, 64, 8)):
        result = run_case(
            experts=experts,
            top_k=top_k,
            tokens=tokens,
            trials=args.trials,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            cases.append(result)

    if rank == 0:
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "gpu": torch.cuda.get_device_name(device),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "dtype": "bfloat16",
                    "world_size": world_size,
                    "cases": cases,
                },
                indent=2,
            )
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
