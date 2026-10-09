"""A rollout replica owned by one spawned process and CUDA device."""

import time
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from astrai.tokenize import AutoTokenizer
from astrai.trainer.backend import ReplicaBackend
from astrai.trainer.rollout.batching import merge_rollouts, slice_batch
from astrai.trainer.rollout.generator import RolloutGenerator
from astrai.trainer.rollout.nccl_transport import NCCLWeightChannel
from astrai.trainer.rollout.protocol import (
    GenerationRequest,
    GenerationResult,
    MessageKind,
    RolloutMessage,
    RolloutProtocolError,
    WorkerReady,
    recv_message,
    send_message,
)
from astrai.trainer.rollout.types import SamplingParams


@dataclass(frozen=True)
class RolloutWorkerSpec:
    device: str
    model_fn: object
    param_path: str
    params: SamplingParams
    max_batch_size: int
    max_seq_len: Optional[int]
    policy_version: int
    model_dtype: str
    rank: int
    world_size: int
    rendezvous_port: int
    nccl_timeout_s: float
    layout: List[Tuple[str, Tuple[int, ...], str]]
    max_prompts_per_worker: int


def run_rollout_worker(conn, spec: RolloutWorkerSpec):
    channel = None
    failed = False
    request_id = 0
    device = spec.device
    try:
        torch.cuda.set_device(device)
        model = spec.model_fn().to(
            device=device, dtype=getattr(torch, spec.model_dtype)
        )
        model.requires_grad_(False)
        model.eval()
        channel = NCCLWeightChannel(
            model, spec.policy_version, expected_layout=spec.layout
        )
        send_message(
            conn,
            RolloutMessage(
                MessageKind.MODEL_READY,
                request_id=0,
                policy_version=spec.policy_version,
            ),
        )
        initial = recv_message(conn)
        request_id = initial.request_id
        if (
            initial.kind != MessageKind.WEIGHT_SYNC
            or initial.policy_version != spec.policy_version
            or initial.payload is not None
        ):
            raise RolloutProtocolError("initial weight sync command is invalid")
        channel.connect(
            rank=spec.rank,
            world_size=spec.world_size,
            port=spec.rendezvous_port,
            timeout_s=spec.nccl_timeout_s,
        )
        channel.broadcast()
        send_message(
            conn,
            RolloutMessage(
                MessageKind.WEIGHT_SYNC_ACK,
                request_id=request_id,
                policy_version=spec.policy_version,
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(spec.param_path)
        backend = ReplicaBackend(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_batch_size=spec.max_batch_size,
            max_seq_len=spec.max_seq_len,
            policy_version=spec.policy_version,
            enable_cuda_graph=True,
        )
        generator = RolloutGenerator(
            backend=backend,
            tokenizer=tokenizer,
            params=spec.params,
            output_device="cpu",
        )
        send_message(
            conn,
            RolloutMessage(
                MessageKind.READY,
                request_id=0,
                policy_version=spec.policy_version,
                payload=WorkerReady(
                    backend.scheduler.cuda_graph_enabled,
                    torch.cuda.max_memory_allocated(device),
                ),
            ),
        )
        while True:
            command = recv_message(conn)
            request_id = command.request_id
            if command.kind == MessageKind.STOP:
                break
            if command.kind == MessageKind.WEIGHT_SYNC:
                version = command.policy_version
                if version is None:
                    raise RolloutProtocolError("weight command is missing a version")
                if version <= backend.policy_version:
                    raise RuntimeError("weight version must advance")

                backend.apply_weight_update(
                    version, lambda _version: channel.broadcast()
                )
                send_message(
                    conn,
                    RolloutMessage(
                        MessageKind.WEIGHT_SYNC_ACK,
                        request_id=request_id,
                        policy_version=version,
                    ),
                )
            elif command.kind == MessageKind.GENERATE:
                round_id, version, chunk = (
                    command.round_id,
                    command.policy_version,
                    command.payload,
                )
                if (
                    round_id is None
                    or version is None
                    or not isinstance(chunk, GenerationRequest)
                ):
                    raise RolloutProtocolError("generate command is incomplete")
                if backend.policy_version != version:
                    raise RuntimeError(
                        "rollout worker did not receive requested version"
                    )
                seeds = chunk.seeds
                chunk = chunk.batch
                started = time.perf_counter()
                total = len(next(iter(chunk.values())))
                if len(seeds) != total * spec.params.group_size:
                    raise RolloutProtocolError(
                        "generate seeds do not match the prompt count"
                    )
                pieces = []
                for begin in range(0, total, spec.max_prompts_per_worker):
                    indices = list(
                        range(
                            begin,
                            min(begin + spec.max_prompts_per_worker, total),
                        )
                    )
                    pieces.append(
                        (
                            indices,
                            generator.generate(
                                slice_batch(chunk, indices, total),
                                seeds=[
                                    seeds[i * spec.params.group_size + g]
                                    for i in indices
                                    for g in range(spec.params.group_size)
                                ],
                            ),
                        )
                    )
                raw = merge_rollouts(pieces, total)
                send_message(
                    conn,
                    RolloutMessage(
                        MessageKind.RESULT,
                        request_id=request_id,
                        round_id=round_id,
                        policy_version=version,
                        payload=GenerationResult(
                            raw,
                            time.perf_counter() - started,
                            torch.cuda.max_memory_allocated(device),
                        ),
                    ),
                )
            else:
                raise RolloutProtocolError(
                    f"unexpected rollout command: {command.kind}"
                )
    except BaseException:
        failed = True
        try:
            send_message(
                conn,
                RolloutMessage(
                    MessageKind.ERROR,
                    request_id=request_id,
                    payload=traceback.format_exc(),
                ),
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        conn.close()
        if channel is not None:
            channel.close(force=failed)
