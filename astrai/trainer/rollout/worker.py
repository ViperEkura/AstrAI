"""A rollout replica owned by one spawned process and CUDA device."""

import time
import traceback
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import torch

from astrai.trainer.backend import ReplicaBackend
from astrai.trainer.rollout.batching import merge_rollouts, slice_batch
from astrai.trainer.rollout.generator import RolloutGenerator
from astrai.trainer.rollout.protocol import (
    GenerationResult,
    MessageKind,
    RolloutMessage,
    RolloutProtocolError,
    WorkerReady,
    recv_message,
    send_message,
)
from astrai.trainer.rollout.types import SamplingParams
from astrai.trainer.rollout.weight_transport import (
    _copy_shared_weights,
    _register_shared,
    _shared_views,
    _unregister_shared,
)


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
    shm_name: str
    layout: list
    max_prompts_per_worker: int


def run_rollout_worker(conn, spec: RolloutWorkerSpec):
    shm = None
    registered = False
    views = None
    request_id = 0
    device = spec.device
    try:
        torch.cuda.set_device(device)
        from astrai.tokenize import AutoTokenizer

        shm = SharedMemory(name=spec.shm_name)
        registered = _register_shared(shm, device)
        views = _shared_views(shm, spec.layout)
        model = spec.model_fn().to(
            device=device, dtype=getattr(torch, spec.model_dtype)
        )
        model.requires_grad_(False)
        model.eval()
        target = dict(model.state_dict(keep_vars=True))
        pinned = (
            None
            if registered
            else {
                name: torch.empty(
                    shape, dtype=getattr(torch, dtype_name), pin_memory=True
                )
                for name, shape, dtype_name, _, _ in spec.layout
            }
        )
        _copy_shared_weights(views, spec.layout, target, pinned, device)
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
                    registered,
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

                def copy_weights(_version):
                    return _copy_shared_weights(
                        views, spec.layout, target, pinned, device
                    )

                backend.apply_weight_update(version, copy_weights)
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
                if round_id is None or version is None or not isinstance(chunk, dict):
                    raise RolloutProtocolError("generate command is incomplete")
                if backend.policy_version != version:
                    raise RuntimeError(
                        "rollout worker did not receive requested version"
                    )
                started = time.perf_counter()
                total = len(next(iter(chunk.values())))
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
                            generator.generate(slice_batch(chunk, indices, total)),
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
        if shm is not None:
            if views is not None:
                views.clear()
            if registered:
                _unregister_shared(shm, device)
            shm.close()
