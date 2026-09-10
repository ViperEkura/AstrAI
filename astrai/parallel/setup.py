import logging
import os
import signal
import socket
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from astrai.parallel.topology import parse_device_order
from astrai.signal_handler import install_early_signal_handlers

logger = logging.getLogger(__name__)


def resolve_local_device_index(
    local_rank: int,
    local_world_size: int,
    device_type: str,
) -> int:
    """Map a logical local rank to an accelerator selected by the planner."""

    if not 0 <= local_rank < local_world_size:
        raise ValueError(
            f"local rank {local_rank} is outside local world size {local_world_size}"
        )
    value = os.environ.get("ASTRAI_DEVICE_ORDER")
    if value is None or device_type == "cpu":
        return local_rank
    return parse_device_order(value, local_world_size)[local_rank]


def find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])


def get_current_device():
    return os.environ["LOCAL_DEVICE"]


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


@contextmanager
def setup_parallel(
    rank: int,
    world_size: int,
    local_rank: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "29500",
    device_type: str = "cuda",
):

    if dist.is_available() and dist.is_initialized():
        yield dist.group.WORLD
        return

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    device_index = resolve_local_device_index(local_rank, local_world_size, device_type)

    if world_size <= 1:
        device_id = torch.device(device_type, device_index)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_DEVICE"] = str(device_id)
        yield None
        return

    device_id = torch.device(device_type, device_index)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_DEVICE"] = str(device_id)

    pg_kwargs = dict(rank=rank, world_size=world_size, backend=backend)
    if backend in ("nccl", "ccl"):
        pg_kwargs["device_id"] = device_id

    dist.init_process_group(**pg_kwargs)

    try:
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
        elif backend == "ccl" and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.set_device(device_id)

        yield dist.group.WORLD
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def only_on_rank(rank, sync=False):
    """
    decorator to run a function only on a specific rank.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ret_args = None
            if get_rank() == rank:
                ret_args = func(*args, **kwargs)

            if sync and dist.is_available() and dist.is_initialized():
                dist.barrier()

            return ret_args

        return wrapper

    return decorator


def _run_single_rank(
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: str,
    device_type: str,
    func: Callable,
    kwargs: dict,
):
    install_early_signal_handlers()
    with setup_parallel(
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        backend=backend,
        master_addr=master_addr,
        master_port=master_port,
        device_type=device_type,
    ):
        func(**kwargs)


class LaunchStrategy(ABC):
    """Strategy for launching a function in a distributed context."""

    def __init__(
        self,
        world_size: int,
        backend: str,
        master_addr: str,
        master_port: str,
        device_type: str,
        start_method: str,
    ):
        self.world_size = world_size
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        self.device_type = device_type
        self.start_method = start_method

    @abstractmethod
    def launch(self, func: Callable, **kwargs):
        raise NotImplementedError


class TorchrunStrategy(LaunchStrategy):
    """External orchestrator (torchrun, SLURM, K8s) — env vars pre-set."""

    def launch(self, func: Callable, **kwargs):
        install_early_signal_handlers()
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        with setup_parallel(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            backend=self.backend,
            master_addr=os.environ.get("MASTER_ADDR", self.master_addr),
            master_port=os.environ.get("MASTER_PORT", self.master_port),
            device_type=self.device_type,
        ):
            func(**kwargs)


class LocalStrategy(LaunchStrategy):
    """Local launcher — single-process or mp.start_processes."""

    def launch(self, func: Callable, **kwargs):
        args = (
            self.world_size,
            self.backend,
            self.master_addr,
            self.master_port,
            self.device_type,
            func,
            kwargs,
        )

        if self.world_size == 1:
            _run_single_rank(0, *args)
            return

        install_early_signal_handlers()
        ctx = mp.start_processes(
            _run_single_rank,
            args=args,
            nprocs=self.world_size,
            start_method=self.start_method,
            join=False,
        )

        parent_stop = threading.Event()
        original_handlers = {}

        def _parent_handler(signum, frame):
            sig = signal.Signals(signum)
            logger.warning(
                "Parent (pid=%d) received %s, forwarding to children...",
                os.getpid(),
                sig.name,
            )
            parent_stop.set()
            for p in ctx.processes:
                if p.is_alive():
                    p.terminate()

        for sig in (signal.SIGTERM, signal.SIGINT):
            prev = signal.signal(sig, _parent_handler)
            if prev not in (signal.SIG_DFL, signal.SIG_IGN, None, _parent_handler):
                original_handlers[sig] = prev

        try:
            while not ctx.join() and not parent_stop.is_set():
                pass
        except BaseException:
            logger.warning(
                "Parent received unexpected exception, terminating children..."
            )
            for p in ctx.processes:
                if p.is_alive():
                    p.terminate()
            raise
        finally:
            for sig, handler in original_handlers.items():
                signal.signal(sig, handler)

            for p in ctx.processes:
                p.join()

            ctx.join()


def _is_external_launcher() -> bool:
    """Whether an external launcher (torchrun/elastic/manual env) started us."""
    if dist.is_torchelastic_launched():
        return True
    if "LOCAL_WORLD_SIZE" in os.environ:
        return True
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return True
    return False


def spawn_parallel_fn(
    func: Callable,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: Optional[str] = None,
    device_type: str = "cuda",
    start_method: str = "spawn",
    **kwargs,
):
    if master_port is None:
        master_port = find_free_port()
    if _is_external_launcher():
        strategy = TorchrunStrategy(
            world_size, backend, master_addr, master_port, device_type, start_method
        )
    else:
        strategy = LocalStrategy(
            world_size, backend, master_addr, master_port, device_type, start_method
        )
    strategy.launch(func, **kwargs)
