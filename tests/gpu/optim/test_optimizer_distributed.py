import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard
from torch.nn.parallel import DistributedDataParallel as DDP

from astrai.model import AutoRegressiveLM
from astrai.optim import NoraNAdamW
from astrai.parallel.setup import find_free_port
from tests.support.capabilities import skip_no_cuda
from tests.support.models import make_tiny_config

pytestmark = skip_no_cuda


def _assign_grads_and_step(model):
    optimizer = NoraNAdamW(model)
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.ones_like(param)
    optimizer.step()
    return optimizer


def test_nora_nadamw_steps_after_ddp_and_fsdp2_wrapping():
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        rank=0,
        world_size=1,
        init_method=f"tcp://127.0.0.1:{find_free_port()}",
    )
    try:
        ddp_model = AutoRegressiveLM(make_tiny_config()).to(
            device="cuda", dtype=torch.bfloat16
        )
        ddp_model = DDP(ddp_model, device_ids=[0], output_device=0)
        ddp_optimizer = _assign_grads_and_step(ddp_model)
        assert ddp_optimizer.state_dict()["nora"]["state"]

        fsdp_model = AutoRegressiveLM(make_tiny_config()).to(
            device="cuda", dtype=torch.bfloat16
        )
        for child in fsdp_model.children():
            if isinstance(child, torch.nn.ModuleList):
                for submodule in child:
                    fully_shard(submodule, reshard_after_forward=False)
            else:
                fully_shard(child, reshard_after_forward=False)

        fsdp_optimizer = _assign_grads_and_step(fsdp_model)
        nora_params = fsdp_optimizer.nora.param_groups[0]["params"]
        assert nora_params
        assert all(isinstance(param, DTensor) for param in nora_params)
        assert all(
            all(
                not isinstance(placement, Shard) or placement.dim == 0
                for placement in param.placements
            )
            for param in nora_params
        )
    finally:
        dist.destroy_process_group()
