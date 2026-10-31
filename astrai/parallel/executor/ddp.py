"""DistributedDataParallel executor."""

import contextlib
import logging
import os

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from astrai.parallel.executor.base import (
    BaseExecutor,
    ExecutorFactory,
    strip_compile_prefix,
)
from astrai.parallel.setup import get_rank, get_world_size

logger = logging.getLogger(__name__)


@ExecutorFactory.register("ddp")
class DDPExecutor(BaseExecutor):
    """Sync initial state, with separate control of runtime buffer sync."""

    def __init__(
        self,
        grad_accum_steps: int = 1,
        dim: int = 0,
        init_sync: bool = True,
        process_group=None,
        bucket_cap_mb: int = 25,
        find_unused_parameters: bool = False,
        check_reduction: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False,
        delay_all_reduce_named_params=None,
        param_to_hook_all_reduce=None,
        mixed_precision=None,
        device_mesh=None,
        forward_sync_buffers: bool = True,
    ):
        super().__init__(grad_accum_steps=grad_accum_steps)
        self._ddp_kwargs = dict(
            dim=dim,
            init_sync=init_sync,
            process_group=process_group,
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
            check_reduction=check_reduction,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
            delay_all_reduce_named_params=delay_all_reduce_named_params,
            param_to_hook_all_reduce=param_to_hook_all_reduce,
            mixed_precision=mixed_precision,
            device_mesh=device_mesh,
            forward_sync_buffers=forward_sync_buffers,
        )

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        if not self.use_distributed:
            logger.warning("DDP backend selected but world_size=1, model not wrapped")
            return model
        local_rank = int(os.environ.get("LOCAL_RANK", get_rank()))
        cpu_model = next(model.parameters()).device.type == "cpu"
        model = DDP(
            model,
            device_ids=None if cpu_model else [local_rank],
            output_device=None if cpu_model else local_rank,
            **self._ddp_kwargs,
        )
        logger.info("Model wrapped with DDP (world_size=%d)", get_world_size())
        return model

    def _no_sync(self, model: nn.Module):
        if isinstance(model, DDP):
            return model.no_sync()
        return contextlib.nullcontext()

    def model_for_inference(self, model: nn.Module) -> nn.Module:
        model = super().model_for_inference(model)
        if isinstance(model, DDP):
            return model.module
        return model

    def model_state_dict(self, model: nn.Module):
        if isinstance(model, DDP):
            return strip_compile_prefix(model.module.state_dict())
        return strip_compile_prefix(model.state_dict())
