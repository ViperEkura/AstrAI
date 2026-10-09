from typing import Optional

from astrai.inference.core.scheduler import Scheduler
from astrai.model.autoregressive_lm import AutoRegressiveLM
from tests.support.models import make_rollout_config
from tests.support.tokenizers import FakeTokenizer


def make_cpu_scheduler(
    model,
    tokenizer=None,
    *,
    max_batch_size: int = 8,
    max_seq_len: int = 64,
    enable_overlap: bool = False,
    enable_cuda_graph: bool = False,
    device: Optional[str] = None,
    **overrides,
) -> Scheduler:
    """Standard test scheduler: CPU-safe defaults, arbitrary overrides.

    Every keyword lands in ``Scheduler(...)`` verbatim, so tests keep direct
    control of page_size/kv_tokens/token_budget while sharing the no-graph,
    torch-native baseline that keeps the suite CPU-runnable.
    """
    if tokenizer is None:
        tokenizer = FakeTokenizer()
    kwargs = dict(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        enable_cuda_graph=enable_cuda_graph,
        enable_overlap=enable_overlap,
        backend="torch_native",
    )
    if device is not None:
        kwargs["device"] = device
    kwargs.update(overrides)
    return Scheduler(model=model, tokenizer=tokenizer, **kwargs)


def make_cpu_model(max_position_embeddings: int = 64):
    """Tiny deterministic-sequence model for scheduler-pipeline tests."""
    return AutoRegressiveLM(
        make_rollout_config(max_position_embeddings=max_position_embeddings)
    ).eval()
