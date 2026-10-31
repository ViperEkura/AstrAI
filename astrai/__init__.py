__version__ = "1.4.0"
__author__ = "ViperEkura"

from astrai.config import (
    AutoRegressiveLMConfig,
    BaseModelConfig,
    ConfigFactory,
    EncoderConfig,
    PipelineConfig,
    TrainConfig,
)
from astrai.dataset import (
    BaseDataset,
    DatasetFactory,
    RDSampler,
    Store,
    StoreFactory,
)
from astrai.factory import BaseFactory
from astrai.inference import (
    InferenceEngine,
    build_engine,
    get_app,
    run_server,
    sample,
)
from astrai.inference.network import ProtocolHandler
from astrai.inference.worker.sample import sample
from astrai.logging import setup_logging
from astrai.model import (
    AutoModel,
    AutoRegressiveLM,
    EmbeddingEncoder,
    LoRAConfig,
    inject_lora,
)
from astrai.parallel import (
    ExecutorFactory,
    get_rank,
    get_world_size,
    only_on_rank,
    spawn_parallel_fn,
)
from astrai.preprocessing import Pipeline, filter_by_length
from astrai.serialization import Checkpoint
from astrai.tokenize import AutoTokenizer, ChatTemplate
from astrai.trainer import (
    BaseStrategy,
    StrategyFactory,
    TrainCallback,
    Trainer,
)

__all__ = [
    "AutoModel",
    "AutoRegressiveLM",
    "AutoRegressiveLMConfig",
    "AutoTokenizer",
    "BaseDataset",
    "BaseFactory",
    "BaseModelConfig",
    "BaseStrategy",
    "ChatTemplate",
    "Checkpoint",
    "ConfigFactory",
    "DatasetFactory",
    "EmbeddingEncoder",
    "EncoderConfig",
    "ExecutorFactory",
    "InferenceEngine",
    "LoRAConfig",
    "Pipeline",
    "PipelineConfig",
    "ProtocolHandler",
    "RDSampler",
    "sample",
    "Store",
    "StoreFactory",
    "StrategyFactory",
    "TrainCallback",
    "TrainConfig",
    "Trainer",
    "build_engine",
    "filter_by_length",
    "get_app",
    "get_rank",
    "get_world_size",
    "inject_lora",
    "only_on_rank",
    "run_server",
    "sample",
    "setup_logging",
    "spawn_parallel_fn",
]

setup_logging()
