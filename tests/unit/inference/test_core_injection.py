from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from astrai.extension.runtime.context import OpSelection
from astrai.inference.core.engine_core import EngineCore
from astrai.inference.frontend.engine import InferenceEngine
from tests.support.tokenizers import FakeTokenizer


def test_engine_operations_use_injected_core():
    core = MagicMock(spec=EngineCore)
    core.max_batch_size = 2
    core.max_seq_len = 16
    core.op_selection = OpSelection((("attention", "torch"), ("swiglu", "cuda")))
    core.cuda_graph_enabled = False
    core.score_ids.return_value = [0.5]
    core.get_stats.return_value = {"queued": 0}
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32))

    engine = InferenceEngine(model, FakeTokenizer(), core=core)
    core.set_event_sink.assert_called_once()
    core.start.assert_called_once()
    assert engine.score([1], [2]) == 0.5
    core.score_ids.assert_called_once_with([[1]], [[2]], per_token=False)
    assert engine.get_stats() == {"queued": 0}
    assert engine.op_selection.choice("attention")[0] == "torch"
    assert engine.op_selection.choice("swiglu")[0] == "cuda"
    assert engine.cuda_graph_enabled is False
    engine.shutdown()
    core.stop.assert_called_once()
    with pytest.raises(AttributeError):
        _ = engine.scheduler
