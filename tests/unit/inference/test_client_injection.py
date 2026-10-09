from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from astrai.inference.frontend.core_client import EngineCoreClient
from astrai.inference.frontend.engine import InferenceEngine
from tests.support.tokenizers import FakeTokenizer


def test_engine_operations_use_injected_client():
    client = MagicMock(spec=EngineCoreClient)
    client.max_batch_size = 2
    client.backend_name = "test"
    client.cuda_graph_enabled = False
    client.score_ids.return_value = [0.5]
    client.stats.return_value = {"queued": 0}
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32))

    engine = InferenceEngine(model, FakeTokenizer(), core_client=client)
    client.set_event_sink.assert_called_once()
    client.start.assert_called_once()
    assert engine.score([1], [2]) == 0.5
    client.score_ids.assert_called_once_with([[1]], [[2]], per_token=False)
    assert engine.get_stats() == {"queued": 0}
    assert engine.backend_name == "test"
    assert engine.cuda_graph_enabled is False
    engine.shutdown()
    client.shutdown.assert_called_once()
    with pytest.warns(DeprecationWarning):
        with pytest.raises(AttributeError):
            _ = engine.scheduler
