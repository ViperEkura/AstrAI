"""Document-isolated pretraining records preserve segment boundaries."""

import torch

from astrai.config.preprocess_config import PipelineConfig
from astrai.dataset import DatasetFactory, seq_collate_fn
from astrai.model import AutoRegressiveLM
from astrai.preprocessing.core import prepare_sequences
from astrai.serialization import save_bin
from astrai.trainer.strategy import SEQStrategy
from tests.support.models import make_tiny_config


def test_seq_record_preprocessing_keeps_segment_lengths():
    config = PipelineConfig.from_dict(
        {
            "preprocessing": {"max_seq_len": 5, "max_packed_len": 5},
            "output": {"record_keys": ["sequence", "seq_lengths"]},
        }
    )
    packed = prepare_sequences({"sequence": [[1, 2, 3], [11, 12]]}, config)
    assert packed["sequence"] == [[1, 2, 3, 11, 12]]
    assert packed["seq_lengths"] == [[3, 2]]


def test_seq_record_bin_collate_and_loss(tmp_path):
    group = {
        "sequence": [torch.tensor([1, 2, 3, 11, 12]), torch.tensor([21, 22])],
        "seq_lengths": [torch.tensor([3, 2]), torch.tensor([2])],
    }
    save_bin(str(tmp_path), group, record_keys=list(group))
    dataset = DatasetFactory.load("seq", str(tmp_path), window_size=5)
    assert len(dataset) == 2
    batch = seq_collate_fn([dataset[0], dataset[1]])
    assert batch["target_ids"].tolist() == [
        [2, 3, -100, 12, -100],
        [22, -100, -100, -100, -100],
    ]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1], [0, 1, 0, 0, 0]]
    assert batch["varlen_cu_seqlens"].tolist() == [0, 3, 5, 7]
    assert batch["varlen_indices"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert batch["varlen_max_seqlen"] == 3

    torch.manual_seed(3)
    model = AutoRegressiveLM(make_tiny_config(vocab_size=64)).eval()
    strategy = SEQStrategy(model, "cpu")
    separate = seq_collate_fn(
        [
            {"input_ids": torch.tensor(ids), "seq_lengths": torch.tensor([len(ids)])}
            for ids in ([1, 2, 3], [11, 12], [21, 22])
        ]
    )
    with torch.no_grad():
        torch.testing.assert_close(
            strategy(batch)["loss"], strategy(separate)["loss"], rtol=1e-5, atol=1e-6
        )
