"""CPU regressions for overflow, record packs and attention isolation."""

import json

import pytest
import torch
from torch.utils.data import DataLoader

from astrai.config.preprocess_config import PipelineConfig
from astrai.dataset import DatasetFactory, sft_collate_fn
from astrai.model import AutoRegressiveLM
from astrai.preprocessing.core import accumulate_record
from astrai.preprocessing.packing import pack_sequences, plan_bfd
from astrai.preprocessing.pipeline import Pipeline
from astrai.serialization import save_bin
from astrai.trainer.strategy import SFTStrategy, make_doc_boundary_mask
from tests.support.models import make_tiny_config


@pytest.mark.parametrize("overflow", ["truncate", "split"])
@pytest.mark.parametrize("packing", ["bfd", "none"])
def test_overflow_and_packing_are_independent(overflow, packing):
    keys = {
        "sequence": [list(range(13)), [20, 21]],
        "loss_mask": [[0] * 3 + [1] * 10, [0, 1]],
    }
    out = pack_sequences(keys, 8, 8, overflow=overflow, packing=packing)
    assert sum(map(len, out["sequence"])) == (15 if overflow == "split" else 10)
    assert len(out["sequence"]) == (
        3 if overflow == "split" and packing == "none" else 2
    )
    for tokens, mask, lengths in zip(
        out["sequence"], out["loss_mask"], out["seq_lengths"]
    ):
        assert len(tokens) == len(mask) == sum(lengths) <= 8
        assert mask == [int(t >= 3 and t != 20) for t in tokens]
        if packing == "none":
            assert len(lengths) == 1


def test_keep_end_and_bfd_planner_contract():
    result = pack_sequences(
        {"sequence": [list(range(11))]}, 4, 8, truncation_mode="keep_end"
    )
    assert result["sequence"] == [[7, 8, 9, 10]]
    assert plan_bfd([6, 4, 3, 2, 1], 8) == [[0, 3], [1, 2, 4]]
    with pytest.raises(ValueError):
        plan_bfd([9], 8)
    with pytest.raises(ValueError, match="aligned"):
        pack_sequences({"sequence": [[1, 2]], "loss_mask": [[1]]}, 4, 4)


def test_missing_masks_backfill_both_directions():
    bucket = {}
    for result in [
        {"sequence": [1, 2]},
        {"sequence": [3, 4, 5], "loss_mask": [0, 1, 1]},
        {"sequence": [6, 7]},
    ]:
        accumulate_record(bucket, result, result["sequence"])
    assert bucket["loss_mask"] == [[1, 1], [0, 1, 1], [1, 1]]


@pytest.mark.parametrize("overflow", ["truncate", "split"])
@pytest.mark.parametrize("packing", ["bfd", "none"])
def test_pipeline_keeps_full_render_until_overflow(
    tmp_path, tokenizer_dir, overflow, packing
):
    config = PipelineConfig.from_dict(
        {
            "input": {
                "sections": [
                    {"field": "text", "action": "train", "add_special_tokens": False}
                ]
            },
            "preprocessing": {
                "max_seq_len": 8,
                "max_packed_len": 8,
                "min_chars": 0,
                "overflow": overflow,
                "packing": packing,
            },
            "output": {"record_keys": ["sequence", "loss_mask", "seq_lengths"]},
        }
    )
    source = tmp_path / "input.jsonl"
    record = {"text": "hello world " * 20}
    source.write_text(json.dumps(record) + chr(10))
    pipeline = Pipeline(config, [str(source)], str(tmp_path / "out"), tokenizer_dir)
    full = pipeline.transform(record)["sequence"]
    assert len(full) > 8
    pipeline.run()
    dataset = DatasetFactory.load("sft", str(tmp_path / "out"), window_size=8)
    expected = len(full) if overflow == "split" else 8
    assert dataset.token_count == expected
    assert sum(dataset[i]["input_ids"].numel() for i in range(len(dataset))) == expected
    assert len(dataset) == ((len(full) + 7) // 8 if overflow == "split" else 1)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=sft_collate_fn)))
    assert batch["input_ids"].size(1) <= 8


def write_packs(path):
    # Second pack is short; the last full or partial record must never disappear.
    group = {
        "sequence": [torch.tensor([1, 2, 3, 11, 12]), torch.tensor([21, 22])],
        "loss_mask": [torch.ones(5, dtype=torch.bool), torch.ones(2, dtype=torch.bool)],
        "seq_lengths": [torch.tensor([3, 2]), torch.tensor([2])],
    }
    save_bin(str(path), group, record_keys=list(group))
    return DatasetFactory.load("sft", str(path), window_size=5)


def test_record_padding_labels_and_boundary_mask(tmp_path):
    dataset = write_packs(tmp_path)
    assert len(dataset) == 2
    assert dataset.token_count == 7
    batch = sft_collate_fn([dataset[0], dataset[1]])
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1], [0, 1, 0, 0, 0]]
    assert batch["target_ids"].tolist() == [
        [2, 3, -100, 12, -100],
        [22, -100, -100, -100, -100],
    ]
    assert batch["loss_mask"].sum().item() == 4
    attention = make_doc_boundary_mask(batch["position_ids"])
    assert not attention[0, 0, 3:, :3].any()
    assert not attention[0, 0].triu(1).any()
    assert attention[0, 0, 4, 3]
    # Padding queries retain a self edge, avoiding fully masked softmax rows.
    assert attention.diagonal(dim1=-2, dim2=-1).all()


@torch.no_grad()
def test_packed_loss_matches_independent_segments(tmp_path):
    dataset = write_packs(tmp_path)
    torch.manual_seed(1)
    model = AutoRegressiveLM(make_tiny_config(vocab_size=64)).eval()
    strategy = SFTStrategy(model, "cpu")
    packed = sft_collate_fn([dataset[0], dataset[1]])
    independent = [
        {
            "input_ids": torch.tensor(ids),
            "loss_mask": torch.ones(len(ids), dtype=torch.bool),
            "seq_lengths": torch.tensor([len(ids)]),
        }
        for ids in ([1, 2, 3], [11, 12], [21, 22])
    ]
    separate = sft_collate_fn(independent)
    torch.testing.assert_close(
        strategy(packed)["loss"], strategy(separate)["loss"], rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("lengths", [[0, 5], [3, 3], []])
def test_reject_invalid_segment_metadata(tmp_path, lengths):
    group = {
        "sequence": [torch.arange(5)],
        "loss_mask": [torch.ones(5)],
        "seq_lengths": [torch.tensor(lengths, dtype=torch.int32)],
    }
    # Empty mmap files cannot be loaded, so a zero-length segment tests the
    # empty/invalid metadata case without relying on storage IO errors.
    if not lengths:
        group["seq_lengths"] = [torch.tensor([0])]
    save_bin(str(tmp_path), group, record_keys=list(group))
    dataset = DatasetFactory.load("sft", str(tmp_path), window_size=5)
    with pytest.raises(ValueError, match="segment lengths"):
        dataset[0]


def test_reject_pack_larger_than_training_context(tmp_path):
    write_packs(tmp_path)
    dataset = DatasetFactory.load("sft", str(tmp_path), window_size=4)
    with pytest.raises(ValueError, match="exceeds training window"):
        dataset[0]


def test_trainer_selects_sft_collator():
    from astrai.config.train_config import ComponentSpec
    from astrai.trainer.components import resolve_collate

    spec = ComponentSpec(type="packed", params={"train_type": "sft"})
    assert resolve_collate(spec) is sft_collate_fn
