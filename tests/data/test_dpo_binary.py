import copy
import glob
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from astrai.config.preprocess_config import PipelineConfig
from astrai.dataset import DatasetFactory, dpo_collate_fn
from astrai.dataset.storage import MmapStore
from astrai.model import AutoRegressiveLM
from astrai.preprocessing.pipeline import Pipeline
from astrai.preprocessing.writer import BinWriter
from astrai.serialization import save_bin
from astrai.trainer.strategy import DPOStrategy
from tests.support.data import make_dpo_chat_config
from tests.support.models import make_tiny_config


@pytest.mark.parametrize("packing", ["none", "bfd"])
@pytest.mark.parametrize(
    "shard_limit", [1, 100_000], ids=["multi-shard", "single-shard"]
)
def test_dpo_pipeline_binary_roundtrip(
    tmp_path, chat_tokenizer_dir, packing, shard_limit
):
    records = []
    for index in range(5):
        records.append(
            {
                name: (
                    [{"role": "user", "content": "Hi " * (index + 1)}]
                    if index % 2
                    else []
                )
                + [{"role": "assistant", "content": text * count}]
                for name, text, count in (
                    ("chosen", "hello ", index + 2),
                    ("rejected", "world ", 7 - index),
                )
            }
        )
    source = tmp_path / "dpo.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    config = make_dpo_chat_config()
    config.output.record_keys = ["chosen", "rejected", "chosen_mask", "rejected_mask"]
    config.preprocessing.packing = packing
    config.output.max_tokens_per_shard = shard_limit
    pipeline = Pipeline(
        config, [str(source)], str(tmp_path / "out"), chat_tokenizer_dir
    )
    expected = []
    for record in records:
        transformed = pipeline.transform(record)
        assert transformed is not None
        expected.append(
            {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in transformed.items()
                if key != "domain"
            }
        )
        for name in ("chosen", "rejected"):
            expected[-1].setdefault(f"{name}_mask", torch.ones_like(expected[-1][name]))
    pipeline.run()
    metadata = sorted((tmp_path / "out").rglob("meta.json"))
    assert len(metadata) == (5 if shard_limit == 1 else 1)
    for shard_index, path in enumerate(metadata):
        meta = json.loads(path.read_text())
        shard_records = [expected[shard_index]] if shard_limit == 1 else expected
        for key in expected[0]:
            offsets = [0]
            for record in shard_records:
                offsets.append(offsets[-1] + len(record[key]))
            assert meta[key].get("offsets") == offsets
    if shard_limit == 1:
        expected = [
            expected[int(Path(path).parent.name.removeprefix("shard_"))]
            for path in glob.glob(
                str(tmp_path / "out" / "**" / "meta.json"), recursive=True
            )
        ]
    dataset = DatasetFactory.load("dpo", str(tmp_path / "out"))
    assert len(dataset) == len(expected)
    for index, record in enumerate(expected):
        for key, value in record.items():
            value = value.bool() if key.endswith("_mask") else value
            torch.testing.assert_close(dataset[index][key], value)
    batches = list(DataLoader(dataset, batch_size=3, collate_fn=dpo_collate_fn))
    expected = [
        {
            key: value.bool() if key.endswith("_mask") else value
            for key, value in record.items()
        }
        for record in expected
    ]
    for batch_index, batch in enumerate(batches):
        direct = dpo_collate_fn(expected[batch_index * 3 : batch_index * 3 + 3])
        for key in direct:
            torch.testing.assert_close(batch[key], direct[key])
    torch.manual_seed(17)
    model = AutoRegressiveLM(make_tiny_config(vocab_size=512))
    reference = copy.deepcopy(model).requires_grad_(False)
    strategy = DPOStrategy(model, "cpu", ref_model=reference)
    loss = strategy(batches[0])["loss"]
    direct_loss = strategy(dpo_collate_fn(expected[:3]))["loss"]
    torch.testing.assert_close(loss, direct_loss)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    assert all(parameter.grad is None for parameter in reference.parameters())


def _write_record_shards(root, shard_sizes):
    records = []
    for index, (chosen_length, rejected_length) in enumerate(
        [(3, 7), (8, 4), (5, 9), (7, 3), (4, 6)]
    ):
        records.append(
            {
                "chosen": torch.arange(chosen_length) + 100 * (index + 1),
                "rejected": torch.arange(rejected_length) + 1000 * (index + 1),
                "chosen_mask": torch.arange(chosen_length) >= 1,
                "rejected_mask": torch.arange(rejected_length) >= 1,
            }
        )
    begin = 0
    for index, count in enumerate(shard_sizes):
        group = {
            key: [record[key] for record in records[begin : begin + count]]
            for key in records[0]
        }
        save_bin(str(root / f"shard_{index:04d}"), group, record_keys=list(group))
        begin += count
    groups = []
    begin = 0
    for count in shard_sizes:
        groups.append(records[begin : begin + count])
        begin += count
    return [
        record
        for path in glob.glob(str(root / "**" / "meta.json"), recursive=True)
        for record in groups[int(Path(path).parent.name.removeprefix("shard_"))]
    ]


@pytest.mark.parametrize("shard_sizes", [(5,), (2, 2, 1)])
def test_dpo_record_shards(tmp_path, shard_sizes):
    records = _write_record_shards(tmp_path, shard_sizes)
    dataset = DatasetFactory.load("dpo", str(tmp_path))
    assert len(dataset) == 5
    for index in [4, 0, 2, 1, 3]:
        for key in records[index]:
            torch.testing.assert_close(dataset[index][key], records[index][key])
    batches = list(DataLoader(dataset, batch_size=3, collate_fn=dpo_collate_fn))
    assert [batch["chosen"].size(0) for batch in batches] == [3, 2]
    for index, batch in enumerate(batches):
        for key, value in dpo_collate_fn(records[index * 3 : index * 3 + 3]).items():
            torch.testing.assert_close(batch[key], value)
    with pytest.raises(ValueError, match="out of bounds"):
        dataset.store.fetch_record(5, "chosen")


def test_record_shards_preserve_stream_and_reload(tmp_path):
    records = _write_record_shards(tmp_path / "records", (2, 2, 1))
    store = MmapStore()
    store.load(str(tmp_path / "records"))
    assert store.num_records == 5
    for key in records[0]:
        direct = torch.cat([record[key] for record in records])
        torch.testing.assert_close(store.fetch(1, 12, key), direct[1:12])
    save_bin(str(tmp_path / "stream"), {"sequence": [torch.arange(20)]})
    store.load(str(tmp_path / "stream"))
    assert store.keys == ["sequence"]
    assert store.num_records == 0
    assert len(store) == 0
    torch.testing.assert_close(store.fetch(0, 20, "sequence"), torch.arange(20))
    store.load(str(tmp_path / "records"))
    assert store.num_records == 5
    torch.testing.assert_close(store[4]["chosen"], records[4]["chosen"])


@pytest.mark.parametrize(
    "offsets", [[1, 3, 11], [0, 8, 7], [0, 3, 12], [0, 3.5, 11], [0, True, 11], [0, 11]]
)
def test_record_shards_reject_invalid_offsets(tmp_path, offsets):
    _write_record_shards(tmp_path, (2, 2, 1))
    meta_path = tmp_path / "shard_0000" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["chosen"]["offsets"] = offsets
    meta_path.write_text(json.dumps(meta))
    store = MmapStore()
    store.load(str(tmp_path))
    with pytest.raises(ValueError, match="offsets|record count"):
        _ = store.num_records


@pytest.mark.parametrize("missing", ["offsets", "field"])
def test_record_shards_reject_incomplete_metadata(tmp_path, missing):
    _write_record_shards(tmp_path, (2, 2, 1))
    meta_path = tmp_path / "shard_0001" / "meta.json"
    meta = json.loads(meta_path.read_text())
    if missing == "offsets":
        del meta["chosen_mask"]["offsets"]
    else:
        del meta["chosen_mask"]
    meta_path.write_text(json.dumps(meta))
    store = MmapStore()
    store.load(str(tmp_path))
    with pytest.raises(ValueError, match="record|offsets|field"):
        _ = store.num_records


def test_dpo_rejects_misaligned_masks(tmp_path):
    _write_record_shards(tmp_path, (5,))
    meta_path = tmp_path / "shard_0000" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["chosen_mask"]["offsets"][1] += 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="chosen_mask"):
        DatasetFactory.load("dpo", str(tmp_path))


def test_dpo_legacy_binary_requires_reprocessing(tmp_path):
    group = {"chosen": [torch.tensor([1, 2])], "rejected": [torch.tensor([3, 4])]}
    group.update({f"{name}_mask": [torch.ones(2)] for name in ("chosen", "rejected")})
    save_bin(str(tmp_path), group)
    with pytest.raises(ValueError, match="offsets.*preprocess"):
        DatasetFactory.load("dpo", str(tmp_path))


@pytest.mark.parametrize("failure", ["missing", "count", "ndim"])
def test_record_writer_rejects_invalid_fields(tmp_path, failure):
    records = _write_record_shards(tmp_path / "source", (5,))
    group = {key: [record[key] for record in records] for key in records[0]}
    if failure == "missing":
        del group["rejected"]
    elif failure == "count":
        group["rejected"].pop()
    else:
        group["rejected_mask"][0] = torch.ones(1, 2)
    with pytest.raises(ValueError, match="Record|rejected"):
        BinWriter().save(
            str(tmp_path / "out"), "default", 0, group, record_keys=list(records[0])
        )
    assert not (tmp_path / "out" / "default" / "shard_0000").exists()


@pytest.mark.parametrize("key", ["chosen", "rejected", "chosen_mask", "rejected_mask"])
def test_writer_does_not_infer_dpo(tmp_path, key):
    BinWriter().save(str(tmp_path), "default", 0, {key: [torch.arange(6)]})
    meta = json.loads((tmp_path / "default" / "shard_0000" / "meta.json").read_text())
    assert set(meta) == {key}
    assert "offsets" not in meta[key]


@pytest.mark.parametrize("record_mode", [False, True])
@pytest.mark.parametrize("names", [("chosen", "rejected"), ("question", "answer")])
def test_custom_pipeline_masks(tmp_path, chat_tokenizer_dir, record_mode, names):
    sources = {
        name: {
            "sections": [{"field": name, "action": "$role", "template": True}],
            "mask_key": f"custom_{name}",
        }
        for name in names
    }
    keys = [key for name in names for key in (name, f"custom_{name}")]
    config = PipelineConfig.from_dict(
        {
            "input": {"sources": sources},
            "mask": {"user": "mask", "assistant": "train"},
            "preprocessing": {"min_chars": 0},
            "output": {"record_keys": keys if record_mode else []},
        }
    )
    record = {
        name: [{"role": "assistant", "content": "hello " * (index + 1)}]
        for index, name in enumerate(names)
    }
    source = tmp_path / "custom.jsonl"
    source.write_text(json.dumps(record) + "\n")
    pipeline = Pipeline(
        config, [str(source)], str(tmp_path / "out"), chat_tokenizer_dir
    )
    expected = pipeline.transform(record)
    pipeline.run()
    meta_path = next((tmp_path / "out").rglob("meta.json"))
    meta = json.loads(meta_path.read_text())
    assert set(meta) == set(keys)
    store = MmapStore()
    store.load(str(tmp_path / "out"))
    for key in keys:
        assert ("offsets" in meta[key]) == record_mode
        assert meta[key]["shape"] == [len(expected[key])]
        length = len(expected[key]) if record_mode else store.token_count
        value = (
            store.fetch_record(0, key) if record_mode else store.fetch(0, length, key)
        )
        torch.testing.assert_close(
            value, torch.tensor(expected[key][:length], dtype=torch.int32)
        )


def test_custom_pipeline_single_chosen(tmp_path, chat_tokenizer_dir):
    config = PipelineConfig.from_dict(
        {
            "input": {
                "sources": {
                    "chosen": {"sections": [{"field": "text", "action": "train"}]}
                }
            },
            "preprocessing": {"min_chars": 0},
        }
    )
    source = tmp_path / "custom.jsonl"
    source.write_text(json.dumps({"text": "hello world"}) + "\n")
    Pipeline(config, [str(source)], str(tmp_path / "out"), chat_tokenizer_dir).run()
    meta = json.loads(next((tmp_path / "out").rglob("meta.json")).read_text())
    assert set(meta) == {"chosen"}
    assert "offsets" not in meta["chosen"]


def test_pipeline_rejects_misaligned_record_masks(tmp_path, chat_tokenizer_dir):
    records = _write_record_shards(tmp_path / "source", (5,))
    config = make_dpo_chat_config()
    config.output.record_keys = list(records[0])
    pipeline = Pipeline(config, [], str(tmp_path / "out"), chat_tokenizer_dir)
    group = {key: [record[key].tolist() for record in records] for key in records[0]}
    group["rejected_mask"][0] = [1]
    with pytest.raises(ValueError, match="rejected_mask"):
        pipeline._flush({"default": group}, {"default": 0})
    assert not list((tmp_path / "out").rglob("meta.json"))


@pytest.mark.parametrize("failure", ["offsets", "record_count"])
def test_stream_does_not_require_valid_record_index(tmp_path, failure):
    group = {
        "sequence": [torch.arange(6), torch.arange(6, 12)],
        "aux": [torch.arange(4), torch.arange(4, 8), torch.arange(8, 12)],
    }
    save_bin(str(tmp_path), group, record_keys=list(group))
    if failure == "offsets":
        meta_path = tmp_path / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["sequence"]["offsets"][0] = 1
        meta_path.write_text(json.dumps(meta))
    store = MmapStore(window_size=3)
    store.load(str(tmp_path))
    assert len(store) == 3
    assert store.sample_window(0) == (0, 3)
    torch.testing.assert_close(store[0]["sequence"], torch.arange(3))
    torch.testing.assert_close(store.fetch(0, 12, "sequence"), torch.arange(12))
    assert not store._record_index_ready
    with pytest.raises(ValueError, match="offsets|record count"):
        store.fetch_record(0, "sequence")
    torch.testing.assert_close(store.fetch(0, 12, "sequence"), torch.arange(12))


@pytest.mark.parametrize("entry", ["count", "length", "item", "fetch", "dataset"])
def test_record_entry_points_validate_offsets(tmp_path, entry):
    _write_record_shards(tmp_path, (5,))
    meta_path = tmp_path / "shard_0000" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["chosen"]["offsets"][0] = 1
    meta_path.write_text(json.dumps(meta))
    store = MmapStore()
    store.load(str(tmp_path))
    with pytest.raises(ValueError, match="offsets"):
        if entry == "count":
            _ = store.num_records
        elif entry == "length":
            len(store)
        elif entry == "item":
            _ = store[0]
        elif entry == "fetch":
            store.fetch_record(0, "chosen")
        else:
            DatasetFactory.load("dpo", store=store)


def test_record_getitem_does_not_hide_stream_fields(tmp_path):
    save_bin(
        str(tmp_path),
        {"tokens": [torch.arange(3), torch.arange(3, 7)], "aux": [torch.arange(7)]},
        record_keys=["tokens"],
    )
    store = MmapStore()
    store.load(str(tmp_path))
    torch.testing.assert_close(store.fetch_record(1, "tokens"), torch.arange(3, 7))
    offsets = store._offsets
    assert store.num_records == 2
    assert store._offsets is offsets
    assert set(store.keys) == {"tokens", "aux"}
    with pytest.raises(ValueError, match="aux.*offsets"):
        _ = store[0]


def test_bin_load_preserves_discovery_order(tmp_path, monkeypatch):
    paths = []
    for index in range(3):
        path = tmp_path / f"shard_{index:04d}"
        save_bin(
            str(path),
            {"sequence": [torch.arange(4) + index * 10]},
            record_keys=["sequence"],
        )
        paths.append(str(path / "meta.json"))
    monkeypatch.setattr(
        "astrai.dataset.storage.glob.glob",
        lambda *args, **kwargs: [paths[2], paths[0], paths[1]],
    )
    store = MmapStore(window_size=3)
    store.load(str(tmp_path))
    expected = torch.cat([torch.arange(4) + index * 10 for index in (2, 0, 1)])
    torch.testing.assert_close(store.fetch(0, 12, "sequence"), expected)
    assert len(store) == 3
    for index in range(3):
        torch.testing.assert_close(
            store.fetch_record(index, "sequence"), expected[index * 4 : index * 4 + 4]
        )
