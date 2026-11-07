import json

from astrai.config.preprocess_config import (
    InputConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from astrai.dataset import DatasetFactory
from astrai.preprocessing.packing import pack_sequences
from astrai.preprocessing.pipeline import Pipeline
from tests.support.data import (
    TEXT_SECTIONS,
)


def test_saved_config_executes_default_packing(tmp_path, tokenizer_dir):
    source = tmp_path / "input.jsonl"
    records = [{"text": text} for text in ("hello world " * 20, "hello", "world")]
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    config = PipelineConfig(
        input=InputConfig(sections=TEXT_SECTIONS),
        preprocessing=ProcessingConfig(max_seq_len=16, max_packed_len=16, min_chars=0),
        output=OutputConfig(
            dtype={"loss_mask": "bool"},
            record_keys=["sequence", "loss_mask", "seq_lengths"],
        ),
    )
    config_path = tmp_path / "config.json"
    config.to_file(str(config_path))
    output = tmp_path / "output"
    pipeline = Pipeline(
        PipelineConfig.from_file(str(config_path)),
        [str(source)],
        str(output),
        tokenizer_dir,
    )
    rendered = [pipeline.transform(record)["sequence"] for record in records]
    assert len(rendered[0]) > 16
    short_tokens = sum(map(len, rendered[1:]))
    assert short_tokens <= 16
    pipeline.run()
    dataset = DatasetFactory.load("sft", str(output), window_size=16)
    # Defaults truncate the long document and BFD merges the two short ones.
    assert len(dataset) == 2
    assert dataset.token_count == 16 + short_tokens
    meta = json.loads((output / "__default__" / "shard_0000" / "meta.json").read_text())
    assert meta["sequence"]["dtype"] == "int32"
    assert meta["loss_mask"]["dtype"] == "bool"


def test_bfd_split_preserves_all_tokens():
    max_len = 10
    keys = {
        "sequence": [list(range(25)), list(range(3))],
        "loss_mask": [[1] * 25, [1] * 3],
    }
    result = pack_sequences(keys, max_len, max_len, overflow="split")

    assert sum(map(len, result["sequence"])) == 28
    for seq, mask in zip(result["sequence"], result["loss_mask"], strict=True):
        assert len(seq) == len(mask) <= max_len
