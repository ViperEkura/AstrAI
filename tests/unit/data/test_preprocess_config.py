import os

import pytest

from astrai.config.preprocess_config import (
    InputConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from astrai.preprocessing import Pipeline
from tests.support.data import INSTRUCTION_SECTIONS, TEXT_SECTIONS, make_dpo_chat_config


def test_default_values():
    config = PipelineConfig()
    assert config.version == 1
    assert config.mask == {}
    assert config.mask_default == "mask"
    assert config.preprocessing.max_seq_len == 2048
    assert config.preprocessing.overflow == "truncate"
    assert config.preprocessing.packing == "bfd"
    assert config.output.storage_format == "bin"
    assert config.output.record_keys == []
    assert config.input.sections is None


def test_from_dict_flat():
    data = {
        "version": 1,
        "input": {
            "sections": [{"field": "messages", "action": "$role", "template": True}]
        },
        "mask": {"system": "mask", "assistant": "train"},
        "mask_default": "mask",
        "preprocessing": {"max_seq_len": 1024},
        "output": {"storage_format": "bin"},
    }
    config = PipelineConfig.from_dict(data)
    assert config.input.sections == [
        {"field": "messages", "action": "$role", "template": True}
    ]
    assert config.mask == {"system": "mask", "assistant": "train"}
    assert config.preprocessing.max_seq_len == 1024
    assert config.output.storage_format == "bin"


def test_to_dict_roundtrip():
    config = PipelineConfig(
        input=InputConfig(sections=INSTRUCTION_SECTIONS),
        mask={"prompt": "mask", "response": "train"},
        mask_default="mask",
    )
    d = config.to_dict()
    config2 = PipelineConfig.from_dict(d)
    assert config2.input.sections == INSTRUCTION_SECTIONS
    assert config2.mask == {"prompt": "mask", "response": "train"}


def test_to_file_from_file(temp_dir):
    config = PipelineConfig(
        input=InputConfig(sections=TEXT_SECTIONS),
        mask={"text": "train"},
        mask_default="mask",
    )
    path = os.path.join(temp_dir, "config.json")
    config.to_file(path)
    loaded = PipelineConfig.from_file(path)
    assert loaded.input.sections == TEXT_SECTIONS
    assert loaded.mask == {"text": "train"}


def test_dpo_config_roundtrip(temp_dir):
    config = make_dpo_chat_config()
    config.output.record_keys = ["chosen", "rejected", "chosen_mask", "rejected_mask"]
    path = os.path.join(temp_dir, "config.json")
    config.to_file(path)
    loaded = PipelineConfig.from_file(path)
    assert loaded.input.sources is not None
    assert "chosen" in loaded.input.sources
    assert "rejected" in loaded.input.sources
    assert loaded.input.sections is None
    assert loaded.output.record_keys == config.output.record_keys


@pytest.mark.parametrize("keys", [["chosen", "chosen"], [""], [" "], "chosen", [1]])
def test_invalid_record_keys(keys):
    with pytest.raises(ValueError):
        PipelineConfig.from_dict({"output": {"record_keys": keys}})


@pytest.mark.parametrize("storage_format", ["jsonl", "parquet"])
def test_pipeline_rejects_unsupported_output_before_loading(storage_format, tmp_path):
    with pytest.raises(ValueError, match="storage_format"):
        OutputConfig(storage_format=storage_format)
    config = PipelineConfig()
    config.output.storage_format = storage_format
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="storage_format"):
        Pipeline(config, [], str(output), tokenizer_path="unused")
    assert not output.exists()


@pytest.mark.parametrize(
    "setting",
    [{"overflow": "reject"}, {"packing": "bfd_split"}, {"packing_strategy": "simple"}],
)
def test_reject_obsolete_or_invalid_packing(setting):
    with pytest.raises(ValueError):
        ProcessingConfig(**setting)
