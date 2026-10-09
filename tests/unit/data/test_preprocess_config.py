import os

from astrai.config.preprocess_config import (
    InputConfig,
    PipelineConfig,
)
from tests.support.data import INSTRUCTION_SECTIONS, TEXT_SECTIONS, make_dpo_chat_config


def test_default_values():
    config = PipelineConfig()
    assert config.version == 1
    assert config.mask == {}
    assert config.mask_default == "mask"
    assert config.preprocessing.max_seq_len == 2048
    assert config.output.storage_format == "bin"
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
    path = os.path.join(temp_dir, "config.json")
    config.to_file(path)
    loaded = PipelineConfig.from_file(path)
    assert loaded.input.sources is not None
    assert "chosen" in loaded.input.sources
    assert "rejected" in loaded.input.sources
    assert loaded.input.sections is None
