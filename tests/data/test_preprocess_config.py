import pytest

from astrai.config.preprocess_config import (
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from astrai.preprocessing import Pipeline


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
