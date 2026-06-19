import json
import os

from astrai.config.preprocess_config import (
    InputConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from astrai.preprocessing.pipeline import Pipeline, filter_by_length
from tests.data.conftest import (
    _CHAT_SECTIONS,
    _INSTRUCTION_SECTIONS,
    _TEXT_SECTIONS,
    make_dpo_chat_config,
    make_grpo_no_template_config,
)


def test_filter_by_length():
    assert filter_by_length("hello world", min_len=5)
    assert not filter_by_length("hi", min_len=5)
    assert not filter_by_length("x" * 100, max_len=50)
    assert filter_by_length("just right", min_len=5, max_len=20)


def test_full_chat_pipeline(temp_dir, chat_tokenizer_dir):
    jsonl_path = os.path.join(temp_dir, "chat.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hi."},
                        {"role": "assistant", "content": "Hello!"},
                    ]
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "4"},
                    ]
                }
            )
            + "\n"
        )

    config = PipelineConfig(
        input=InputConfig(sections=_CHAT_SECTIONS),
        mask={"system": "mask", "user": "mask", "assistant": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=2048),
        output=OutputConfig(storage_format="bin", domain_key=None),
    )

    out_dir = os.path.join(temp_dir, "output")
    Pipeline(
        config=config,
        input_paths=[jsonl_path],
        output_dir=out_dir,
        tokenizer_path=chat_tokenizer_dir,
    ).run()

    meta_path = os.path.join(out_dir, "__default__", "shard_0000", "meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert "sequence" in meta
    assert "loss_mask" in meta
    assert meta["sequence"]["dtype"] == "int32"
    assert meta["loss_mask"]["dtype"] == "int32"


def test_full_text_pipeline(temp_dir, tokenizer_dir):
    jsonl_path = os.path.join(temp_dir, "text.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "text": "Hello world this is a test document with enough characters to pass the minimum length filter."
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "text": "Another document for testing purposes with sufficient length to be processed."
                }
            )
            + "\n"
        )

    config = PipelineConfig(
        input=InputConfig(sections=_TEXT_SECTIONS),
        preprocessing=ProcessingConfig(max_seq_len=2048, min_chars=10),
        output=OutputConfig(storage_format="bin"),
    )

    out_dir = os.path.join(temp_dir, "output")
    Pipeline(
        config=config,
        input_paths=[jsonl_path],
        output_dir=out_dir,
        tokenizer_path=tokenizer_dir,
    ).run()

    meta_path = os.path.join(out_dir, "__default__", "shard_0000", "meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert "sequence" in meta
    assert "loss_mask" not in meta


def test_full_instruction_pipeline(temp_dir, tokenizer_dir):
    jsonl_path = os.path.join(temp_dir, "instruct.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "prompt": "Tell me a joke",
                    "response": "Why did the chicken cross the road?",
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "prompt": "What is AI?",
                    "response": "Artificial Intelligence is a field of computer science.",
                }
            )
            + "\n"
        )

    config = PipelineConfig(
        input=InputConfig(sections=_INSTRUCTION_SECTIONS),
        mask={"prompt": "mask", "response": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=2048),
        output=OutputConfig(storage_format="bin"),
    )

    out_dir = os.path.join(temp_dir, "output")
    Pipeline(
        config=config,
        input_paths=[jsonl_path],
        output_dir=out_dir,
        tokenizer_path=tokenizer_dir,
    ).run()

    meta_path = os.path.join(out_dir, "__default__", "shard_0000", "meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert "sequence" in meta
    assert "loss_mask" in meta


def test_dtype_override(temp_dir, tokenizer_dir):
    jsonl_path = os.path.join(temp_dir, "data.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "Q", "response": "A"}) + "\n")

    config = PipelineConfig(
        input=InputConfig(sections=_INSTRUCTION_SECTIONS),
        mask={"prompt": "mask", "response": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=2048),
        output=OutputConfig(storage_format="bin", dtype={"loss_mask": "bool"}),
    )

    out_dir = os.path.join(temp_dir, "output")
    Pipeline(
        config=config,
        input_paths=[jsonl_path],
        output_dir=out_dir,
        tokenizer_path=tokenizer_dir,
    ).run()

    meta_path = os.path.join(out_dir, "__default__", "shard_0000", "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["sequence"]["dtype"] == "int32"
    assert meta["loss_mask"]["dtype"] == "bool"


def test_dpo_pipeline(temp_dir, chat_tokenizer_dir):
    jsonl_path = os.path.join(temp_dir, "dpo.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "chosen": [
                        {"role": "user", "content": "Hi."},
                        {"role": "assistant", "content": "Hello!"},
                    ],
                    "rejected": [
                        {"role": "user", "content": "Hi."},
                        {"role": "assistant", "content": "Go away."},
                    ],
                }
            )
            + "\n"
        )

    out_dir = os.path.join(temp_dir, "output")
    Pipeline(
        config=make_dpo_chat_config(),
        input_paths=[jsonl_path],
        output_dir=out_dir,
        tokenizer_path=chat_tokenizer_dir,
    ).run()

    meta_path = os.path.join(out_dir, "__default__", "shard_0000", "meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert "chosen" in meta
    assert "rejected" in meta
    assert "chosen_mask" in meta
    assert "rejected_mask" in meta
    assert "sequence" not in meta


def test_grpo_pipeline(temp_dir, tokenizer_dir):
    jsonl_path = os.path.join(temp_dir, "grpo.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "prompt": "Question?",
                    "responses": ["Answer A", "Answer B"],
                    "rewards": [0.8, 0.3],
                }
            )
            + "\n"
        )

    out_dir = os.path.join(temp_dir, "output")
    Pipeline(
        config=make_grpo_no_template_config(),
        input_paths=[jsonl_path],
        output_dir=out_dir,
        tokenizer_path=tokenizer_dir,
    ).run()

    meta_path = os.path.join(out_dir, "__default__", "shard_0000", "meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert "prompts" in meta
    assert "responses" in meta
    assert "masks" in meta
    assert "rewards" in meta
    assert "sequence" not in meta
