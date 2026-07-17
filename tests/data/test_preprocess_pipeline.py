import json
import os

from astrai.config.preprocess_config import (
    InputConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from astrai.preprocessing.packing import PackingStrategyFactory
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


# ---------------------------------------------------------------------------
# BFD split packing
# ---------------------------------------------------------------------------

_TRU = "keep_start"


def _total_tokens(keys, key="sequence"):
    return sum(len(s) for s in keys[key])


def test_bfd_split_preserves_all_tokens():
    """No tokens are lost — split chunks are kept, not truncated away."""
    packer = PackingStrategyFactory.create("bfd_split")
    max_len = 10
    keys = {
        "sequence": [list(range(25)), list(range(3))],
        "loss_mask": [[1] * 25, [1] * 3],
    }
    result = packer.apply(keys, max_len, _TRU)

    assert _total_tokens(result) == 28
    for seq in result["sequence"]:
        assert len(seq) <= max_len


def test_bfd_split_chunk_alignment():
    """loss_mask chunks must align with sequence chunks."""
    packer = PackingStrategyFactory.create("bfd_split")
    max_len = 10
    keys = {
        "sequence": [list(range(25))],
        "loss_mask": [[0] * 5 + [1] * 20],
    }
    result = packer.apply(keys, max_len, _TRU)

    for seq, mask in zip(result["sequence"], result["loss_mask"]):
        assert len(seq) == len(mask)


def test_bfd_split_short_unchanged():
    """Sequences under max_packed_len should not be split."""
    packer = PackingStrategyFactory.create("bfd_split")
    max_len = 10
    keys = {"sequence": [list(range(5))], "loss_mask": [[1] * 5]}
    result = packer.apply(keys, max_len, _TRU)

    assert _total_tokens(result) == 5
    assert len(result["sequence"]) >= 1


def test_bfd_split_vs_bfd():
    """bfd loses tokens from over-length sequences; bfd_split does not."""
    max_len = 10
    keys = {
        "sequence": [list(range(25)), list(range(8))],
        "loss_mask": [[1] * 25, [1] * 8],
    }

    bfd = PackingStrategyFactory.create("bfd").apply(keys, max_len, _TRU)
    split = PackingStrategyFactory.create("bfd_split").apply(keys, max_len, _TRU)

    assert _total_tokens(bfd) < 33
    assert _total_tokens(split) == 33
