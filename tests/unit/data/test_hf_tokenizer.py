"""HF tokenizer terminal tokens and template variables survive native loading."""

import json

import pytest
from tokenizers import Tokenizer, models

from astrai.tokenize import AutoTokenizer
from astrai.trainer.callbacks.checkpoint import _copy_tokenizer_files


def _write_tokenizer(root, *, eos="<eos>", generation=None):
    tokens = ["<unk>", "<pad>", "<bos>", "<eos>", "<eot>", "<think>", "hello"]
    tokenizer = Tokenizer(
        models.WordLevel({t: i for i, t in enumerate(tokens)}, unk_token="<unk>")
    )
    tokenizer.add_special_tokens(tokens[:-1])
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": eos,
                "pad_token": "<pad>",
                "unk_token": {"content": "<unk>", "special": True},
                "chat_template": "{{ bos_token }}{{ messages[0]['content'] }}{{ eos_token }}",
            }
        )
    )
    if generation is not None:
        (root / "generation_config.json").write_text(json.dumps(generation))


@pytest.mark.parametrize("added_token", [False, True])
def test_hf_named_tokens_use_only_declared_eos_and_render_template(
    tmp_path, added_token
):
    _write_tokenizer(tmp_path, eos={"content": "<eos>"} if added_token else "<eos>")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    assert tokenizer.eos_token_id == 3
    assert tokenizer.pad_token_id == 1
    assert tokenizer.unk_token_id == 0
    assert tokenizer.stop_ids == [3]
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}], tokenize=False
        )
        == "<bos>hello<eos>"
    )


def test_hf_generation_multiple_eos_survives_save_reload(tmp_path):
    _write_tokenizer(tmp_path, generation={"eos_token_id": [3, 4, 3]})
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    assert tokenizer.stop_ids == [3, 4]
    assert 5 not in tokenizer.stop_ids  # reasoning marker is not a terminal token
    target = tmp_path / "native"
    tokenizer.save_pretrained(str(target))
    restored = AutoTokenizer.from_pretrained(target)
    assert restored.stop_ids == [3, 4]
    assert restored.pad_token_id == 1


def test_checkpoint_keeps_every_hf_generation_terminal_token(tmp_path):
    _write_tokenizer(tmp_path, generation={"eos_token_id": [3, 4]})
    original = AutoTokenizer.from_pretrained(tmp_path)
    target = tmp_path / "checkpoint"
    target.mkdir()
    _copy_tokenizer_files(str(tmp_path), str(target))
    restored = AutoTokenizer.from_pretrained(target)
    assert restored.stop_ids == original.stop_ids == [3, 4]
    assert restored.pad_token_id == original.pad_token_id


@pytest.mark.parametrize("ids", [[True], [-1], [2**100], [9999], "3"])
def test_hf_rejects_invalid_terminal_token_declarations(tmp_path, ids):
    _write_tokenizer(tmp_path, generation={"eos_token_id": ids})
    with pytest.raises(ValueError, match="EOS"):
        AutoTokenizer.from_pretrained(tmp_path)


def test_hf_missing_declared_eos_fails_instead_of_length_only_generation(tmp_path):
    _write_tokenizer(tmp_path, eos="<not-in-vocabulary>")
    with pytest.raises(ValueError, match="EOS token is absent"):
        AutoTokenizer.from_pretrained(tmp_path)


@pytest.mark.parametrize("generation", [{"eos_token_id": 4}, {"eos_token_id": [4]}])
def test_generation_eos_overrides_tokenizer_eos(tmp_path, generation):
    _write_tokenizer(tmp_path, generation=generation)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    assert tokenizer.eos_token_id == 3
    assert tokenizer.stop_ids == [4]


def test_generation_eos_without_named_tokenizer_fields(tmp_path):
    _write_tokenizer(tmp_path, generation={"eos_token_id": [4]})
    (tmp_path / "tokenizer_config.json").write_text("{}")
    assert AutoTokenizer.from_pretrained(tmp_path).stop_ids == [4]


def test_native_stop_ids_override_copied_generation_config(tmp_path):
    _write_tokenizer(tmp_path, generation={"eos_token_id": [3]})
    path = tmp_path / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config["stop_token_ids"] = [4]
    path.write_text(json.dumps(config))
    assert AutoTokenizer.from_pretrained(tmp_path).stop_ids == [4]
