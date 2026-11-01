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
                "unk_token": {
                    "content": "<unk>",
                    "special": True,
                    "__type": "AddedToken",
                },
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
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}], tokenize=False
        )
        == "<bos>hello<eos>"
    )


@pytest.mark.parametrize(
    "legacy", [{"special_tokens": {"eos": "<eos>"}}, {"stop_token_ids": [4]}]
)
def test_legacy_fields_require_explicit_migration(tmp_path, legacy):
    _write_tokenizer(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="migration"):
        AutoTokenizer.from_pretrained(tmp_path)


@pytest.mark.parametrize(
    "config", [None, {}, {"pad_token": "<pad>"}, {"eos_token": None}]
)
def test_reload_replaces_eos_and_template_instead_of_retaining_old_state(
    tmp_path, config
):
    _write_tokenizer(tmp_path, generation={"eos_token_id": [3, 4]})
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _write_tokenizer(other)
    path = other / "tokenizer_config.json"
    if config is None:
        path.unlink()
    else:
        path.write_text(json.dumps(config))
    tokenizer.load(other)
    fresh = AutoTokenizer.from_pretrained(other)
    assert tokenizer.eos_token_id is None
    assert tokenizer.bos_token_id is None
    assert tokenizer.pad_token_id == fresh.pad_token_id
    with pytest.raises(RuntimeError, match="Chat template not set"):
        tokenizer.apply_chat_template([{"role": "user", "content": "hello"}])


@pytest.mark.parametrize(
    "bad_file", ["tokenizer_config.json", "special_tokens_map.json"]
)
def test_failed_reload_preserves_previous_tokenizer_state(tmp_path, bad_file):
    _write_tokenizer(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    other = tmp_path / "bad"
    other.mkdir()
    _write_tokenizer(other, eos="<eot>")
    (other / bad_file).write_text("{broken")
    with pytest.raises(ValueError):
        tokenizer.load(other)
    assert tokenizer.eos_token_id == 3
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}], tokenize=False
        )
        == "<bos>hello<eos>"
    )


def test_constructor_defaults_fill_absent_fields_without_leaking_to_reload(tmp_path):
    _write_tokenizer(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("{}")
    defaults = {"bos_token": "<bos>", "eos_token": "<eos>"}
    tokenizer = AutoTokenizer(
        tmp_path,
        special_token_map=defaults,
        chat_template="{{bos_token}}custom{{eos_token}}",
    )
    assert tokenizer.apply_chat_template([], tokenize=False) == "<bos>custom<eos>"
    assert defaults == {"bos_token": "<bos>", "eos_token": "<eos>"}
    tokenizer.load(tmp_path)
    assert tokenizer.eos_token_id is None


def test_hf_sidecar_tokens_and_template_are_loaded_and_copied(tmp_path):
    _write_tokenizer(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps({"eos_token": "<eot>", "pad_token": "<pad>"})
    )
    (tmp_path / "chat_template.jinja").write_text("sidecar {{eos_token}}")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    assert tokenizer.apply_chat_template([], tokenize=False) == "sidecar <eot>"
    target = tmp_path / "checkpoint"
    target.mkdir()
    _copy_tokenizer_files(str(tmp_path), str(target))
    assert (
        AutoTokenizer.from_pretrained(target).apply_chat_template([], tokenize=False)
        == "sidecar <eot>"
    )


def test_saving_over_directory_clears_stale_template_and_token_sidecars(tmp_path):
    _write_tokenizer(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("{}")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    target = tmp_path / "saved"
    target.mkdir()
    (target / "chat_template.jinja").write_text("stale")
    (target / "special_tokens_map.json").write_text(json.dumps({"eos_token": "<eos>"}))
    tokenizer.save_pretrained(target)
    restored = AutoTokenizer.from_pretrained(target)
    assert restored.eos_token_id is None
    with pytest.raises(RuntimeError, match="Chat template not set"):
        restored.apply_chat_template([])


def test_tokenizer_does_not_read_or_write_generation_config(tmp_path):
    _write_tokenizer(tmp_path)
    (tmp_path / "generation_config.json").write_text("{broken")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    assert tokenizer.eos_token_id == 3
    with pytest.raises(AttributeError):
        _ = tokenizer.stop_ids
    target = tmp_path / "export"
    tokenizer.save_pretrained(target)
    assert not (target / "generation_config.json").exists()


def test_added_tokens_and_extra_names_match_hf(tmp_path):
    hf = pytest.importorskip("transformers", minversion="4.51.3")
    _write_tokenizer(tmp_path)
    config_path = tmp_path / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config.update(
        tokenizer_class="PreTrainedTokenizerFast",
        added_tokens_decoder={
            "7": {
                "content": "<audio>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        },
        extra_special_tokens={"audio_token": "<audio>"},
        model_max_length=777,
        padding_side="left",
    )
    config_path.write_text(json.dumps(config))
    actual = AutoTokenizer.from_pretrained(tmp_path)
    reference = hf.AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert actual.audio_token_id == reference.audio_token_id == 7
    assert (
        actual.encode("<audio>", add_special_tokens=False)
        == reference.encode("<audio>", add_special_tokens=False)
        == [7]
    )
    target = tmp_path / "export"
    actual.save_pretrained(target)
    restored = hf.AutoTokenizer.from_pretrained(target, local_files_only=True)
    assert restored.audio_token_id == 7 and restored.model_max_length == 777
    assert restored.padding_side == "left"


def test_tool_template_json_matches_hf(tmp_path):
    hf = pytest.importorskip("transformers", minversion="4.51.3")
    _write_tokenizer(tmp_path)
    path = tmp_path / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config.update(
        tokenizer_class="PreTrainedTokenizerFast",
        chat_template="{{ tools | tojson }}{{ messages[0].content }}",
    )
    path.write_text(json.dumps(config))
    actual = AutoTokenizer.from_pretrained(tmp_path)
    reference = hf.AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    messages = [{"role": "user", "content": "查天气"}]
    tools = [
        {"type": "function", "function": {"name": "天气", "description": "<a>&天气"}}
    ]
    assert actual.apply_chat_template(
        messages, tokenize=False, tools=tools
    ) == reference.apply_chat_template(messages, tokenize=False, tools=tools)
