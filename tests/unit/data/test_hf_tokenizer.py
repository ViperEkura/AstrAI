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


@pytest.mark.parametrize(
    "legacy", [{"special_tokens": {"eos": "<eos>"}}, {"stop_token_ids": [4]}]
)
def test_legacy_fields_require_explicit_migration(tmp_path, legacy):
    _write_tokenizer(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="migration"):
        AutoTokenizer.from_pretrained(tmp_path)


@pytest.mark.parametrize("eos", ["<eos>", None])
def test_named_special_tokens_do_not_all_become_stop_ids(tmp_path, eos):
    _write_tokenizer(tmp_path, eos=eos)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    expected = [] if eos is None else [3]
    assert tokenizer.stop_ids == expected
    tokenizer.stop_ids.append(5)
    assert tokenizer.stop_ids == expected
    target = tmp_path / "saved"
    tokenizer.save_pretrained(target)
    assert AutoTokenizer.from_pretrained(target).stop_ids == expected


@pytest.mark.parametrize("ids", [None, [], [4, 4]])
def test_generation_eos_override_and_disable_survive_save(tmp_path, ids):
    _write_tokenizer(tmp_path, generation={"eos_token_id": ids})
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    expected = [] if ids is None else list(dict.fromkeys(ids))
    assert tokenizer.stop_ids == expected
    target = tmp_path / "saved"
    tokenizer.save_pretrained(target)
    assert AutoTokenizer.from_pretrained(target).stop_ids == expected


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
    assert tokenizer.stop_ids == fresh.stop_ids == []
    assert tokenizer.eos_token_id is None
    assert tokenizer.bos_token_id is None
    assert tokenizer.pad_token_id == fresh.pad_token_id
    with pytest.raises(RuntimeError, match="Chat template not set"):
        tokenizer.apply_chat_template([{"role": "user", "content": "hello"}])


@pytest.mark.parametrize(
    "bad_file", ["tokenizer_config.json", "generation_config.json"]
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
    assert tokenizer.stop_ids == [3]
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}], tokenize=False
        )
        == "<bos>hello<eos>"
    )


def test_invalid_stop_ids_leave_previous_vocabulary_and_template_unchanged(tmp_path):
    _write_tokenizer(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    backend = tokenizer._tokenizer
    _write_tokenizer(tmp_path, eos="<eot>", generation={"eos_token_id": [9999]})
    with pytest.raises(ValueError, match="EOS"):
        tokenizer.load(tmp_path)
    assert tokenizer._tokenizer is backend
    assert tokenizer.eos_token_id == 3
    assert tokenizer.stop_ids == [3]


def test_constructor_defaults_fill_absent_fields_without_leaking_to_reload(tmp_path):
    _write_tokenizer(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("{}")
    defaults = {"bos_token": "<bos>", "eos_token": "<eos>"}
    tokenizer = AutoTokenizer(
        tmp_path,
        special_token_map=defaults,
        chat_template="{{bos_token}}custom{{eos_token}}",
    )
    assert tokenizer.stop_ids == [3]
    assert tokenizer.apply_chat_template([], tokenize=False) == "<bos>custom<eos>"
    assert defaults == {"bos_token": "<bos>", "eos_token": "<eos>"}
    tokenizer.load(tmp_path)
    assert tokenizer.stop_ids == []
    assert tokenizer.eos_token_id is None


def test_hf_sidecar_tokens_and_template_are_loaded_and_copied(tmp_path):
    _write_tokenizer(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps({"eos_token": "<eot>", "pad_token": "<pad>"})
    )
    (tmp_path / "chat_template.jinja").write_text("sidecar {{eos_token}}")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    assert tokenizer.stop_ids == [4]
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
    assert restored.stop_ids == []
    assert restored.eos_token_id is None
    with pytest.raises(RuntimeError, match="Chat template not set"):
        restored.apply_chat_template([])


def test_export_is_readable_by_hf_tokenizer_and_generation_config(tmp_path):
    hf = pytest.importorskip("transformers", minversion="4.51.3")
    _write_tokenizer(
        tmp_path,
        generation={"eos_token_id": [3, 4], "do_sample": True, "temperature": 0.7},
    )
    path = tmp_path / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config.update(
        model_max_length=777, padding_side="left", clean_up_tokenization_spaces=False
    )
    path.write_text(json.dumps(config))
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    target = tmp_path / "export"
    tokenizer.save_pretrained(target)
    config = json.loads((target / "tokenizer_config.json").read_text())
    assert "special_tokens" not in config and "stop_token_ids" not in config
    reference = hf.AutoTokenizer.from_pretrained(target, local_files_only=True)
    generation = hf.GenerationConfig.from_pretrained(target, local_files_only=True)
    assert (reference.bos_token_id, reference.eos_token_id, reference.pad_token_id) == (
        2,
        3,
        1,
    )
    assert reference.encode("hello", add_special_tokens=False) == tokenizer.encode(
        "hello", add_special_tokens=False
    )
    messages = [{"role": "user", "content": "hello"}]
    assert reference.apply_chat_template(
        messages, tokenize=False
    ) == tokenizer.apply_chat_template(messages, tokenize=False)
    assert reference.model_max_length == 777 and reference.padding_side == "left"
    assert generation.eos_token_id == [3, 4]
    assert generation.temperature == 0.7 and generation.do_sample
    hf_target = tmp_path / "hf_saved"
    reference.save_pretrained(hf_target)
    generation.save_pretrained(hf_target)
    assert AutoTokenizer.from_pretrained(hf_target).stop_ids == [3, 4]
