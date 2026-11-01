# Hugging Face file conventions

Native AstrAI directories use their saved configuration and weights directly.
Importing external HF weights additionally requires a compatible model-owned mapping.
Tokenizer and generation files follow the separate conventions below; these
file names alone do not imply Transformers model-loader compatibility.

## External model config mapping

Place `hf_mapping.json` beside `config.json` in each model directory. AstrAI
reads this file only when loading that directory; it does not select mappings
by model family or ship built-in model specs. Each model owns its field mapping.

A version 1 mapping is a JSON object:

| Key | Meaning |
| --- | --- |
| `version` | Required; currently `1` |
| `source` | Optional dotted path to a nested text config |
| `required` | Source paths that must exist |
| `fields` | AstrAI target path to source path |
| `defaults` | Target path to fallback source path or `{"constant": value}` |
| `constants` | Target path to literal JSON value |
| `operations` | Generic derived values, applied after the other sections |
| `weights` | Optional tensor conversion settings for this model directory |

For example, a directory can contain this `hf_mapping.json`:

```json
{
  "version": 1,
  "required": ["hidden_size", "num_attention_heads"],
  "fields": {
    "vocab_size": "vocab_size",
    "hidden_size": "hidden_size",
    "num_hidden_layers": "num_hidden_layers",
    "intermediate_size": "intermediate_size",
    "rms_norm_eps": "rms_norm_eps",
    "tie_word_embeddings": "tie_word_embeddings",
    "max_position_embeddings": "max_position_embeddings",
    "rope_theta": "rope_theta",
    "attention.num_heads": "num_attention_heads",
    "attention.num_kv_heads": "num_key_value_heads"
  },
  "defaults": {
    "attention.num_kv_heads": "num_attention_heads"
  }
}
```

Paths use dot notation. Missing optional `fields` are skipped. Missing
`required` fields and missing `defaults` sources raise a clear error.
`constants` always overwrite a mapped value. The result is validated as
`AutoRegressiveLMConfig`; attention belongs under `attention`.

`operations` supports two generic forms:

```json
[
  {
    "op": "map_values",
    "source": "layer_types",
    "target": "attention.layers",
    "values": {"full_attention": "gqa", "linear_attention": "gdn"}
  },
  {
    "op": "multiply",
    "sources": ["head_dim", "rope_parameters.partial_rotary_factor"],
    "target": "attention.gqa.rotary_dim"
  }
]
```

An operand in `multiply` can also be `{"constant": 2}`. These operations
contain no Python expressions. A model directory can use only the fields
it needs. The same model-owned file can declare tensor conversion details:

```json
{
  "weights": {
    "norm_weight_offset": 1,
    "skip_prefixes": ["model.visual.", "mtp."]
  }
}
```

`norm_weight_offset` adds a number to imported decoder, final, Q and K
RMSNorm weights. Set it to `1` for checkpoints whose normalization uses
`1 + weight`. Gated DeltaNet's gated output norm is not shifted.
`skip_prefixes` lists checkpoint keys outside the text decoder to omit
without warnings. A doubled per-head Q projection is split into the Q
projection and its output gate when `attention.output_gate` is true.
Gated DeltaNet's fused Q/K/V projection is split by the head dimensions
in `attention.gdn`. Unsupported tensor layouts still fail strict loading.

For a hybrid text model, map `layer_types` to `attention.layers`, place
the GQA and GDN dimensions under `attention.gqa` and `attention.gdn`,
and use `weights_format="hf"` to require the mapping during load.
The text decoder skips any visual or MTP weights only when their prefixes
are explicitly listed. Image and video inputs require a separate vision
implementation. Paged inference for GDN layers also requires recurrent
cache integration; the dense training forward supports autoregressive
text generation by recomputing the prompt.

## Tokenizer and generation files

AstrAI reads HF tokenizer files: `tokenizer.json`, `tokenizer_config.json`,
optional `special_tokens_map.json`, and `chat_template.jinja`. Named tokens
use `bos_token`, `eos_token`, `pad_token`, and `additional_special_tokens`;
their IDs come from the vocabulary. Private `special_tokens` and
`stop_token_ids` fields are rejected and must be migrated before loading.
Saving exports the chat template as `chat_template.jinja`. When both inline and
standalone templates are present, the standalone file takes precedence.

Generation defaults belong to the model's `generation_config.json`. If it
is absent, HF generation fields in `config.json` are used. The tokenizer's
EOS declaration does not implicitly set the generation stop conditions.
Per-call keyword arguments override the supplied `generation_config`, which
overrides the model defaults. Both a dictionary and a Transformers
`GenerationConfig` instance are accepted. Omitting `eos_token_id` inherits
the default; explicitly passing `None` or `[]` disables EOS stopping.
Length limits still apply. `max_new_tokens` takes precedence over `max_length`;
the latter includes the prompt. With neither configured, AstrAI allows up to
20 new tokens, bounded by the engine sequence limit.

```python
engine.generate(
    prompt,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.8,
)
```

As in HF, `do_sample=False` selects greedy decoding; setting `temperature`
alone does not enable sampling. AstrAI supports greedy and multinomial
sampling, not every Transformers generation algorithm. Unsupported beam,
repetition-penalty and minimum-length settings raise errors. The
`max_tokens` argument remains the frontend alias for `max_new_tokens`;
passing both is an error.

Training checkpoints export tokenizer and generation files into staging
before manifest creation. This file convention does not convert AstrAI's
native model architecture, weight names or optimizer state into a
Transformers model.
