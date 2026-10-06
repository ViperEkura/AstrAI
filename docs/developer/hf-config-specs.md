# Hugging Face config mapping

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
