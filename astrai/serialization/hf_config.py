"""Import an HF config using a declarative mapping beside the checkpoint."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HF_MAPPING_FILENAME = "hf_mapping.json"
_MISSING = object()


def load_hf_mapping(model_dir: str | Path) -> dict[str, Any] | None:
    """Read the model-owned mapping, if present."""
    path = Path(model_dir) / HF_MAPPING_FILENAME
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if mapping.get("version") != 1:
        raise ValueError(f"{path} requires version 1")
    allowed = {
        "version",
        "source",
        "required",
        "fields",
        "constants",
        "defaults",
        "operations",
        "weights",
    }
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{path} has unknown mapping keys: {sorted(unknown)}")
    weights = mapping.get("weights", {})
    if not isinstance(weights, dict):
        raise ValueError(f"{path} weights must be an object")
    if set(weights) - {"norm_weight_offset", "skip_prefixes"}:
        raise ValueError(f"{path} has unknown weight options")
    offset = weights.get("norm_weight_offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, (int, float)):
        raise ValueError(f"{path} norm_weight_offset must be numeric")
    prefixes = weights.get("skip_prefixes", [])
    if not isinstance(prefixes, list) or any(
        not isinstance(prefix, str) or not prefix for prefix in prefixes
    ):
        raise ValueError(f"{path} skip_prefixes must be a list of nonempty strings")
    return mapping


def _get_path(raw: Mapping[str, Any], path: str) -> Any:
    value: Any = raw
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _set_path(result: dict, path: str, value: Any) -> None:
    if not path or any(not part for part in path.split(".")):
        raise ValueError(f"invalid target path: {path!r}")
    target = result
    parts = path.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f"mapping target conflicts at {path!r}")
    target[parts[-1]] = value


def _resolve_operand(source: Mapping[str, Any], operand: Any) -> Any:
    if isinstance(operand, str):
        value = _get_path(source, operand)
        if value is _MISSING:
            raise ValueError(f"HF config is missing {operand}")
        return value
    if isinstance(operand, dict) and set(operand) == {"constant"}:
        return operand["constant"]
    raise ValueError("operation operands must be source paths or constant objects")


def _apply_operation(source: Mapping[str, Any], result: dict, operation: dict) -> None:
    if not isinstance(operation, dict):
        raise ValueError("each mapping operation must be an object")
    name = operation.get("op")
    target = operation.get("target")
    if not isinstance(target, str):
        raise ValueError("mapping operation requires a target path")
    if name == "multiply":
        values = operation.get("sources")
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError("multiply requires at least two sources")
        value = 1
        for operand in values:
            item = _resolve_operand(source, operand)
            if not isinstance(item, (int, float)) or isinstance(item, bool):
                raise ValueError("multiply sources must be numbers")
            value *= item
        if isinstance(value, float) and value.is_integer():
            value = int(value)
    elif name == "map_values":
        values = operation.get("values")
        if not isinstance(values, dict):
            raise ValueError("map_values requires a values object")
        original = _resolve_operand(source, operation.get("source"))

        def mapped(item):
            if item not in values:
                raise ValueError(f"unmapped value {item!r} for {target}")
            return values[item]

        value = (
            [mapped(item) for item in original]
            if isinstance(original, list)
            else mapped(original)
        )
    else:
        raise ValueError(f"unknown HF mapping operation: {name!r}")
    _set_path(result, target, value)


def convert_hf_config(
    raw: dict[str, Any], mapping: Mapping[str, Any]
) -> dict[str, Any]:
    """Convert a config with an explicit model-directory mapping."""
    if not isinstance(mapping, Mapping) or mapping.get("version") != 1:
        raise ValueError("HF mapping must be a version 1 object")
    source = raw
    if "source" in mapping:
        source = _get_path(raw, mapping["source"])
        if not isinstance(source, Mapping):
            raise ValueError(f"HF config requires {mapping['source']}")
    required = mapping.get("required", [])
    if not isinstance(required, list):
        raise ValueError("HF mapping required must be a list")
    for path in required:
        if _get_path(source, path) is _MISSING:
            raise ValueError(f"HF config is missing {path}")
    if source.get("attention_bias") or source.get("mlp_bias"):
        raise NotImplementedError(
            "attention_bias / mlp_bias checkpoints are not supported"
        )
    result: dict[str, Any] = {
        "model_type": "autoregressive_lm",
        "source_model_type": source.get("model_type", raw.get("model_type")),
    }
    for section in ("fields", "defaults", "constants"):
        entries = mapping.get(section, {})
        if not isinstance(entries, Mapping):
            raise ValueError(f"HF mapping {section} must be an object")
        for target, origin in entries.items():
            if section == "fields":
                if not isinstance(origin, str):
                    raise ValueError(f"HF mapping source for {target} must be a path")
                value = _get_path(source, origin)
            elif section == "defaults":
                existing = _get_path(result, target)
                if existing is not _MISSING and existing is not None:
                    continue
                value = _resolve_operand(source, origin)
            else:
                value = origin
            if value is not _MISSING:
                _set_path(result, target, value)
    operations = mapping.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("HF mapping operations must be a list")
    for operation in operations:
        _apply_operation(source, result, operation)
    return result
