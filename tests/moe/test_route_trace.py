"""Tests for the version-zero semantic MoE route trace contract."""

import json
from dataclasses import replace

import pytest
import safetensors.torch as safetensors
import torch

from astrai.moe import (
    ROUTE_TRACE_SCHEMA_VERSION,
    PaddingLayout,
    RouteIdentityV0,
    RouterSchemaV0,
    RouteTokenLayoutV0,
    RouteTraceCodecV0,
    RouteTraceLevel,
    RouteTraceV0,
    RouteTraceValidationError,
    SelectedWeightSemantics,
    TokenSpanKind,
    canonical_json_digest,
    pack_route_ids,
    recommended_route_id_dtype,
    require_compatible_route_trace,
    validate_route_trace,
)

_MANIFEST_TENSOR = "__route_trace_manifest_v0__"


def _schema(
    *,
    num_experts=8,
    padding_layout=PaddingLayout.RIGHT,
    backend="astrai-torch",
):
    return RouterSchemaV0(
        num_moe_layers=2,
        num_experts=num_experts,
        top_k=2,
        expert_parallel_world_size=1,
        score_function="softmax-fp32-before-topk",
        score_dtype="float32",
        expert_id_ordering="score-descending-stable-index",
        selected_weight_semantics=SelectedWeightSemantics.SELECTED_RENORMALIZED,
        token_ordering="single-sequence-token-major",
        padding_layout=padding_layout,
        backend=backend,
        kernel_semantics_version="torch-topk-v1",
        parallel_layout_hash=canonical_json_digest(
            {"expert_parallel_world_size": 1, "placement": "all-experts-local"}
        ),
    )


def _trace(*, num_experts=8, level=RouteTraceLevel.IDS_WEIGHTS_MARGIN):
    schema = _schema(num_experts=num_experts)
    identity = RouteIdentityV0(
        policy_version=7,
        model_revision="model-revision-7",
        router_state_version=11,
        checkpoint_revision="checkpoint-7",
        router_schema_hash=schema.fingerprint,
        load_controller_version=3,
    )
    token_layout = RouteTokenLayoutV0(
        sample_id="sample-17",
        sequence_token_count=6,
        prompt_token_count=2,
        token_offset=2,
        token_count=4,
        span_kind=TokenSpanKind.RESPONSE,
    )
    topk_ids = pack_route_ids(
        torch.tensor(
            [
                [[0, 1], [2, 3]],
                [[1, 2], [3, 4]],
                [[2, 3], [4, 5]],
                [[0, 0], [0, 0]],
            ]
        ),
        num_experts,
    )
    valid_mask = torch.tensor([True, True, True, False])
    if level is RouteTraceLevel.IDS:
        return RouteTraceV0(
            identity=identity,
            router_schema=schema,
            token_layout=token_layout,
            level=level,
            topk_ids=topk_ids,
            valid_mask=valid_mask,
        )
    selected_weights = torch.tensor(
        [
            [[0.75, 0.25], [0.60, 0.40]],
            [[0.55, 0.45], [0.80, 0.20]],
            [[0.50, 0.50], [0.65, 0.35]],
            [[0.00, 0.00], [0.00, 0.00]],
        ],
        dtype=torch.float16,
    )
    topk_margin = torch.tensor(
        [[0.20, 0.10], [0.01, 0.30], [0.05, 0.25], [0.00, 0.00]],
        dtype=torch.float16,
    )
    return RouteTraceV0(
        identity=identity,
        router_schema=schema,
        token_layout=token_layout,
        level=level,
        topk_ids=topk_ids,
        selected_weights=selected_weights,
        topk_margin=topk_margin,
        valid_mask=valid_mask,
    )


def _rewrite_manifest(blob, mutate):
    tensors = safetensors.load(blob)
    manifest = json.loads(bytes(tensors[_MANIFEST_TENSOR].tolist()))
    mutate(manifest, tensors)
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    tensors[_MANIFEST_TENSOR] = torch.tensor(list(manifest_bytes), dtype=torch.uint8)
    return safetensors.save(tensors)


@pytest.mark.parametrize(
    ("num_experts", "expected_dtype"),
    [
        (8, torch.uint8),
        (256, torch.uint8),
        (257, torch.uint16),
        (65536, torch.uint16),
        (65537, torch.uint32),
    ],
)
def test_route_id_dtype_is_narrow_and_codec_round_trips(num_experts, expected_dtype):
    trace = _trace(num_experts=num_experts)

    blob = RouteTraceCodecV0.dumps(trace)
    restored = RouteTraceCodecV0.loads(blob)

    assert restored.topk_ids.dtype is expected_dtype
    assert restored.identity == trace.identity
    assert restored.router_schema == trace.router_schema
    assert restored.token_layout == trace.token_layout
    assert restored.level is RouteTraceLevel.IDS_WEIGHTS_MARGIN
    assert restored.valid_token_count == 3
    assert (
        restored.payload_nbytes
        == 52 + 16 * torch.empty((), dtype=expected_dtype).element_size()
    )
    assert torch.equal(restored.topk_ids, trace.topk_ids.cpu())
    assert torch.equal(restored.selected_weights, trace.selected_weights.cpu())
    assert torch.equal(restored.topk_margin, trace.topk_margin.cpu())
    assert torch.equal(restored.valid_mask, trace.valid_mask.cpu())
    assert RouteTraceCodecV0.dumps(restored) == blob
    assert restored.artifact_digest == trace.artifact_digest


def test_ids_only_level_round_trips_without_fabricating_optional_metrics():
    trace = _trace(level=RouteTraceLevel.IDS)

    restored = RouteTraceCodecV0.loads(RouteTraceCodecV0.dumps(trace))

    assert restored.selected_weights is None
    assert restored.topk_margin is None
    assert restored.valid_mask is not None


@pytest.mark.parametrize("num_experts", [0, -1, 2**32 + 1, True])
def test_recommended_dtype_rejects_invalid_expert_counts(num_experts):
    with pytest.raises(RouteTraceValidationError, match="num_experts"):
        recommended_route_id_dtype(num_experts)


def test_pack_route_ids_rejects_negative_and_out_of_range_values():
    with pytest.raises(RouteTraceValidationError, match="outside the declared range"):
        pack_route_ids(torch.tensor([[[-1, 0]]]), 8)
    with pytest.raises(RouteTraceValidationError, match="outside the declared range"):
        pack_route_ids(torch.tensor([[[0, 8]]]), 8)
    with pytest.raises(RouteTraceValidationError, match="integer dtype"):
        pack_route_ids(torch.tensor([[[0.0, 1.0]]]), 8)


@pytest.mark.parametrize("policy_version", [-1, True, 1.5])
def test_identity_rejects_invalid_policy_version(policy_version):
    schema = _schema()
    with pytest.raises(RouteTraceValidationError, match="policy_version"):
        RouteIdentityV0(
            policy_version=policy_version,
            model_revision="model",
            router_state_version=0,
            checkpoint_revision="checkpoint",
            router_schema_hash=schema.fingerprint,
        )


def test_schema_hash_mismatch_fails_closed():
    trace = _trace()
    bad_identity = replace(trace.identity, router_schema_hash="0" * 64)

    with pytest.raises(RouteTraceValidationError, match="router_schema_hash"):
        replace(trace, identity=bad_identity)


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        (
            RouteTokenLayoutV0(
                sample_id="sample",
                sequence_token_count=8,
                prompt_token_count=3,
                token_offset=3,
                token_count=2,
                span_kind=TokenSpanKind.RESPONSE,
            ),
            None,
        ),
    ],
)
def test_valid_token_layout(layout, message):
    assert layout.token_offset == 3
    assert message is None


def test_invalid_token_boundaries_fail_closed():
    with pytest.raises(RouteTraceValidationError, match="exceeds"):
        RouteTokenLayoutV0(
            sample_id="sample",
            sequence_token_count=4,
            prompt_token_count=2,
            token_offset=3,
            token_count=2,
            span_kind=TokenSpanKind.CONTIGUOUS,
        )
    with pytest.raises(RouteTraceValidationError, match="inside the prompt"):
        RouteTokenLayoutV0(
            sample_id="sample",
            sequence_token_count=4,
            prompt_token_count=2,
            token_offset=1,
            token_count=2,
            span_kind=TokenSpanKind.RESPONSE,
        )
    with pytest.raises(RouteTraceValidationError, match="exactly cover"):
        RouteTokenLayoutV0(
            sample_id="sample",
            sequence_token_count=4,
            prompt_token_count=2,
            token_offset=0,
            token_count=1,
            span_kind=TokenSpanKind.PROMPT,
        )


def test_tensor_shape_dtype_range_and_uniqueness_are_strict():
    trace = _trace()
    with pytest.raises(RouteTraceValidationError, match="shape"):
        replace(trace, topk_ids=trace.topk_ids[:3])
    with pytest.raises(RouteTraceValidationError, match="compact dtype"):
        replace(trace, topk_ids=trace.topk_ids.to(torch.int64))
    out_of_range = trace.topk_ids.clone()
    out_of_range[0, 0, 0] = trace.router_schema.num_experts
    with pytest.raises(RouteTraceValidationError, match="outside the declared range"):
        replace(trace, topk_ids=out_of_range)
    duplicate = trace.topk_ids.clone()
    duplicate[0, 0] = duplicate[0, 0, 0]
    with pytest.raises(RouteTraceValidationError, match="duplicate experts"):
        replace(trace, topk_ids=duplicate)
    with pytest.raises(RouteTraceValidationError, match="materialized strided"):
        replace(
            trace,
            topk_ids=torch.empty(
                trace.topk_ids.shape, dtype=trace.topk_ids.dtype, device="meta"
            ),
        )


def test_padding_mask_and_invalid_rows_must_be_canonical():
    trace = _trace()
    with pytest.raises(RouteTraceValidationError, match="right padding"):
        replace(trace, valid_mask=torch.tensor([True, False, True, False]))
    invalid_ids = trace.topk_ids.clone()
    invalid_ids[-1, 0, 0] = 1
    with pytest.raises(RouteTraceValidationError, match="canonical zero expert"):
        replace(trace, topk_ids=invalid_ids)
    invalid_weights = trace.selected_weights.clone()
    invalid_weights[-1, 0, 0] = 1
    with pytest.raises(RouteTraceValidationError, match="canonical zero weights"):
        replace(trace, selected_weights=invalid_weights)


def test_explicit_padding_requires_a_mask():
    trace = _trace()
    schema = replace(trace.router_schema, padding_layout=PaddingLayout.EXPLICIT_MASK)
    identity = replace(trace.identity, router_schema_hash=schema.fingerprint)
    with pytest.raises(RouteTraceValidationError, match="requires valid_mask"):
        replace(trace, router_schema=schema, identity=identity, valid_mask=None)


def test_weight_and_margin_contract_rejects_invalid_values():
    trace = _trace()
    wrong_sum = trace.selected_weights.clone()
    wrong_sum[0, 0] = torch.tensor([0.2, 0.2])
    with pytest.raises(RouteTraceValidationError, match="sum to one"):
        replace(trace, selected_weights=wrong_sum)
    nan_margin = trace.topk_margin.clone()
    nan_margin[0, 0] = float("nan")
    with pytest.raises(RouteTraceValidationError, match="finite"):
        replace(trace, topk_margin=nan_margin)
    with pytest.raises(RouteTraceValidationError, match="cannot carry"):
        replace(trace, level=RouteTraceLevel.IDS)
    with pytest.raises(RouteTraceValidationError, match="requires weights"):
        replace(trace, selected_weights=None, topk_margin=None)
    with pytest.raises(RouteTraceValidationError, match="torch.Tensor"):
        replace(trace, selected_weights="not-a-tensor")
    negative_zero = trace.topk_margin.clone()
    negative_zero[0, 0] = -0.0
    with pytest.raises(RouteTraceValidationError, match="canonical positive zero"):
        replace(trace, topk_margin=negative_zero)


def test_full_softmax_selected_mass_allows_subunit_mass_only():
    trace = _trace()
    schema = replace(
        trace.router_schema,
        selected_weight_semantics=SelectedWeightSemantics.FULL_SOFTMAX_SELECTED,
    )
    identity = replace(trace.identity, router_schema_hash=schema.fingerprint)
    weights = trace.selected_weights * 0.5
    valid = replace(
        trace, router_schema=schema, identity=identity, selected_weights=weights
    )
    validate_route_trace(valid)
    too_large = weights.clone()
    too_large[0, 0] = torch.tensor([0.8, 0.4])
    with pytest.raises(RouteTraceValidationError, match="selected mass"):
        replace(valid, selected_weights=too_large)


def test_exact_consumer_compatibility_rejects_every_identity_or_semantic_mismatch():
    trace = _trace()
    require_compatible_route_trace(
        trace,
        expected_identity=trace.identity,
        expected_router_schema=trace.router_schema,
        expected_token_layout=trace.token_layout,
        minimum_level=RouteTraceLevel.IDS_WEIGHTS_MARGIN,
    )
    with pytest.raises(RouteTraceValidationError, match="route identity"):
        require_compatible_route_trace(
            trace,
            expected_identity=replace(trace.identity, policy_version=8),
            expected_router_schema=trace.router_schema,
            expected_token_layout=trace.token_layout,
        )
    with pytest.raises(RouteTraceValidationError, match="router schema"):
        require_compatible_route_trace(
            trace,
            expected_identity=trace.identity,
            expected_router_schema=replace(
                trace.router_schema, backend="other-backend"
            ),
            expected_token_layout=trace.token_layout,
        )
    with pytest.raises(RouteTraceValidationError, match="token layout"):
        require_compatible_route_trace(
            trace,
            expected_identity=trace.identity,
            expected_router_schema=trace.router_schema,
            expected_token_layout=replace(trace.token_layout, sample_id="other-sample"),
        )
    ids_only = _trace(level=RouteTraceLevel.IDS)
    with pytest.raises(RouteTraceValidationError, match="does not satisfy"):
        require_compatible_route_trace(
            ids_only,
            expected_identity=ids_only.identity,
            expected_router_schema=ids_only.router_schema,
            expected_token_layout=ids_only.token_layout,
            minimum_level=RouteTraceLevel.IDS_WEIGHTS_MARGIN,
        )


def test_codec_rejects_unknown_schema_fields_versions_and_tensors():
    blob = RouteTraceCodecV0.dumps(_trace())
    unknown_field = _rewrite_manifest(
        blob, lambda manifest, _: manifest.update({"future": 1})
    )
    with pytest.raises(RouteTraceValidationError, match="manifest keys"):
        RouteTraceCodecV0.loads(unknown_field)
    future_version = _rewrite_manifest(
        blob,
        lambda manifest, _: manifest.update(
            {"route_trace_schema_version": ROUTE_TRACE_SCHEMA_VERSION + 1}
        ),
    )
    with pytest.raises(
        RouteTraceValidationError, match="unsupported route trace schema version"
    ):
        RouteTraceCodecV0.loads(future_version)
    boolean_version = _rewrite_manifest(
        blob,
        lambda manifest, _: manifest.update({"route_trace_schema_version": False}),
    )
    with pytest.raises(
        RouteTraceValidationError, match="unsupported route trace schema version"
    ):
        RouteTraceCodecV0.loads(boolean_version)
    wrong_byte_order = _rewrite_manifest(
        blob, lambda manifest, _: manifest.update({"byte_order": "big"})
    )
    with pytest.raises(
        RouteTraceValidationError, match="unsupported route trace byte order"
    ):
        RouteTraceCodecV0.loads(wrong_byte_order)

    def add_tensor(_, tensors):
        tensors["tensor.future"] = torch.zeros(1, dtype=torch.uint8)

    extra_tensor = _rewrite_manifest(blob, add_tensor)
    with pytest.raises(RouteTraceValidationError, match="tensor names"):
        RouteTraceCodecV0.loads(extra_tensor)


def test_codec_rejects_corruption_and_size_budget_violation():
    blob = RouteTraceCodecV0.dumps(_trace())

    def corrupt_ids(_, tensors):
        tensors["tensor.topk_ids"][0] ^= 1

    corrupted = _rewrite_manifest(blob, corrupt_ids)
    with pytest.raises(RouteTraceValidationError, match="checksum mismatch"):
        RouteTraceCodecV0.loads(corrupted)
    with pytest.raises(RouteTraceValidationError, match="exceeds the size limit"):
        RouteTraceCodecV0.loads(blob, max_serialized_bytes=len(blob) - 1)
    with pytest.raises(RouteTraceValidationError, match="safetensors"):
        RouteTraceCodecV0.loads(b"not-a-safetensors-file")


def test_manifest_subobjects_are_strictly_versioned():
    blob = RouteTraceCodecV0.dumps(_trace())
    unknown_identity = _rewrite_manifest(
        blob, lambda manifest, _: manifest["identity"].update({"future": 1})
    )
    with pytest.raises(RouteTraceValidationError, match="identity keys"):
        RouteTraceCodecV0.loads(unknown_identity)
