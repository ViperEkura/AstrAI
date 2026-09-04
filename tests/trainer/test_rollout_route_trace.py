"""Tests for immutable RouteTraceV0 bindings at rollout cache boundaries."""

import json
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from astrai.moe import (
    PaddingLayout,
    RouteIdentityV0,
    RouterSchemaV0,
    RouteTokenLayoutV0,
    RouteTraceLevel,
    RouteTraceV0,
    SelectedWeightSemantics,
    TokenSpanKind,
    canonical_json_digest,
    pack_route_ids,
)
from astrai.parallel import get_rank, get_world_size, spawn_parallel_fn
from astrai.trainer.rollout import (
    BaseRewardModel,
    RawRollout,
    RolloutRunner,
)
from astrai.trainer.route_trace import (
    ROLLOUT_ROUTE_TRACE_SCHEMA_VERSION,
    RolloutRouteTraceBatchV0,
    RolloutRouteTraceError,
    RolloutRouteTraceItemV0,
    rollout_route_sample_id,
)
from astrai.trainer.route_trace_transport import (
    ROLLOUT_ROUTE_TRACE_SHARD_SCHEMA_VERSION,
    RolloutRouteTraceShardCodecV0,
    RolloutRouteTraceShardManifestV0,
    RolloutRouteTraceTransportV0,
)


def _rollout_tensors():
    prompts = torch.tensor([[0, 11, 12], [21, 22, 23]], dtype=torch.long)
    prompt_mask = torch.tensor([[False, True, True], [True, True, True]])
    responses = torch.tensor(
        [
            [[31, 32, 0, 0], [33, 34, 35, 0]],
            [[41, 0, 0, 0], [42, 43, 44, 45]],
        ],
        dtype=torch.long,
    )
    response_mask = responses != 0
    return prompts, prompt_mask, responses, response_mask


def _schema():
    return RouterSchemaV0(
        num_moe_layers=2,
        num_experts=8,
        top_k=2,
        expert_parallel_world_size=1,
        score_function="softmax-fp32-before-topk",
        score_dtype="float32",
        expert_id_ordering="score-descending-stable-index",
        selected_weight_semantics=SelectedWeightSemantics.SELECTED_RENORMALIZED,
        token_ordering="rollout-batch-group-response-token-major",
        padding_layout=PaddingLayout.RIGHT,
        backend="unit-test",
        kernel_semantics_version="torch-topk-v1",
        parallel_layout_hash=canonical_json_digest({"placement": "all-local"}),
    )


def _trace_grid(
    *, policy_version=3, level=RouteTraceLevel.IDS, rollout_id="rollout-17"
):
    prompts, prompt_mask, _, response_mask = _rollout_tensors()
    schema = _schema()
    identity = RouteIdentityV0(
        policy_version=policy_version,
        model_revision="model-3",
        router_state_version=3,
        checkpoint_revision="checkpoint-3",
        router_schema_hash=schema.fingerprint,
        load_controller_version=1,
    )
    traces = []
    for batch_index in range(response_mask.shape[0]):
        row = []
        prompt_length = int(prompt_mask[batch_index].sum().item())
        for group_index in range(response_mask.shape[1]):
            mask = response_mask[batch_index, group_index].clone()
            ids = torch.zeros(4, 2, 2, dtype=torch.int64)
            for token_index in range(int(mask.sum().item())):
                for layer_index in range(2):
                    first = (batch_index + group_index + token_index + layer_index) % 8
                    ids[token_index, layer_index] = torch.tensor(
                        [first, (first + 1) % 8]
                    )
            kwargs = {}
            if level is RouteTraceLevel.IDS_WEIGHTS_MARGIN:
                weights = torch.zeros(4, 2, 2, dtype=torch.float16)
                margins = torch.zeros(4, 2, dtype=torch.float16)
                weights[mask] = torch.tensor([0.625, 0.375], dtype=torch.float16)
                margins[mask] = 0.125
                kwargs = {"selected_weights": weights, "topk_margin": margins}
            row.append(
                RouteTraceV0(
                    identity=identity,
                    router_schema=schema,
                    token_layout=RouteTokenLayoutV0(
                        sample_id=rollout_route_sample_id(
                            rollout_id, batch_index, group_index
                        ),
                        sequence_token_count=prompt_length + 4,
                        prompt_token_count=prompt_length,
                        token_offset=prompt_length,
                        token_count=4,
                        span_kind=TokenSpanKind.RESPONSE,
                    ),
                    level=level,
                    topk_ids=pack_route_ids(ids, schema.num_experts),
                    valid_mask=mask,
                    **kwargs,
                )
            )
        traces.append(row)
    return traces


def _raw_rollout(*, route_trace_batch=None):
    prompts, prompt_mask, responses, response_mask = _rollout_tensors()
    return RawRollout(
        prompts=prompts,
        prompt_mask=prompt_mask,
        responses=responses,
        response_mask=response_mask,
        logprobs_old=torch.zeros_like(responses, dtype=torch.float32),
        policy_version=3,
        prompt_texts=["prompt-0", "prompt-1"],
        response_texts=[["a", "b"], ["c", "d"]],
        route_trace_batch=route_trace_batch,
    )


def _bind(raw, traces=None, *, rollout_id="rollout-17"):
    return RolloutRouteTraceBatchV0.bind(
        rollout_id=rollout_id,
        policy_version=raw.policy_version,
        prompts=raw.prompts,
        prompt_mask=raw.prompt_mask,
        responses=raw.responses,
        response_mask=raw.response_mask,
        logprobs_old=raw.logprobs_old,
        traces=(_trace_grid(rollout_id=rollout_id) if traces is None else traces),
    )


def test_binding_is_deterministic_immutable_and_decodes_in_grid_order():
    raw = _raw_rollout()
    traces = _trace_grid(level=RouteTraceLevel.IDS_WEIGHTS_MARGIN)
    bound = _bind(raw, traces)
    repeated = _bind(raw, traces)

    assert bound.schema_version == ROLLOUT_ROUTE_TRACE_SCHEMA_VERSION
    assert bound.batch_size == 2
    assert bound.group_size == 2
    assert bound.policy_version == 3
    assert bound.identity == traces[0][0].identity
    assert bound.router_schema == traces[0][0].router_schema
    assert bound.level is RouteTraceLevel.IDS_WEIGHTS_MARGIN
    assert bound.artifact_digest == repeated.artifact_digest
    assert type(bound.items) is tuple
    assert type(bound.items[0]) is tuple
    assert type(bound.items[0][0].payload) is bytes

    original_id = int(traces[0][0].topk_ids[0, 0, 0].item())
    traces[0][0].topk_ids[0, 0, 0] = 7
    decoded = bound.decode_traces()
    assert int(decoded[0][0].topk_ids[0, 0, 0].item()) == original_id
    assert decoded[1][1].token_layout.sample_id == rollout_route_sample_id(
        "rollout-17", 1, 1
    )


def test_binding_rejects_policy_grid_identity_level_and_sample_mismatches():
    raw = _raw_rollout()
    with pytest.raises(RolloutRouteTraceError, match="policy version"):
        RolloutRouteTraceBatchV0.bind(
            rollout_id="rollout",
            policy_version=4,
            prompts=raw.prompts,
            prompt_mask=raw.prompt_mask,
            responses=raw.responses,
            response_mask=raw.response_mask,
            logprobs_old=raw.logprobs_old,
            traces=_trace_grid(),
        )
    with pytest.raises(RolloutRouteTraceError, match="B/G dimensions"):
        _bind(raw, _trace_grid()[:1])

    mixed_identity = _trace_grid()
    other_identity = replace(mixed_identity[1][1].identity, model_revision="other")
    mixed_identity[1][1] = replace(mixed_identity[1][1], identity=other_identity)
    with pytest.raises(RolloutRouteTraceError, match="mix behavior identities"):
        _bind(raw, mixed_identity)

    mixed_level = _trace_grid()
    mixed_level[1][1] = _trace_grid(level=RouteTraceLevel.IDS_WEIGHTS_MARGIN)[1][1]
    with pytest.raises(RolloutRouteTraceError, match="mix trace levels"):
        _bind(raw, mixed_level)

    duplicate_sample = _trace_grid()
    duplicate_sample[1][1] = replace(
        duplicate_sample[1][1],
        token_layout=replace(
            duplicate_sample[1][1].token_layout,
            sample_id=duplicate_sample[0][0].token_layout.sample_id,
        ),
    )
    with pytest.raises(RolloutRouteTraceError, match="sample IDs must be unique"):
        _bind(raw, duplicate_sample)

    swapped = _trace_grid()
    swapped[0][0], swapped[0][1] = swapped[0][1], swapped[0][0]
    with pytest.raises(RolloutRouteTraceError, match="rollout grid position"):
        _bind(raw, swapped)


def test_validation_rejects_policy_prompt_response_and_mask_mutation():
    raw = _raw_rollout()
    bound = _bind(raw)
    bound.validate_against(
        policy_version=raw.policy_version,
        prompts=raw.prompts,
        prompt_mask=raw.prompt_mask,
        responses=raw.responses,
        response_mask=raw.response_mask,
        logprobs_old=raw.logprobs_old,
    )

    with pytest.raises(RolloutRouteTraceError, match="policy version"):
        bound.validate_against(
            policy_version=4,
            prompts=raw.prompts,
            prompt_mask=raw.prompt_mask,
            responses=raw.responses,
            response_mask=raw.response_mask,
            logprobs_old=raw.logprobs_old,
        )
    prompts = raw.prompts.clone()
    prompts[0, -1] += 1
    with pytest.raises(RolloutRouteTraceError, match="prompt tokens"):
        bound.validate_against(
            policy_version=3,
            prompts=prompts,
            prompt_mask=raw.prompt_mask,
            responses=raw.responses,
            response_mask=raw.response_mask,
            logprobs_old=raw.logprobs_old,
        )
    responses = raw.responses.clone()
    responses[0, 0, 0] += 1
    with pytest.raises(RolloutRouteTraceError, match="response tokens"):
        bound.validate_against(
            policy_version=3,
            prompts=raw.prompts,
            prompt_mask=raw.prompt_mask,
            responses=responses,
            response_mask=raw.response_mask,
            logprobs_old=raw.logprobs_old,
        )
    responses = raw.responses.clone()
    response_mask = raw.response_mask.clone()
    responses[0, 0, 1] = 0
    response_mask[0, 0, 1] = False
    with pytest.raises(RolloutRouteTraceError, match="response tokens"):
        bound.validate_against(
            policy_version=3,
            prompts=raw.prompts,
            prompt_mask=raw.prompt_mask,
            responses=responses,
            response_mask=response_mask,
            logprobs_old=raw.logprobs_old,
        )

    logprobs_old = raw.logprobs_old.clone()
    logprobs_old[0, 0, 0] = -1.0
    with pytest.raises(RolloutRouteTraceError, match="behavior logprobs"):
        bound.validate_against(
            policy_version=3,
            prompts=raw.prompts,
            prompt_mask=raw.prompt_mask,
            responses=raw.responses,
            response_mask=raw.response_mask,
            logprobs_old=logprobs_old,
        )


def test_binding_rejects_trace_boundaries_and_validity_mask_mismatch():
    raw = _raw_rollout()
    bad_boundary = _trace_grid()
    layout = bad_boundary[0][0].token_layout
    bad_boundary[0][0] = replace(
        bad_boundary[0][0],
        token_layout=replace(
            layout,
            sequence_token_count=layout.sequence_token_count + 1,
            token_offset=layout.token_offset + 1,
        ),
    )
    with pytest.raises(RolloutRouteTraceError, match="token boundaries"):
        _bind(raw, bad_boundary)

    bad_mask = _trace_grid()
    trace = bad_mask[0][0]
    mask = trace.valid_mask.clone()
    mask[1] = False
    ids = trace.topk_ids.clone()
    ids[1] = 0
    bad_mask[0][0] = replace(trace, topk_ids=ids, valid_mask=mask)
    with pytest.raises(RolloutRouteTraceError, match="validity mask"):
        _bind(raw, bad_mask)


def test_binding_rejects_noncanonical_rollout_tensor_layout():
    raw = _raw_rollout()
    prompt_mask = raw.prompt_mask.clone()
    prompt_mask[0] = torch.tensor([True, False, True])
    with pytest.raises(RolloutRouteTraceError, match="left-padding"):
        RolloutRouteTraceBatchV0.bind(
            rollout_id="rollout",
            policy_version=3,
            prompts=raw.prompts,
            prompt_mask=prompt_mask,
            responses=raw.responses,
            response_mask=raw.response_mask,
            logprobs_old=raw.logprobs_old,
            traces=_trace_grid(),
        )

    positive_logprob = raw.logprobs_old.clone()
    positive_logprob[0, 0, 0] = 0.1
    with pytest.raises(RolloutRouteTraceError, match="non-positive"):
        RolloutRouteTraceBatchV0.bind(
            rollout_id="rollout-17",
            policy_version=3,
            prompts=raw.prompts,
            prompt_mask=raw.prompt_mask,
            responses=raw.responses,
            response_mask=raw.response_mask,
            logprobs_old=positive_logprob,
            traces=_trace_grid(),
        )

    padded_negative_zero = raw.logprobs_old.clone()
    padded_negative_zero[0, 0, -1] = -0.0
    with pytest.raises(RolloutRouteTraceError, match="canonical positive zero"):
        RolloutRouteTraceBatchV0.bind(
            rollout_id="rollout-17",
            policy_version=3,
            prompts=raw.prompts,
            prompt_mask=raw.prompt_mask,
            responses=raw.responses,
            response_mask=raw.response_mask,
            logprobs_old=padded_negative_zero,
            traces=_trace_grid(),
        )
    responses = raw.responses.clone()
    responses[0, 0, -1] = 99
    with pytest.raises(RolloutRouteTraceError, match="padded response token"):
        RolloutRouteTraceBatchV0.bind(
            rollout_id="rollout",
            policy_version=3,
            prompts=raw.prompts,
            prompt_mask=raw.prompt_mask,
            responses=responses,
            response_mask=raw.response_mask,
            logprobs_old=raw.logprobs_old,
            traces=_trace_grid(),
        )


def test_item_rejects_mutable_corrupt_or_mismatched_payload_headers():
    item = RolloutRouteTraceItemV0.from_trace(_trace_grid()[0][0])
    restored = RolloutRouteTraceItemV0.from_payload(
        item.payload,
        max_serialized_bytes=len(item.payload),
    )
    assert restored == item
    with pytest.raises(RolloutRouteTraceError, match="size limit"):
        RolloutRouteTraceItemV0.from_payload(
            item.payload,
            max_serialized_bytes=len(item.payload) - 1,
        )
    with pytest.raises(RolloutRouteTraceError, match="immutable bytes"):
        replace(item, payload=bytearray(item.payload))
    with pytest.raises(RolloutRouteTraceError, match="payload digest"):
        replace(item, payload=item.payload[:-1] + bytes([item.payload[-1] ^ 1]))
    with pytest.raises(RolloutRouteTraceError, match="identity header"):
        replace(item, identity=replace(item.identity, model_revision="other"))


def test_batch_constructor_rejects_unknown_versions_and_mutable_grids():
    raw = _raw_rollout()
    bound = _bind(raw)
    with pytest.raises(RolloutRouteTraceError, match="unsupported.*schema version"):
        replace(bound, schema_version=1)
    with pytest.raises(RolloutRouteTraceError, match="non-empty tuple grid"):
        replace(bound, items=list(bound.items))
    with pytest.raises(RolloutRouteTraceError, match="non-empty tuple"):
        replace(bound, items=(list(bound.items[0]), bound.items[1]))
    with pytest.raises(RolloutRouteTraceError, match="RouteTraceV0"):
        _bind(raw, [["bad", "bad"], ["bad", "bad"]])


class _Generator:
    def __init__(self, raw):
        self.raw = raw
        self.policy_version = raw.policy_version

    def generate(self, _batch):
        return self.raw

    def with_policy_snapshot(self, inspect):
        return inspect(self.policy_version)

    def update_weights(self, policy_version):
        self.policy_version = policy_version
        return policy_version

    def apply_weight_update(self, policy_version, update):
        result = update()
        self.policy_version = (
            self.policy_version + 1 if policy_version is None else policy_version
        )
        return result


class _Reward(BaseRewardModel):
    def __init__(self, mutate=None):
        self.mutate = mutate

    def score(self, prompts, responses):
        if self.mutate is not None:
            self.mutate()
        return torch.ones(len(prompts), len(responses[0]))


def _runner(raw, *, mutate=None):
    return RolloutRunner(
        _Generator(raw), _Reward(mutate), rollout_interval=4, max_policy_lag=3
    )


def test_runner_preserves_and_validates_trace_binding_on_publish_and_reuse():
    raw = _raw_rollout()
    raw.route_trace_batch = _bind(raw)
    runner = _runner(raw)

    result, fresh = runner({"instruction": ["same"]})
    assert fresh is True
    assert result.route_trace_batch is raw.route_trace_batch
    cached, fresh = runner({"instruction": ["same"]})
    assert cached is result
    assert fresh is False

    cached.responses[0, 0, 0] += 1
    with pytest.raises(RolloutRouteTraceError, match="response tokens"):
        runner({"instruction": ["same"]})


def test_runner_revalidates_trace_after_slow_scoring_before_cache_publish():
    raw = _raw_rollout()
    raw.route_trace_batch = _bind(raw)

    def mutate_tokens():
        raw.responses[0, 0, 0] += 1

    runner = _runner(raw, mutate=mutate_tokens)
    with pytest.raises(RolloutRouteTraceError, match="response tokens"):
        runner({"instruction": ["same"]})
    assert runner._cache is None


def test_runner_rejects_wrong_binding_type_but_none_keeps_default_path():
    plain = _raw_rollout()
    result, fresh = _runner(plain)({"instruction": ["plain"]})
    assert fresh is True
    assert result.route_trace_batch is None

    invalid = _raw_rollout(route_trace_batch="not-a-route-trace-batch")
    with pytest.raises(RolloutRouteTraceError, match="RolloutRouteTraceBatchV0"):
        _runner(invalid)({"instruction": ["invalid"]})


def test_manifest_is_payload_free_and_digest_binds_rollout_order():
    raw = _raw_rollout()
    first = _bind(raw, rollout_id="rollout-a")
    second = _bind(raw, rollout_id="rollout-b")

    manifest = first.manifest()
    assert "payload" not in manifest["items"][0][0]
    assert manifest["payload_nbytes"] == first.payload_nbytes
    assert manifest["items"][0][0]["artifact_digest"]
    assert first.artifact_digest != second.artifact_digest


def test_sharded_transport_is_deterministic_bounded_and_lossless():
    raw = _raw_rollout()
    bound = _bind(raw, _trace_grid(level=RouteTraceLevel.IDS_WEIGHTS_MARGIN))

    transport = RolloutRouteTraceTransportV0.from_batch(
        bound,
        max_shard_bytes=1024 * 1024,
        max_items_per_shard=2,
    )
    repeated = RolloutRouteTraceTransportV0.from_batch(
        bound,
        max_shard_bytes=1024 * 1024,
        max_items_per_shard=2,
    )

    assert len(transport.shard_payloads) == 2
    assert transport.shard_payloads == repeated.shard_payloads
    assert transport.manifest == repeated.manifest
    assert transport.manifest.schema_version == ROLLOUT_ROUTE_TRACE_SHARD_SCHEMA_VERSION
    assert all(
        descriptor.encoded_nbytes <= 1024 * 1024
        for descriptor in transport.manifest.shards
    )
    restored_manifest = RolloutRouteTraceShardManifestV0.loads(
        transport.manifest.dumps()
    )
    assert restored_manifest == transport.manifest
    assert restored_manifest.artifact_digest == transport.manifest.artifact_digest

    restored = restored_manifest.assemble(transport.shard_payloads)
    assert restored.artifact_digest == bound.artifact_digest
    assert restored.payload_nbytes == bound.payload_nbytes
    assert tuple(item.payload for row in restored.items for item in row) == tuple(
        item.payload for row in bound.items for item in row
    )
    restored.validate_against(
        policy_version=raw.policy_version,
        prompts=raw.prompts,
        prompt_mask=raw.prompt_mask,
        responses=raw.responses,
        response_mask=raw.response_mask,
        logprobs_old=raw.logprobs_old,
    )


def test_sharded_transport_fails_closed_on_missing_reordered_or_corrupt_frames():
    raw = _raw_rollout()
    bound = _bind(raw)
    transport = RolloutRouteTraceTransportV0.from_batch(
        bound,
        max_items_per_shard=1,
    )
    manifest = transport.manifest

    with pytest.raises(RolloutRouteTraceError, match="count"):
        manifest.assemble(transport.shard_payloads[:-1])
    reordered = list(transport.shard_payloads)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(RolloutRouteTraceError, match="size|digest"):
        manifest.assemble(reordered)

    corrupt = bytearray(transport.shard_payloads[0])
    corrupt[-1] ^= 1
    with pytest.raises(RolloutRouteTraceError, match="digest"):
        manifest.verify_shard(0, corrupt)
    with pytest.raises(RolloutRouteTraceError, match="immutable bytes"):
        replace(transport, shard_payloads=(corrupt, *transport.shard_payloads[1:]))

    foreign = RolloutRouteTraceTransportV0.from_batch(
        _bind(raw, rollout_id="rollout-other"),
        max_items_per_shard=1,
    )
    with pytest.raises(RolloutRouteTraceError, match="size|digest"):
        manifest.verify_shard(0, foreign.shard_payloads[0])


def test_shard_manifest_and_frame_decoders_reject_schema_and_size_ambiguity():
    transport = RolloutRouteTraceTransportV0.from_batch(
        _bind(_raw_rollout()),
        max_items_per_shard=1,
    )
    manifest_payload = transport.manifest.dumps()
    manifest = json.loads(manifest_payload)

    unknown = dict(manifest)
    unknown["future"] = True
    with pytest.raises(RolloutRouteTraceError, match="unknown"):
        RolloutRouteTraceShardManifestV0.loads(
            json.dumps(unknown, sort_keys=True, separators=(",", ":")).encode()
        )
    future = dict(manifest)
    future["schema_version"] = 1
    with pytest.raises(RolloutRouteTraceError, match="schema_version"):
        RolloutRouteTraceShardManifestV0.loads(
            json.dumps(future, sort_keys=True, separators=(",", ":")).encode()
        )
    with pytest.raises(RolloutRouteTraceError, match="canonical JSON"):
        RolloutRouteTraceShardManifestV0.loads(
            json.dumps(manifest, sort_keys=True, indent=2).encode()
        )
    with pytest.raises(RolloutRouteTraceError, match="size limit"):
        RolloutRouteTraceShardManifestV0.loads(
            manifest_payload,
            max_manifest_bytes=len(manifest_payload) - 1,
        )
    with pytest.raises(RolloutRouteTraceError, match="max_shard_bytes"):
        RolloutRouteTraceShardManifestV0.loads(
            manifest_payload,
            max_shard_bytes=transport.manifest.shards[0].encoded_nbytes - 1,
        )
    with pytest.raises(RolloutRouteTraceError, match="max_transport_bytes"):
        RolloutRouteTraceShardManifestV0.loads(
            manifest_payload,
            max_transport_bytes=(
                len(manifest_payload) + transport.manifest.encoded_nbytes - 1
            ),
        )

    frame = transport.shard_payloads[0]
    with pytest.raises(RolloutRouteTraceError, match="size limit"):
        RolloutRouteTraceShardCodecV0.loads(
            frame,
            max_shard_bytes=len(frame) - 1,
        )
    with pytest.raises(RolloutRouteTraceError, match="magic"):
        RolloutRouteTraceShardCodecV0.loads(b"badmagic" + frame[8:])
    with pytest.raises(RolloutRouteTraceError, match="trailing bytes"):
        RolloutRouteTraceShardCodecV0.loads(
            frame + b"x",
            max_shard_bytes=len(frame) + 1,
        )


def test_sharded_transport_rejects_an_item_larger_than_the_shard_cap():
    transport = RolloutRouteTraceTransportV0.from_batch(
        _bind(_raw_rollout()),
        max_items_per_shard=1,
    )
    smallest_frame = min(
        descriptor.encoded_nbytes for descriptor in transport.manifest.shards
    )
    with pytest.raises(RolloutRouteTraceError, match="one route trace item"):
        RolloutRouteTraceTransportV0.from_batch(
            _bind(_raw_rollout()),
            max_shard_bytes=smallest_frame - 1,
            max_items_per_shard=1,
        )


def _distributed_route_shard_worker():
    rank = get_rank()
    world_size = get_world_size()
    transport = RolloutRouteTraceTransportV0.from_batch(
        _bind(_raw_rollout()),
        max_items_per_shard=1,
    )
    local = transport.payloads_for_rank(rank, world_size)
    local_indices = tuple(index for index, _ in local)
    assert local_indices == transport.manifest.shard_indices_for_rank(rank, world_size)
    for index, payload in local:
        shard = transport.manifest.verify_shard(index, payload)
        assert shard.shard_index == index

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_indices)
    flattened = [index for indices in gathered for index in indices]
    assert sorted(flattened) == list(range(len(transport.shard_payloads)))
    assert len(flattened) == len(set(flattened))


@pytest.mark.parametrize("world_size", [2, 4])
def test_shard_assignment_is_complete_on_real_gloo_groups(world_size):
    spawn_parallel_fn(
        _distributed_route_shard_worker,
        world_size=world_size,
        backend="gloo",
        device_type="cpu",
    )
