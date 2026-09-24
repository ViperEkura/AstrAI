import json

import pytest
import torch

from astrai.trainer.training_telemetry import (
    NullTrainingTelemetry,
    TokenCostModel,
    TrainingTelemetry,
    count_batch_tokens,
)


@pytest.mark.parametrize(
    ("strategy", "batch", "expected"),
    [
        (
            "sft",
            {
                "input_ids": torch.tensor([[11, 12, 0, 0], [21, 22, 23, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
                "loss_mask": torch.tensor([[0, 1, 0, 0], [0, 1, 1, 0]]),
            },
            (5, 0),
        ),
        (
            "dpo",
            {
                "chosen": torch.tensor([[11, 12, 13, 0], [21, 22, 0, 0]]),
                "rejected": torch.tensor([[11, 14, 0, 0], [21, 23, 24, 0]]),
                "chosen_attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
                "rejected_attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
            },
            (10, 0),
        ),
        (
            "grpo",
            {
                "prompts": torch.tensor([[0, 11, 12], [21, 22, 23]]),
                "prompt_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
                "responses": torch.tensor(
                    [
                        [[31, 32, 0], [33, 34, 35]],
                        [[41, 0, 0], [42, 43, 0]],
                    ]
                ),
                "masks": torch.tensor(
                    [
                        [[1, 1, 0], [1, 1, 1]],
                        [[1, 0, 0], [1, 1, 0]],
                    ]
                ),
            },
            (10, 8),
        ),
    ],
)
def test_count_batch_tokens_matches_strategy_compute_shape(strategy, batch, expected):
    snapshots = {key: value.clone() for key, value in batch.items()}

    counts = count_batch_tokens(batch, strategy=strategy)

    assert (counts.input_tokens, counts.output_tokens) == expected
    assert counts.total_tokens == sum(expected)
    assert all(torch.equal(batch[key], value) for key, value in snapshots.items())


def test_token_cost_model_only_reports_configured_proxies():
    model = TokenCostModel(
        flops_per_token=6.0,
        activation_bytes_per_token=12,
        communication_bytes_per_token=4,
        duration_ms_per_token=0.25,
        confidence=0.7,
    )

    estimate = model.estimate(8)

    assert estimate.tokens == 8
    assert estimate.flops == 48.0
    assert estimate.activation_bytes == 96
    assert estimate.communication_bytes == 32
    assert estimate.expected_duration_ms == 2.0
    assert estimate.confidence == 0.7
    assert TokenCostModel().estimate(8).flops is None
    assert TokenCostModel().estimate(8).expected_duration_ms is None


class _MemoryProbe:
    def __init__(self, peak: int):
        self.peak = peak
        self.started_with = None

    def start(self, device):
        self.started_with = device

    def stop(self) -> int:
        return self.peak


def test_observe_batch_tracks_rank_local_pressure_and_emits_structured_trace():
    traces = []
    times = iter((10.0, 10.25))
    probe = _MemoryProbe(peak=123456)
    telemetry = TrainingTelemetry(
        cost_model=TokenCostModel(flops_per_token=2.0),
        sink=traces.append,
        clock=lambda: next(times),
        memory_probe=probe,
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
    }

    with telemetry.observe_batch(
        batch,
        strategy="seq",
        rank=2,
        world_size=4,
        epoch=3,
        optimizer_step=7,
        policy_version=11,
        device="cuda:2",
    ) as work:
        assert work.work_id == "train-r2-e3-s7-m0"
        assert work.phase == "train"
        assert work.rank == 2
        assert work.world_size == 4
        assert work.policy_version == 11
        assert work.cost.tokens == 3
        assert telemetry.inflight_tokens == 3

    assert telemetry.inflight_tokens == 0
    assert probe.started_with == "cuda:2"
    assert len(traces) == 1
    trace = traces[0]
    assert trace.work_item == work
    assert trace.status == "ok"
    assert trace.host_duration_ms == 250.0
    assert trace.peak_hbm_bytes == 123456
    assert trace.inflight_tokens_at_start == 3
    assert trace.inflight_tokens_at_end == 0
    assert trace.error_type is None
    payload = json.loads(trace.to_json())
    assert payload["event"] == "training_work_item_completed"
    assert payload["work_item"]["work_id"] == work.work_id


def test_observe_batch_releases_pressure_without_masking_training_error(caplog):
    traces = []

    def broken_sink(trace):
        traces.append(trace)
        raise RuntimeError("sink failed")

    telemetry = TrainingTelemetry(sink=broken_sink)
    batch = {"input_ids": torch.ones((2, 4), dtype=torch.long)}

    with (
        pytest.raises(ValueError, match="training failed"),
        telemetry.observe_batch(
            batch,
            strategy="seq",
            rank=0,
            world_size=1,
            epoch=0,
            optimizer_step=0,
        ),
    ):
        raise ValueError("training failed")

    assert telemetry.inflight_tokens == 0
    assert traces[0].status == "error"
    assert traces[0].error_type == "ValueError"
    assert "training telemetry sink failed" in caplog.text


def test_disabled_telemetry_does_not_inspect_batch_or_emit_trace():
    class _ExplodingBatch:
        def get(self, key, default=None):
            raise AssertionError("disabled telemetry inspected the batch")

    telemetry = NullTrainingTelemetry()

    with telemetry.observe_batch(
        _ExplodingBatch(),
        strategy="seq",
        rank=0,
        world_size=1,
        epoch=0,
        optimizer_step=0,
    ) as work:
        assert work is None

    assert telemetry.inflight_tokens == 0


def test_trainer_emits_one_trace_per_completed_batch(
    base_test_env, train_config_factory, device, caplog
):
    from tests.helpers import RandomTokenDataset

    train_config = train_config_factory(
        model_fn=lambda: base_test_env["model"],
        dataset=RandomTokenDataset(length=4, max_length=8),
        test_dir=base_test_env["test_dir"],
        device=device,
        batch_per_device=2,
        training_telemetry_enabled=True,
    )

    with caplog.at_level("INFO", logger="astrai.trainer.training_telemetry"):
        from astrai.trainer import Trainer

        Trainer(train_config).train()

    payloads = [
        json.loads(record.message.removeprefix("training_telemetry "))
        for record in caplog.records
        if record.message.startswith("training_telemetry ")
    ]
    assert len(payloads) == 2
    assert [payload["work_item"]["cost"]["tokens"] for payload in payloads] == [
        16,
        16,
    ]
    assert all(payload["work_item"]["rank"] == 0 for payload in payloads)
    assert all(payload["inflight_tokens_at_end"] == 0 for payload in payloads)
