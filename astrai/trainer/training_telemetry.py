"""Rank-local observability for future training admission decisions.

The telemetry path deliberately does not choose a rank or synchronize workers.  It
turns each local microbatch into a :class:`WorkItem`, tracks its token pressure,
and emits one structured completion trace that a later scheduler can consume.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchTokenCounts:
    """Non-overlapping token counts used by one training microbatch."""

    input_tokens: int
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class CostEstimate:
    """A token-exact estimate with optional calibrated cost proxies."""

    tokens: int
    flops: float | None
    activation_bytes: int | None
    communication_bytes: int | None
    expected_duration_ms: float | None
    confidence: float


@dataclass(frozen=True)
class TokenCostModel:
    """Convert exact token counts into explicitly configured cost proxies.

    A zero coefficient means "unknown" and is reported as ``None`` rather than a
    fabricated zero-cost estimate.
    """

    flops_per_token: float = 0.0
    activation_bytes_per_token: int = 0
    communication_bytes_per_token: int = 0
    duration_ms_per_token: float = 0.0
    confidence: float = 1.0

    def __post_init__(self) -> None:
        coefficients = (
            self.flops_per_token,
            self.activation_bytes_per_token,
            self.communication_bytes_per_token,
            self.duration_ms_per_token,
        )
        if any(value < 0 for value in coefficients):
            raise ValueError("training cost coefficients must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("training cost confidence must be in [0, 1]")

    def estimate(self, tokens: int) -> CostEstimate:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        return CostEstimate(
            tokens=tokens,
            flops=(tokens * self.flops_per_token if self.flops_per_token else None),
            activation_bytes=(
                tokens * self.activation_bytes_per_token
                if self.activation_bytes_per_token
                else None
            ),
            communication_bytes=(
                tokens * self.communication_bytes_per_token
                if self.communication_bytes_per_token
                else None
            ),
            expected_duration_ms=(
                tokens * self.duration_ms_per_token
                if self.duration_ms_per_token
                else None
            ),
            confidence=self.confidence,
        )


@dataclass(frozen=True)
class WorkItem:
    """One rank-local training unit observed without changing its placement."""

    work_id: str
    phase: str
    strategy: str
    rank: int
    world_size: int
    epoch: int
    optimizer_step: int
    microbatch: int
    input_tokens: int
    output_tokens: int
    cost: CostEstimate
    policy_version: int | None = None
    estimation_error: str | None = None


@dataclass(frozen=True)
class TrainingTrace:
    """Structured completion record for a :class:`WorkItem`."""

    work_item: WorkItem
    status: str
    host_duration_ms: float
    peak_hbm_bytes: int
    inflight_tokens_at_start: int
    inflight_tokens_at_end: int
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "training_work_item_completed",
            **asdict(self),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _numel(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, Mapping):
        return sum(_numel(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_numel(item) for item in value)
    return 1 if value is not None else 0


def _mask_tokens(mask: Any) -> int:
    if isinstance(mask, torch.Tensor):
        return int(mask.detach().to(dtype=torch.int64).sum().item())
    if isinstance(mask, Mapping):
        return sum(_mask_tokens(item) for item in mask.values())
    if isinstance(mask, (list, tuple)):
        return sum(_mask_tokens(item) for item in mask)
    return int(bool(mask))


def _sequence_tokens(value: Any, mask: Any | None) -> int:
    return _mask_tokens(mask) if mask is not None else _numel(value)


def _group_size(batch: Mapping[str, Any]) -> int:
    responses = batch.get("responses")
    if isinstance(responses, torch.Tensor) and responses.ndim >= 3:
        return int(responses.shape[1])
    if isinstance(responses, (list, tuple)) and responses:
        first = responses[0]
        return len(first) if isinstance(first, (list, tuple)) else 1
    masks = batch.get("masks")
    if isinstance(masks, torch.Tensor) and masks.ndim >= 3:
        return int(masks.shape[1])
    return 1


def count_batch_tokens(batch: Mapping[str, Any], *, strategy: str) -> BatchTokenCounts:
    """Count tokens that participate in the strategy's model forwards.

    Prompt tokens in GRPO are expanded once per response group, matching the
    strategy implementation. DPO counts both chosen and rejected sequences.
    Sequence/SFT training counts attention-valid inputs and never double-counts
    labels or the loss mask.
    """

    if "chosen" in batch and "rejected" in batch:
        chosen = _sequence_tokens(batch["chosen"], batch.get("chosen_attention_mask"))
        rejected = _sequence_tokens(
            batch["rejected"], batch.get("rejected_attention_mask")
        )
        return BatchTokenCounts(input_tokens=chosen + rejected)

    if "prompts" in batch and "responses" in batch:
        prompts = _sequence_tokens(batch["prompts"], batch.get("prompt_mask"))
        responses = _sequence_tokens(batch["responses"], batch.get("masks"))
        return BatchTokenCounts(
            input_tokens=prompts * _group_size(batch),
            output_tokens=responses,
        )

    if "input_ids" in batch:
        return BatchTokenCounts(
            input_tokens=_sequence_tokens(
                batch["input_ids"], batch.get("attention_mask")
            )
        )

    raise ValueError(f"cannot estimate tokens for {strategy!r} batch")


class MemoryProbe(Protocol):
    def start(self, device: str | torch.device | None) -> None: ...

    def stop(self) -> int: ...


class _CudaPeakMemoryProbe:
    def __init__(self) -> None:
        self._device: torch.device | None = None

    def start(self, device: str | torch.device | None) -> None:
        self._device = None
        if device is None:
            return
        resolved = torch.device(device)
        if resolved.type != "cuda" or not torch.cuda.is_available():
            return
        self._device = resolved
        torch.cuda.reset_peak_memory_stats(resolved)

    def stop(self) -> int:
        if self._device is None:
            return 0
        return int(torch.cuda.max_memory_allocated(self._device))


TraceSink = Callable[[TrainingTrace], None]


class TrainingTelemetry:
    """Observe local microbatches and expose their instantaneous token pressure."""

    def __init__(
        self,
        *,
        cost_model: TokenCostModel | None = None,
        sink: TraceSink | None = None,
        clock: Callable[[], float] = time.perf_counter,
        memory_probe: MemoryProbe | None = None,
    ) -> None:
        self.cost_model = cost_model or TokenCostModel()
        self._sink = sink
        self._clock = clock
        self._memory_probe = memory_probe or _CudaPeakMemoryProbe()
        self._lock = threading.Lock()
        self._next_microbatch = 0
        self._inflight_tokens = 0

    @property
    def inflight_tokens(self) -> int:
        with self._lock:
            return self._inflight_tokens

    def _new_work_item(
        self,
        batch: Mapping[str, Any],
        *,
        strategy: str,
        rank: int,
        world_size: int,
        epoch: int,
        optimizer_step: int,
        policy_version: int | None,
    ) -> tuple[WorkItem, int]:
        estimation_error = None
        try:
            counts = count_batch_tokens(batch, strategy=strategy)
        except Exception as exc:  # noqa: BLE001 - telemetry must fail open
            counts = BatchTokenCounts(0, 0)
            estimation_error = type(exc).__name__
            logger.warning("training telemetry token estimation failed: %s", exc)

        cost = self.cost_model.estimate(counts.total_tokens)
        with self._lock:
            microbatch = self._next_microbatch
            self._next_microbatch += 1
            self._inflight_tokens += cost.tokens
            inflight = self._inflight_tokens

        return (
            WorkItem(
                work_id=(f"train-r{rank}-e{epoch}-s{optimizer_step}-m{microbatch}"),
                phase="train",
                strategy=strategy,
                rank=rank,
                world_size=world_size,
                epoch=epoch,
                optimizer_step=optimizer_step,
                microbatch=microbatch,
                input_tokens=counts.input_tokens,
                output_tokens=counts.output_tokens,
                cost=cost,
                policy_version=policy_version,
                estimation_error=estimation_error,
            ),
            inflight,
        )

    def _start_memory_probe(self, device: str | torch.device | None) -> None:
        try:
            self._memory_probe.start(device)
        except Exception as exc:  # noqa: BLE001 - telemetry must fail open
            logger.warning("training telemetry HBM probe start failed: %s", exc)

    def _stop_memory_probe(self) -> int:
        try:
            return self._memory_probe.stop()
        except Exception as exc:  # noqa: BLE001 - telemetry must fail open
            logger.warning("training telemetry HBM probe stop failed: %s", exc)
            return 0

    def _emit(self, trace: TrainingTrace) -> None:
        logger.info("training_telemetry %s", trace.to_json())
        if self._sink is None:
            return
        try:
            self._sink(trace)
        except Exception:
            logger.exception("training telemetry sink failed")

    @contextmanager
    def observe_batch(
        self,
        batch: Mapping[str, Any],
        *,
        strategy: str,
        rank: int,
        world_size: int,
        epoch: int,
        optimizer_step: int,
        policy_version: int | None = None,
        device: str | torch.device | None = None,
    ) -> Iterator[WorkItem]:
        work_item, inflight_at_start = self._new_work_item(
            batch,
            strategy=strategy,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
            optimizer_step=optimizer_step,
            policy_version=policy_version,
        )
        self._start_memory_probe(device)
        started_at = self._clock()
        status = "ok"
        error_type = None
        try:
            yield work_item
        except BaseException as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            duration_ms = max(0.0, (self._clock() - started_at) * 1000.0)
            peak_hbm_bytes = self._stop_memory_probe()
            with self._lock:
                self._inflight_tokens -= work_item.cost.tokens
                inflight_at_end = self._inflight_tokens
            self._emit(
                TrainingTrace(
                    work_item=work_item,
                    status=status,
                    host_duration_ms=duration_ms,
                    peak_hbm_bytes=peak_hbm_bytes,
                    inflight_tokens_at_start=inflight_at_start,
                    inflight_tokens_at_end=inflight_at_end,
                    error_type=error_type,
                )
            )


class NullTrainingTelemetry:
    """Minimal disabled implementation that never inspects the batch."""

    @property
    def inflight_tokens(self) -> int:
        return 0

    @contextmanager
    def observe_batch(self, batch: Any, **kwargs: Any) -> Iterator[None]:
        yield None


def create_training_telemetry(config: Any) -> TrainingTelemetry | NullTrainingTelemetry:
    if not config.training_telemetry_enabled:
        return NullTrainingTelemetry()
    return TrainingTelemetry(
        cost_model=TokenCostModel(
            flops_per_token=config.training_cost_flops_per_token,
            activation_bytes_per_token=(
                config.training_cost_activation_bytes_per_token
            ),
            communication_bytes_per_token=(
                config.training_cost_communication_bytes_per_token
            ),
            duration_ms_per_token=config.training_cost_duration_ms_per_token,
            confidence=config.training_cost_confidence,
        )
    )
