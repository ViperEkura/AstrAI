"""Unified per-task perf/stats: timing records, context-manager scopes, aggregate reporting."""

import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Deque, Dict, Generator, List, Literal, Optional

from astrai.config.inference_config import InferenceConfig

_config = InferenceConfig()


@dataclass
class TaskTiming:
    """Timestamp snapshots and computed metrics for one generation task.

    Created by :class:`MetricsCollector` at task-registration time;
    updated via ``record`` / ``mark_finished``.
    """

    task_id: str
    arrival_time: float
    prefill_start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    _decode_steps: int = 0
    _decode_total_s: float = 0.0

    # derived metrics

    @property
    def queue_wait_ms(self) -> Optional[float]:
        if self.prefill_start_time is not None:
            return (self.prefill_start_time - self.arrival_time) * 1000
        return None

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_time is not None:
            return (self.first_token_time - self.arrival_time) * 1000
        return None

    @property
    def prefill_tps(self) -> Optional[float]:
        if self.prefill_start_time is not None and self.first_token_time is not None:
            d = self.first_token_time - self.prefill_start_time
            if d > 0 and self.input_tokens > 0:
                return self.input_tokens / d
        return None

    @property
    def decode_tps(self) -> Optional[float]:
        if self.first_token_time is not None and self.finish_time is not None:
            d = self.finish_time - self.first_token_time
            dt = self.output_tokens - 1
            if dt > 0 and d > 0:
                return dt / d
        return None

    @property
    def decode_avg_ms(self) -> Optional[float]:
        if self._decode_steps > 0 and self._decode_total_s > 0:
            return (self._decode_total_s / self._decode_steps) * 1000
        return None

    @property
    def e2e_latency_ms(self) -> Optional[float]:
        if self.finish_time is not None:
            return (self.finish_time - self.arrival_time) * 1000
        return None

    @property
    def total_tps(self) -> Optional[float]:
        if self.finish_time is not None:
            total = self.input_tokens + self.output_tokens
            d = self.finish_time - self.arrival_time
            if total > 0 and d > 0:
                return total / d
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "queue_wait_ms": (
                round(self.queue_wait_ms, 2) if self.queue_wait_ms is not None else None
            ),
            "ttft_ms": (round(self.ttft_ms, 2) if self.ttft_ms is not None else None),
            "prefill_tps": (
                round(self.prefill_tps, 2) if self.prefill_tps is not None else None
            ),
            "decode_tps": (
                round(self.decode_tps, 2) if self.decode_tps is not None else None
            ),
            "decode_avg_ms": (
                round(self.decode_avg_ms, 2) if self.decode_avg_ms is not None else None
            ),
            "total_tps": (
                round(self.total_tps, 2) if self.total_tps is not None else None
            ),
            "e2e_latency_ms": (
                round(self.e2e_latency_ms, 2)
                if self.e2e_latency_ms is not None
                else None
            ),
        }


class MetricsCollector:
    """Single-owner perf/stats hub for all generation tasks.

    Usage::

        metrics = MetricsCollector()
        metrics.register(task_id, arrival_time)

        with metrics.record(task_ids, "prefill"):
            run_prefill(...)

        metrics.mark_finished(task_id, input_tokens, output_tokens)

        stats = metrics.get_stats()
    """

    def __init__(self, max_recent: int = _config.max_recent_tasks):
        self._timings: Dict[str, TaskTiming] = {}
        self._completed: Deque[TaskTiming] = deque(maxlen=max_recent)
        self._lock = threading.Lock()

        self._ttft_ms_sum = 0.0
        self._ttft_ms_count = 0
        self._decode_tps_sum = 0.0
        self._decode_tps_count = 0
        self._e2e_ms_sum = 0.0
        self._e2e_ms_count = 0

    def register(self, task_id: str):
        """Create a timing record for a newly-created task."""
        with self._lock:
            self._timings[task_id] = TaskTiming(
                task_id=task_id, arrival_time=time.time()
            )

    def mark_finished(self, task_id: str, input_tokens: int, output_tokens: int):
        """Close timing for a finished/aborted task and move it to completed."""
        with self._lock:
            timing = self._timings.pop(task_id, None)
            if timing is None:
                return
            timing.finish_time = time.time()
            timing.input_tokens = input_tokens
            timing.output_tokens = output_tokens
            self._completed.append(timing)
            self._accumulate(timing)

    # timing scopes

    @contextmanager
    def record(
        self, task_ids: List[str], phase: Literal["prefill", "decode"]
    ) -> Generator[None, None, None]:
        tic = time.time()
        yield
        toc = time.time()
        dt = toc - tic
        with self._lock:
            for tid in task_ids:
                t = self._timings.get(tid)
                if t is None:
                    continue
                if phase == "prefill":
                    t.prefill_start_time = tic
                    t.first_token_time = toc
                elif phase == "decode":
                    t._decode_steps += 1
                    t._decode_total_s += dt

    # aggregate stats

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats: Dict[str, Any] = {"in_flight_tasks": len(self._timings)}
            if self._ttft_ms_count > 0:
                stats["avg_ttft_ms"] = round(self._ttft_ms_sum / self._ttft_ms_count, 2)
            if self._decode_tps_count > 0:
                stats["avg_decode_tps"] = round(
                    self._decode_tps_sum / self._decode_tps_count, 2
                )
            if self._e2e_ms_count > 0:
                stats["avg_e2e_latency_ms"] = round(
                    self._e2e_ms_sum / self._e2e_ms_count, 2
                )
            if self._completed:
                stats["recent_tasks"] = [t.to_dict() for t in self._completed]
            return stats

    # internal

    def _accumulate(self, t: TaskTiming):
        if t.ttft_ms is not None:
            self._ttft_ms_sum += t.ttft_ms
            self._ttft_ms_count += 1
        if t.decode_tps is not None:
            self._decode_tps_sum += t.decode_tps
            self._decode_tps_count += 1
        if t.e2e_latency_ms is not None:
            self._e2e_ms_sum += t.e2e_latency_ms
            self._e2e_ms_count += 1
