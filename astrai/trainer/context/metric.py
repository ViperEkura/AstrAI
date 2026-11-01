"""Committed metric history shared by training and checkpoint persistence."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MetricSnapshot:
    """A fixed prefix of history, independent of subsequent training updates."""

    source_path: Optional[Path]
    records: Tuple[str, ...]


class MetricHistory:
    """Stage update metrics, then retain committed records until publication.

    Saved history stays on disk; memory holds only records since the last save.
    The trainer owns commit/discard and the checkpoint callback owns save/ack.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        source = checkpoint_path / "metric.jsonl" if checkpoint_path else None
        self._source_path = source if source is not None and source.is_file() else None
        self._pending: List[Dict[str, Any]] = []
        self._committed: List[str] = []

    def stage(self, record: Dict[str, Any]) -> None:
        self._pending.append(dict(record))

    def commit(self, step: int, consumed_samples: int) -> None:
        records = [
            json.dumps({**record, "step": step, "consumed_samples": consumed_samples})
            + "\n"
            for record in self._pending
        ]
        self._committed.extend(records)
        self.discard_pending()

    def discard_pending(self) -> None:
        self._pending.clear()

    def snapshot(self) -> MetricSnapshot:
        return MetricSnapshot(self._source_path, tuple(self._committed))

    def mark_saved(self, checkpoint_path: Path) -> None:
        """Confirm synchronous publication before training resumes."""
        self._source_path = checkpoint_path / "metric.jsonl"
        self._committed.clear()
