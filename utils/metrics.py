"""Metrics collection -- a rolling time series of organism vitals.

The organism records one snapshot per tick.  The dashboard reads the latest
snapshot; analysis code can read the full series.  Kept dependency-free so it is
trivial to unit test.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Iterable


class Metrics:
    """A bounded time series of per-tick snapshots."""

    def __init__(self, history: int = 512) -> None:
        self._rows: deque[dict[str, Any]] = deque(maxlen=history)

    def record(self, snapshot: dict[str, Any]) -> None:
        self._rows.append(dict(snapshot))

    @property
    def latest(self) -> dict[str, Any]:
        return dict(self._rows[-1]) if self._rows else {}

    def series(self, key: str) -> list[Any]:
        return [row.get(key) for row in self._rows]

    def average(self, key: str, window: int | None = None) -> float:
        values = [v for v in self.series(key) if isinstance(v, (int, float))]
        if window is not None:
            values = values[-window:]
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "ticks": float(len(self._rows)),
            "avg_energy": self.average("energy"),
            "avg_entropy": self.average("entropy"),
            "avg_curiosity": self.average("curiosity"),
            "avg_learning_rate": self.average("learning_rate"),
        }

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterable[dict[str, Any]]:
        return iter(self._rows)
