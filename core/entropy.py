"""Entropy: a module's internal uncertainty and source of thermodynamic drive.

High entropy means the entity is uncertain and should *explore* -- it tends to
spawn curiosity.  Low entropy means the entity is confident and should
*exploit* stable knowledge.  Every entity relaxes its entropy toward a personal
baseline over time; that self-regulation is what keeps the organism from either
freezing (all exploitation) or boiling (all exploration).
"""
from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class EntropyField:
    """A self-regulating uncertainty level in ``[0, 1]``."""

    level: float = 0.5
    baseline: float = 0.3
    relaxation: float = 0.05
    high_threshold: float = 0.66
    low_threshold: float = 0.33

    def __post_init__(self) -> None:
        self.level = _clamp(self.level)

    def increase(self, amount: float) -> None:
        self.level = _clamp(self.level + abs(amount))

    def decrease(self, amount: float) -> None:
        self.level = _clamp(self.level - abs(amount))

    def relax(self, dt: float = 1.0) -> None:
        """Pull entropy toward its baseline (exponential relaxation)."""
        self.level += (self.baseline - self.level) * _clamp(self.relaxation * dt)
        self.level = _clamp(self.level)

    def perturb(self, prediction_error: float) -> None:
        """Raise entropy in proportion to surprise, saturating near 1."""
        self.increase(abs(prediction_error) * (1.0 - self.level))

    @property
    def is_high(self) -> bool:
        return self.level >= self.high_threshold

    @property
    def is_low(self) -> bool:
        return self.level <= self.low_threshold

    @property
    def temperature(self) -> float:
        """Entropy read as a softmax-style temperature (kept strictly positive)."""
        return 0.05 + self.level
