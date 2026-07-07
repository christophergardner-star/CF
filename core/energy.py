"""Energy accounting for the organism and every module.

Everything the kernel does costs energy: observing, thinking, recalling a
memory, learning, communicating, planning.  Energy regenerates slowly (rest).
When a budget runs low the owning entity is pushed toward sleep; when it is
plentiful the entity is free to explore.  ``EnergyBudget`` is intentionally
generic so the very same type powers the whole organism *and* each module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(Enum):
    """A unit of work that draws on an energy budget."""

    OBSERVE = "observe"
    THINK = "think"
    MEMORY_LOOKUP = "memory_lookup"
    CREATE_MEMORY = "create_memory"
    LEARN = "learn"
    COMMUNICATE = "communicate"
    PLAN = "plan"


DEFAULT_COSTS: dict[Action, float] = {
    Action.OBSERVE: 0.5,
    Action.THINK: 1.0,
    Action.MEMORY_LOOKUP: 0.75,
    Action.CREATE_MEMORY: 1.25,
    Action.LEARN: 2.0,
    Action.COMMUNICATE: 0.5,
    Action.PLAN: 1.5,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(slots=True)
class EnergyBudget:
    """A refillable pool of energy with per-action costs."""

    capacity: float = 100.0
    level: float = 100.0
    regen_rate: float = 3.0
    costs: dict[Action, float] = field(default_factory=lambda: dict(DEFAULT_COSTS))

    def __post_init__(self) -> None:
        self.level = _clamp(self.level, 0.0, self.capacity)

    @property
    def fraction(self) -> float:
        return self.level / self.capacity if self.capacity else 0.0

    def cost_of(self, action: Action, scale: float = 1.0) -> float:
        return self.costs.get(action, 1.0) * scale

    def can_afford(self, action: Action, scale: float = 1.0) -> bool:
        return self.level >= self.cost_of(action, scale)

    def consume(self, action: Action, scale: float = 1.0) -> bool:
        """Spend the cost of ``action``; return ``True`` iff it was affordable."""
        return self.spend(self.cost_of(action, scale))

    def spend(self, amount: float) -> bool:
        if amount <= 0:
            return True
        if self.level < amount:
            return False
        self.level -= amount
        return True

    def regenerate(self, dt: float = 1.0, multiplier: float = 1.0) -> None:
        self.level = _clamp(self.level + self.regen_rate * dt * multiplier, 0.0, self.capacity)

    def is_low(self, threshold: float = 0.2) -> bool:
        return self.fraction <= threshold

    def is_high(self, threshold: float = 0.8) -> bool:
        return self.fraction >= threshold
