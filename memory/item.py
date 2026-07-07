"""The unit of memory.

A :class:`MemoryItem` carries the things every memory needs regardless of which
store it lives in: importance, a timestamp, a strength that decays over time,
an access counter that lets frequently used memories strengthen, and links to
related memories.  Salience (strength x importance) is the single scalar the
rest of the system ranks by.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass(slots=True)
class MemoryItem:
    content: Any
    label: str = ""                       # hashable key used for consolidation
    importance: float = 0.5
    timestamp: float = field(default_factory=time)
    strength: float = 1.0
    decay_rate: float = 0.03
    access_count: int = 0
    connections: set[int] = field(default_factory=set)
    id: int = -1

    @property
    def salience(self) -> float:
        """How much this memory "matters" right now."""
        return self.strength * (0.25 + 0.75 * self.importance)

    @property
    def alive(self) -> bool:
        return self.strength > 0.05

    def reinforce(self, amount: float = 0.25) -> None:
        """Accessing / replaying a memory makes it stickier."""
        self.access_count += 1
        self.strength = min(1.5, self.strength + amount)

    def decay(self, dt: float = 1.0) -> None:
        """Weaken over time; important memories resist decay."""
        effective = self.decay_rate * (1.0 - 0.6 * self.importance)
        self.strength *= math.exp(-effective * dt)

    def link(self, other_id: int) -> None:
        if other_id != self.id:
            self.connections.add(other_id)
