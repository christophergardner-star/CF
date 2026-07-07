"""Perception: the organism's window onto the world.

Perception is the only module attached to an external sensor (injected as a
callable, so the module stays decoupled from any particular environment).  Each
tick it samples the world and publishes an :class:`ObservationEvent`, tagging it
with a cheap novelty estimate computed from a short memory of recent percepts.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable

from core.events import Event, ObservationEvent
from core.module import Module

Sensor = Callable[[], Any]


class PerceptionModule(Module):
    subscriptions: tuple[type[Event], ...] = ()  # senses the world, not the bus

    def __init__(self, *args: Any, sensor: Sensor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sensor: Sensor = sensor or (lambda: None)
        self._recent: deque[Any] = deque(maxlen=32)
        self.state["seen"] = 0

    def wants_to_act(self) -> bool:
        # There is always a world to sense, so perception is self-driving --
        # but (via has_work) it still stalls if load collapses its action set.
        return True

    def urgency(self) -> float:
        return max(0.4, super().urgency())

    def observe(self) -> list[Event]:
        percept = self._sensor()
        self.state["current"] = percept
        return self.drain_inbox()

    def think(self, observations: list[Event]) -> list[Event]:
        percept = self.state.get("current")
        if percept is None:
            return []
        novelty = self._novelty(percept)
        self._recent.append(percept)
        self.state["seen"] += 1
        # Surprising input raises internal entropy (drives downstream curiosity).
        self.entropy.perturb(novelty * 0.5)
        return [ObservationEvent(
            source=self.name,
            payload={"token": str(percept), "percept": percept, "novelty": novelty},
            priority=0.4 + 0.5 * novelty)]

    def learn(self, thoughts: list[Event]) -> None:
        # Confidence tracks familiarity: the more we have seen, the steadier.
        familiarity = min(1.0, self.state["seen"] / 64.0)
        self.confidence = 0.7 * self.confidence + 0.3 * familiarity

    def _novelty(self, percept: Any) -> float:
        if not self._recent:
            return 1.0
        seen = sum(1 for p in self._recent if p == percept)
        return 1.0 / (1.0 + seen)
