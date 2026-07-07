"""Critic: evaluates plans and thoughts, and turns evaluation into learning.

The critic estimates a value for each plan (rewarding low-surprise, coherent
plans) and emits a :class:`CritiqueEvent` back to the planner plus a
:class:`LearningEvent` that ripples out to curiosity and memory.  Learning is
what actually lowers the organism's global uncertainty, so the critic is the
module that "consolidates knowledge".
"""
from __future__ import annotations

from typing import Any

from core.events import (CritiqueEvent, Event, LearningEvent, PlannerEvent,
                         ThoughtEvent)
from core.module import Module


class CriticModule(Module):
    subscriptions: tuple[type[Event], ...] = (PlannerEvent, ThoughtEvent)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.state["evaluations"] = 0
        self.state["avg_value"] = 0.0

    def observe(self) -> list[Event]:
        return self.drain_inbox()

    def think(self, observations: list[Event]) -> list[Event]:
        outputs: list[Event] = []
        for event in observations:
            if isinstance(event, PlannerEvent):
                outputs.extend(self._evaluate_plan(event))
            elif isinstance(event, ThoughtEvent):
                self._register_surprise(event)
        return outputs

    def _evaluate_plan(self, event: PlannerEvent) -> list[Event]:
        goal = str(event.get("about", ""))
        boldness = float(event.get("boldness", 0.3))
        # Value a plan for being decisive but not reckless (peaks at mid boldness).
        value = max(0.0, 1.0 - abs(boldness - 0.45) * 1.5)
        # Under low fidelity the critic's own read of value is noisy -- so it
        # misjudges plans and rejects good ones, feeding the re-planning loop.
        value += self.rng.uniform(-1.0, 1.0) * (1.0 - self.fidelity) * 0.4
        value = max(0.0, min(1.0, value))
        self._track(value)
        critique = CritiqueEvent(
            source=self.name,
            payload={"about": goal, "value": value, "verdict":
                     "pursue" if value > 0.5 else "revise"},
            priority=value)
        learning = LearningEvent(
            source=self.name,
            payload={"about": goal, "concept": goal, "value": value},
            priority=value)
        self.entropy.decrease(0.06 * value)  # good evaluation reduces uncertainty
        return [critique, learning]

    def _register_surprise(self, event: ThoughtEvent) -> None:
        surprise = float(event.get("surprise", 0.0))
        # A surprising world means our value estimates are less trustworthy.
        self.entropy.perturb(surprise * 0.1)

    def _track(self, value: float) -> None:
        n = self.state["evaluations"]
        self.state["avg_value"] = (self.state["avg_value"] * n + value) / (n + 1)
        self.state["evaluations"] = n + 1
        self.confidence = 0.85 * self.confidence + 0.15 * value

    def learn(self, thoughts: list[Event]) -> None:
        if self.state["avg_value"] > 0.5:
            self.entropy.decrease(0.02)
