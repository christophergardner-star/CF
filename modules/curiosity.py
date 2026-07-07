"""Curiosity: the exploration drive.

Curiosity rises with novelty, prediction error and high entropy, and falls after
successful learning.  It maintains a tiny predictive model (how often each token
has been seen) so it can estimate surprise, and emits a :class:`CuriosityEvent`
whenever something is worth exploring.  Learning events cool it back down --
closing the explore/exploit loop.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from core.events import (CuriosityEvent, Event, LearningEvent, ObservationEvent)
from core.module import Module


class CuriosityModule(Module):
    subscriptions: tuple[type[Event], ...] = (ObservationEvent, LearningEvent)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._counts: dict[str, int] = defaultdict(int)
        self.state["curiosity"] = 0.3
        self.state["prediction_error"] = 0.0

    def wants_to_act(self) -> bool:
        return super().wants_to_act() or self.state["curiosity"] > 0.55

    def urgency(self) -> float:
        return min(1.0, super().urgency() + 0.5 * self.state["curiosity"])

    def observe(self) -> list[Event]:
        return self.drain_inbox()

    def think(self, observations: list[Event]) -> list[Event]:
        outputs: list[Event] = []
        for event in observations:
            if isinstance(event, LearningEvent):
                self._satisfy(event)
            elif isinstance(event, ObservationEvent):
                outputs.extend(self._explore(event))
        return outputs

    def _explore(self, event: ObservationEvent) -> list[Event]:
        token = str(event.get("token", ""))
        seen = self._counts[token]
        self._counts[token] = seen + 1
        prediction_error = 1.0 / (1.0 + seen)  # unseen -> 1, familiar -> small
        self.state["prediction_error"] = prediction_error
        self.entropy.perturb(prediction_error)
        self.state["curiosity"] = min(1.0, self.state["curiosity"] + 0.4 * prediction_error)
        if prediction_error < 0.3 and not self.entropy.is_high:
            return []  # familiar & confident -> nothing to be curious about
        return [CuriosityEvent(
            source=self.name,
            payload={"about": token, "novelty": prediction_error,
                     "curiosity": self.state["curiosity"]},
            priority=self.state["curiosity"])]

    def _satisfy(self, event: LearningEvent) -> None:
        value = float(event.get("value", 0.5))
        self.state["curiosity"] = max(0.0, self.state["curiosity"] - 0.3 * (0.5 + value))
        self.entropy.decrease(0.15 * (0.5 + value))
        self.confidence = min(1.0, self.confidence + 0.05)

    def learn(self, thoughts: list[Event]) -> None:
        # Curiosity slowly self-satiates even without explicit learning.
        self.state["curiosity"] *= 0.97
