"""Hypothesis: a lightweight module the organism grows for itself.

The structural adapter spawns a ``HypothesisModule`` when curiosity stays high
for a while: a fresh organ dedicated to chewing on whatever the organism is
currently curious about.  It listens for curiosity, forms a throwaway
"hypothesis" and, once it has gathered a little evidence, emits a
:class:`LearningEvent` -- helping cool the very curiosity that created it.  If it
falls idle it gets pruned again.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from core.events import CuriosityEvent, Event, LearningEvent
from core.module import Module


class HypothesisModule(Module):
    subscriptions: tuple[type[Event], ...] = (CuriosityEvent,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._evidence: dict[str, int] = defaultdict(int)
        self.state["confirmed"] = 0

    def observe(self) -> list[Event]:
        return self.drain_inbox()

    def think(self, observations: list[Event]) -> list[Event]:
        outputs: list[Event] = []
        for event in observations:
            if not isinstance(event, CuriosityEvent):
                continue
            about = str(event.get("about", ""))
            self._evidence[about] += 1
            # Enough repeated evidence -> we "confirm" a small hypothesis.
            if self._evidence[about] >= 3:
                self._evidence[about] = 0
                self.state["confirmed"] += 1
                outputs.append(LearningEvent(
                    source=self.name,
                    payload={"about": about, "concept": about, "value": 0.6},
                    priority=0.6))
        return outputs

    def learn(self, thoughts: list[Event]) -> None:
        if thoughts:
            self.confidence = min(1.0, self.confidence + 0.05)
            self.entropy.decrease(0.05)
