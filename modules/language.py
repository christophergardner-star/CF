"""Language: internal narration and simple sequence prediction.

The language module turns perceptions and curiosity into symbolic
:class:`ThoughtEvent`\\s.  It keeps a tiny first-order Markov model over tokens so
it can *predict* the next token; the gap between prediction and reality is its
own prediction error, which nudges its confidence and entropy.  This is a
stand-in for "language cognition", not a neural network.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from core.events import (CuriosityEvent, Event, ObservationEvent, ThoughtEvent)
from core.module import Module


class LanguageModule(Module):
    subscriptions: tuple[type[Event], ...] = (ObservationEvent, CuriosityEvent)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._prev: str | None = None
        self.state["last_prediction"] = None
        self.state["hits"] = 0
        self.state["total"] = 0

    def observe(self) -> list[Event]:
        return self.drain_inbox()

    def think(self, observations: list[Event]) -> list[Event]:
        outputs: list[Event] = []
        for event in observations:
            if isinstance(event, ObservationEvent):
                outputs.extend(self._narrate(event))
            elif isinstance(event, CuriosityEvent):
                outputs.append(self._wonder(event))
        return outputs

    def _narrate(self, event: ObservationEvent) -> list[Event]:
        token = str(event.get("token", ""))
        predicted = self.state.get("last_prediction")
        correct = predicted == token
        self.state["total"] += 1
        self.state["hits"] += int(correct)
        if self._prev is not None:
            self._transitions[self._prev][token] += 1
        self._prev = token
        # Low fidelity = noisy retrieval: the prediction is sometimes dropped.
        prediction = self._predict(token)
        if self.fidelity < 1.0 and self.rng.random() > self.fidelity:
            prediction = None
        self.state["last_prediction"] = prediction
        # A missed prediction is surprising and worth a thought.
        surprise = 0.0 if correct else 1.0
        self.entropy.perturb(surprise * 0.3)
        return [ThoughtEvent(
            source=self.name,
            payload={"about": token, "token": token,
                     "predicts": self.state["last_prediction"],
                     "surprise": surprise},
            priority=0.4 + 0.3 * surprise)]

    def _wonder(self, event: CuriosityEvent) -> Event:
        about = str(event.get("about", ""))
        return ThoughtEvent(
            source=self.name,
            payload={"about": about, "note": "wondering", "predicts": self._predict(about)},
            priority=float(event.priority))

    def _predict(self, token: str) -> str | None:
        following = self._transitions.get(token)
        if not following:
            return None
        return max(following.items(), key=lambda kv: kv[1])[0]

    def learn(self, thoughts: list[Event]) -> None:
        total = self.state["total"]
        accuracy = self.state["hits"] / total if total else 0.0
        self.confidence = 0.8 * self.confidence + 0.2 * accuracy
        if accuracy > 0.5:
            self.entropy.decrease(0.05)
