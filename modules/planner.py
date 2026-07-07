"""Planner: turns curiosity and thoughts into goals.

When curiosity spikes (or a surprising thought arrives) the planner forms a
small goal -- "investigate token X" -- and pushes it onto a goal stack, emitting
a :class:`PlannerEvent`.  It uses its temperature (derived from entropy) to
decide how boldly to explore: hot -> chase novelty, cool -> consolidate the
current goal.  Critic feedback closes plans out.
"""
from __future__ import annotations

from typing import Any

from core.energy import Action
from core.events import (CritiqueEvent, CuriosityEvent, Event, PlannerEvent,
                         ThoughtEvent)
from core.module import Module


class PlannerModule(Module):
    subscriptions: tuple[type[Event], ...] = (CuriosityEvent, ThoughtEvent, CritiqueEvent)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.state["goals"] = []          # active goal stack (list[str])
        self.state["completed"] = 0

    def wants_to_act(self) -> bool:
        return super().wants_to_act() or bool(self.state["goals"])

    def observe(self) -> list[Event]:
        return self.drain_inbox()

    def think(self, observations: list[Event]) -> list[Event]:
        outputs: list[Event] = []
        for event in observations:
            if isinstance(event, CuriosityEvent):
                outputs.extend(self._form_goal(event))
            elif isinstance(event, CritiqueEvent):
                self._close_goal(event)
        # If we hold a goal and have energy, (re)issue the current plan.  A plan
        # rejected by the critic stays open, so re-planning costs more PLAN work.
        if self.state["goals"] and self.can_afford(Action.PLAN):
            self.work(Action.PLAN)
            outputs.append(self._issue_plan())
        return outputs

    def _form_goal(self, event: CuriosityEvent) -> list[Event]:
        about = str(event.get("about", ""))
        goals: list[str] = self.state["goals"]
        if about and about not in goals and float(event.priority) > 0.4:
            goals.append(about)
            del goals[:-5]  # bounded backlog: keep the five most recent goals
            self.entropy.increase(0.05)  # opening a goal adds a little drive
        return []

    def _issue_plan(self) -> Event:
        goal = self.state["goals"][-1]
        # Planning depth is set by the reachable action manifold (which load
        # shrinks), and steadiness by fidelity: a stressed planner makes
        # shallower, noisier plans -- and once capacity drops below PLAN it
        # cannot plan at all, which is what eventually collapses it to rest.
        depth = max(1, 1 + round(2 * self.action_capacity))
        steps = ["attend", "predict", "verify"][:depth]
        noise = self.rng.uniform(-1.0, 1.0) * (1.0 - self.fidelity) * 0.4
        boldness = max(0.0, min(1.0, self.temperature + noise))
        return PlannerEvent(
            source=self.name,
            payload={"goal": goal, "about": goal, "boldness": boldness,
                     "steps": steps, "depth": depth},
            priority=min(1.0, 0.5 + boldness))

    def _close_goal(self, event: CritiqueEvent) -> None:
        # Only a positive verdict retires a goal; a "revise" keeps it open, which
        # forces the planner to keep working on it (the load-feedback loop).
        if event.get("verdict") != "pursue":
            return
        goal = str(event.get("about", ""))
        goals: list[str] = self.state["goals"]
        if goal in goals:
            goals.remove(goal)
            self.state["completed"] += 1
            self.confidence = min(1.0, self.confidence + 0.05)
            self.entropy.decrease(0.05)

    def learn(self, thoughts: list[Event]) -> None:
        # Fewer open goals -> calmer planner.
        backlog = len(self.state["goals"])
        if backlog == 0:
            self.entropy.decrease(0.03)
