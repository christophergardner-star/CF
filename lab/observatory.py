"""The Observatory: drives an organism tick-by-tick and records everything.

It taps the event bus (counting every event by type and capturing structural
changes) and, after each tick, snapshots the organism's vitals and every
module's state into a :class:`Frame`.  It also tracks the birth and death tick
of every episodic memory so that memory-survival curves can be computed later.

The Observatory never mutates the organism; perturbations passed to :meth:`run`
do that at their scheduled ticks.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from core.events import Event, StructuralEvent


@dataclass
class Frame:
    tick: int
    energy: float
    energy_fraction: float
    entropy: float
    curiosity: float
    learning_rate: float
    health: float
    active: int
    sleeping: int
    events_delta: int
    event_counts: dict[str, int]
    memories: dict[str, int]
    module_states: dict[str, str]
    module_energy: dict[str, float]
    module_entropy: dict[str, float]
    module_confidence: dict[str, float]
    perturbations: list[str] = field(default_factory=list)
    structural: list[str] = field(default_factory=list)
    load: float = 0.0
    fidelity: float = 1.0
    reservoir: float = 0.0
    module_load: dict[str, float] = field(default_factory=dict)


class Observatory:
    def __init__(self, organism) -> None:
        self.organism = organism
        self.frames: list[Frame] = []
        self.first_seen: dict[int, int] = {}
        self.last_seen: dict[int, int] = {}
        self._counts: dict[str, int] = defaultdict(int)
        self._delta = 0
        self._structural: list[str] = []
        organism.bus.subscribe(Event, self._on_event)

    # -- bus tap --------------------------------------------------------------
    def _on_event(self, event: Event) -> None:
        self._counts[event.kind] += 1
        self._delta += 1
        if isinstance(event, StructuralEvent):
            self._structural.append(str(event.get("change", "?")))

    # -- driving the organism -------------------------------------------------
    def run(self, ticks: int, perturbations: Iterable | None = None,
            on_frame: Callable[[Frame], None] | None = None) -> list[Frame]:
        schedule: dict[int, list] = defaultdict(list)
        for perturbation in perturbations or []:
            schedule[perturbation.at_tick].append(perturbation)

        for _ in range(ticks):
            tick = self.organism.age + 1
            applied: list[str] = []
            for perturbation in schedule.get(tick, []):
                perturbation.apply(self.organism)
                applied.append(perturbation.label)
            self.organism.tick()
            frame = self._capture(applied)
            self.frames.append(frame)
            if on_frame is not None:
                on_frame(frame)
        return self.frames

    def _capture(self, applied: list[str]) -> Frame:
        org = self.organism
        snap = org.metrics.latest
        reports = org.reports()
        tick = org.age

        for item in org.memory.episodic.all():
            self.first_seen.setdefault(item.id, tick)
            self.last_seen[item.id] = tick

        frame = Frame(
            tick=tick,
            energy=float(snap.get("energy", 0.0)),
            energy_fraction=float(snap.get("energy_fraction", 0.0)),
            entropy=float(snap.get("entropy", 0.0)),
            curiosity=float(snap.get("curiosity", 0.0)),
            learning_rate=float(snap.get("learning_rate", 0.0)),
            health=float(snap.get("health", 0.0)),
            active=int(snap.get("active", 0)),
            sleeping=int(snap.get("sleeping", 0)),
            events_delta=self._delta,
            event_counts=dict(self._counts),
            memories=org.memory.snapshot(),
            module_states={r["name"]: r["state"] for r in reports},
            module_energy={r["name"]: r["energy"] for r in reports},
            module_entropy={r["name"]: r["entropy"] for r in reports},
            module_confidence={r["name"]: r["confidence"] for r in reports},
            perturbations=applied,
            structural=list(self._structural),
            load=float(snap.get("load", 0.0)),
            fidelity=float(snap.get("fidelity", 1.0)),
            reservoir=float(snap.get("reservoir", 0.0)),
            module_load={r["name"]: r.get("load", 0.0) for r in reports},
        )
        self._counts = defaultdict(int)
        self._delta = 0
        self._structural = []
        return frame

    # -- memory survival ------------------------------------------------------
    def memory_lifetimes(self) -> list[tuple[int, bool]]:
        """(lifetime_in_ticks, died_before_end) for every episodic memory seen."""
        final = self.organism.age
        lifetimes: list[tuple[int, bool]] = []
        for mid, first in self.first_seen.items():
            last = self.last_seen[mid]
            lifetimes.append((last - first + 1, last < final))
        return lifetimes
