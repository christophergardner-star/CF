"""Perturbation operators -- the experimenter's scalpel.

Each perturbation is scheduled at a tick and applied to a *live* organism using
only its public surface.  This is how biology is studied: lesion a part, starve
the system, flood it with contradictions, and measure what happens.

    LesionModule       remove an "organ" (does another compensate?)
    AblateSemantic     delete a fraction of consolidated knowledge
    StarveEnergy       cut the energy supply (which module dies first?)
    FloodContradiction inject inconsistent observations (does entropy explode?)
    SuppressSleep      forbid sleep (does long-term retention suffer?)
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from core.events import ObservationEvent


@dataclass
class Perturbation:
    at_tick: int
    label: str

    def apply(self, organism) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class LesionModule(Perturbation):
    name: str = ""

    def apply(self, organism) -> None:
        organism.retire_module(self.name)


@dataclass
class AblateSemantic(Perturbation):
    fraction: float = 0.3
    seed: int = 0

    def apply(self, organism) -> None:
        semantic = organism.memory.semantic
        keys = semantic.keys()
        if not keys:
            return
        rng = random.Random(self.seed)
        count = int(round(len(keys) * self.fraction))
        for key in rng.sample(keys, min(count, len(keys))):
            semantic.forget(key)


@dataclass
class StarveEnergy(Perturbation):
    factor: float = 0.2

    def apply(self, organism) -> None:
        organism.energy.level *= self.factor
        organism.energy.regen_rate *= self.factor
        for module in organism.modules:
            module.energy.level *= self.factor
            module.energy.regen_rate *= self.factor


@dataclass
class FloodContradiction(Perturbation):
    burst: int = 8
    seed: int = 0

    def apply(self, organism) -> None:
        # The same context ("signal") is mapped to contradictory tokens, which
        # maximises prediction error in language and drives entropy up.
        rng = random.Random(self.seed + organism.age)
        for _ in range(self.burst):
            token = rng.choice(["A", "B"])
            organism.bus.publish(ObservationEvent(
                source="world",
                payload={"token": token, "percept": token, "novelty": 1.0,
                         "label": "signal"},
                priority=0.9))


@dataclass
class SuppressSleep(Perturbation):
    """Keep every module perpetually awake (the "no rest" condition)."""

    def apply(self, organism) -> None:
        organism.scheduler.sleep_after_idle = 10 ** 9
        organism.scheduler.wake_threshold = 0.0
        for module in organism.modules:
            module.energy.regen_rate = max(module.energy.regen_rate,
                                           module.energy.capacity)
            module.energy.level = module.energy.capacity
            module.wake(organism.age)  # up now, and never allowed to nod off
