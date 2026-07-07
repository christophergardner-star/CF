"""Semantic memory: consolidated, timeless concepts.

Where episodic memory stores *what happened*, semantic memory stores *what is
generally true*.  Consolidation promotes recurring episodic patterns into
concepts here; each concept accrues support and strength as evidence repeats and
slowly decays when it stops being reinforced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import time


@dataclass(slots=True)
class Concept:
    key: str
    strength: float = 0.5
    importance: float = 0.5
    support: int = 0
    last_update: float = field(default_factory=time)


class SemanticMemory:
    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = {}

    def integrate(self, key: str, importance: float = 0.5, weight: float = 1.0) -> Concept:
        concept = self._concepts.get(key)
        if concept is None:
            concept = Concept(key=key, strength=0.4, importance=importance)
            self._concepts[key] = concept
        concept.support += 1
        concept.strength = min(2.0, concept.strength + 0.2 * weight)
        concept.importance = max(concept.importance, importance)
        concept.last_update = time()
        return concept

    def query(self, key: str) -> Concept | None:
        return self._concepts.get(key)

    def keys(self) -> list[str]:
        return list(self._concepts)

    def forget(self, key: str) -> bool:
        """Remove a concept outright (used by ablation experiments)."""
        return self._concepts.pop(key, None) is not None

    def top(self, k: int = 5) -> list[Concept]:
        return sorted(self._concepts.values(), key=lambda c: c.strength, reverse=True)[:k]

    def decay(self, dt: float = 1.0, rate: float = 0.01) -> int:
        """Weaken concepts; prune those that fade below a floor (unless well
        supported).  Returns the number pruned."""
        pruned = 0
        for key in list(self._concepts):
            concept = self._concepts[key]
            concept.strength *= (1.0 - rate * dt)
            if concept.strength < 0.1 and concept.support < 3:
                del self._concepts[key]
                pruned += 1
        return pruned

    def __len__(self) -> int:
        return len(self._concepts)
