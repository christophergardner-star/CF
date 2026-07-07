"""Episodic memory: time-stamped experiences that decay unless reinforced.

Recall returns the most salient matching memories and reinforces them on the way
out, so memories that are used often stay strong while unused ones fade and are
eventually pruned.
"""
from __future__ import annotations

from typing import Callable, Iterable

from memory.item import MemoryItem

Predicate = Callable[[MemoryItem], bool]


class EpisodicMemory:
    def __init__(self, capacity: int = 512) -> None:
        self.capacity = capacity
        self._items: dict[int, MemoryItem] = {}

    def store(self, item: MemoryItem) -> None:
        self._items[item.id] = item
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        if len(self._items) <= self.capacity:
            return
        # Forget the least salient memories first.
        ordered = sorted(self._items.values(), key=lambda m: m.salience)
        for victim in ordered[: len(self._items) - self.capacity]:
            self._items.pop(victim.id, None)

    def recall(self, k: int = 5, predicate: Predicate | None = None,
               reinforce: bool = True) -> list[MemoryItem]:
        pool: Iterable[MemoryItem] = self._items.values()
        if predicate is not None:
            pool = (m for m in pool if predicate(m))
        top = sorted(pool, key=lambda m: m.salience, reverse=True)[:k]
        if reinforce:
            for item in top:
                item.reinforce(0.15)
        return top

    def frequently_accessed(self, min_access: float) -> list[MemoryItem]:
        return [m for m in self._items.values() if m.access_count >= min_access]

    def decay(self, dt: float = 1.0) -> int:
        """Decay every memory and prune the dead ones; return count pruned."""
        for item in self._items.values():
            item.decay(dt)
        dead = [mid for mid, m in self._items.items() if not m.alive]
        for mid in dead:
            self._items.pop(mid, None)
        return len(dead)

    def all(self) -> list[MemoryItem]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)
