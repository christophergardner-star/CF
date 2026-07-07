"""Working memory: a small, volatile buffer of what is being thought about now.

Modelled loosely on the classic 7 +/- 2 capacity limit.  When it overflows the
least salient item is evicted -- the manager forwards evicted items to episodic
memory, which is how short-term experience becomes long-term experience.
"""
from __future__ import annotations

from memory.item import MemoryItem


class WorkingMemory:
    def __init__(self, capacity: int = 7) -> None:
        self.capacity = capacity
        self._items: list[MemoryItem] = []

    def add(self, item: MemoryItem) -> MemoryItem | None:
        """Add ``item``; return an evicted item if capacity was exceeded."""
        self._items.append(item)
        if len(self._items) <= self.capacity:
            return None
        self._items.sort(key=lambda m: m.salience)
        return self._items.pop(0)

    def decay(self, dt: float = 1.0) -> None:
        for item in self._items:
            item.decay(dt * 1.5)  # working memory fades fast
        self._items = [m for m in self._items if m.alive]

    def items(self) -> list[MemoryItem]:
        return list(self._items)

    def clear(self) -> list[MemoryItem]:
        flushed, self._items = self._items, []
        return flushed

    def __len__(self) -> int:
        return len(self._items)
