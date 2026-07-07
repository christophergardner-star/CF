"""The memory manager: the only thing that touches the memory stores.

It subscribes to the event bus and forms memories automatically from the
organism's experience -- observations become episodic memories, thoughts pass
through working memory, and learning events reinforce semantic concepts.  It
also answers recall requests that arrive as :class:`MemoryEvent` messages, so
even memory access is fully event driven and no module holds a reference to it.
"""
from __future__ import annotations

import logging
from typing import Any

from core.events import (EventBus, LearningEvent, MemoryEvent, ObservationEvent,
                         ThoughtEvent)
from memory.consolidation import ConsolidationStats, MemoryConsolidation
from memory.episodic import EpisodicMemory
from memory.item import MemoryItem
from memory.semantic import SemanticMemory
from memory.working import WorkingMemory


class MemoryManager:
    def __init__(self, bus: EventBus, working: WorkingMemory,
                 episodic: EpisodicMemory, semantic: SemanticMemory,
                 consolidation: MemoryConsolidation,
                 logger: logging.Logger | None = None) -> None:
        self.bus = bus
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.consolidation = consolidation
        self.log = logger or logging.getLogger("memory")
        self._next_id = 0

        bus.subscribe(ObservationEvent, self._on_observation)
        bus.subscribe(ThoughtEvent, self._on_thought)
        bus.subscribe(LearningEvent, self._on_learning)
        bus.subscribe(MemoryEvent, self._on_memory)

    # -- construction ---------------------------------------------------------
    def _new_item(self, content: Any, *, importance: float, label: str) -> MemoryItem:
        item = MemoryItem(content=content, importance=importance, label=label,
                          id=self._next_id)
        self._next_id += 1
        return item

    # -- automatic memory formation ------------------------------------------
    def _on_observation(self, event: ObservationEvent) -> None:
        novelty = float(event.get("novelty", 0.5))
        label = str(event.get("token", event.get("percept", "")))
        item = self._new_item(dict(event.payload),
                              importance=0.4 + 0.5 * novelty, label=label)
        self.episodic.store(item)

    def _on_thought(self, event: ThoughtEvent) -> None:
        label = str(event.get("about", event.get("token", "")))
        item = self._new_item(dict(event.payload),
                              importance=float(event.priority), label=label)
        evicted = self.working.add(item)
        if evicted is not None:
            self.episodic.store(evicted)  # short-term -> long-term

    def _on_learning(self, event: LearningEvent) -> None:
        label = str(event.get("about", event.get("concept", "learning")))
        self.semantic.integrate(label, importance=0.7,
                                weight=float(event.get("value", 0.5)) + 0.5)

    def _on_memory(self, event: MemoryEvent) -> None:
        if event.get("op") != "recall":
            return  # ignore store/result messages -- avoids self-recursion
        query = event.get("query")
        k = int(event.get("k", 3))
        predicate = None
        if query is not None:
            predicate = lambda m: query in (m.label, str(m.content))  # noqa: E731
        hits = self.episodic.recall(k=k, predicate=predicate)
        self.bus.publish(MemoryEvent(
            source="memory",
            payload={"op": "result",
                     "requester": event.get("requester"),
                     "query": query,
                     "items": [m.content for m in hits]}))

    # -- consolidation & reporting -------------------------------------------
    def consolidate(self, idle: bool = False,
                    fidelity: float = 1.0) -> ConsolidationStats:
        return self.consolidation.consolidate(
            self.working, self.episodic, self.semantic, idle=idle, fidelity=fidelity)

    @property
    def count(self) -> int:
        return len(self.working) + len(self.episodic) + len(self.semantic)

    def snapshot(self) -> dict[str, int]:
        return {"working": len(self.working), "episodic": len(self.episodic),
                "semantic": len(self.semantic), "total": self.count}
