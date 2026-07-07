from core.events import (EventBus, LearningEvent, ObservationEvent, ThoughtEvent)
from memory.consolidation import MemoryConsolidation
from memory.episodic import EpisodicMemory
from memory.item import MemoryItem
from memory.manager import MemoryManager
from memory.semantic import SemanticMemory
from memory.working import WorkingMemory


def test_memory_item_decay_and_reinforce():
    item = MemoryItem(content="x", importance=0.0, strength=1.0, decay_rate=0.5)
    item.decay()
    assert item.strength < 1.0
    before = item.strength
    item.reinforce()
    assert item.strength > before
    assert item.access_count == 1


def test_working_memory_evicts_least_salient():
    wm = WorkingMemory(capacity=2)
    assert wm.add(MemoryItem(content="a", importance=0.9, id=1)) is None
    assert wm.add(MemoryItem(content="b", importance=0.8, id=2)) is None
    evicted = wm.add(MemoryItem(content="c", importance=0.7, id=3))
    assert evicted is not None and evicted.content == "c"  # lowest salience leaves
    assert len(wm) == 2


def test_episodic_recall_reinforces_and_prunes():
    ep = EpisodicMemory(capacity=10)
    for i in range(3):
        ep.store(MemoryItem(content=i, importance=0.6, id=i))
    top = ep.recall(k=2)
    assert len(top) == 2
    assert all(m.access_count == 1 for m in top)
    weak = MemoryItem(content="weak", importance=0.0, strength=0.06, decay_rate=1.0, id=99)
    ep.store(weak)
    ep.decay()
    assert all(m.content != "weak" for m in ep.all())


def test_manager_forms_memories_from_events():
    bus = EventBus()
    manager = MemoryManager(bus, WorkingMemory(4), EpisodicMemory(64),
                            SemanticMemory(), MemoryConsolidation())
    bus.publish(ObservationEvent(source="perception",
                                 payload={"token": "red", "novelty": 0.9}))
    bus.publish(ThoughtEvent(source="language", payload={"about": "red"}))
    bus.publish(LearningEvent(source="critic", payload={"about": "red", "value": 0.8}))
    bus.pump()
    assert len(manager.episodic) >= 1
    assert len(manager.working) >= 1
    assert manager.semantic.query("red") is not None


def test_consolidation_promotes_frequent_memories():
    ep = EpisodicMemory(64)
    sem = SemanticMemory()
    item = MemoryItem(content="blue", label="blue", importance=0.7, id=1)
    item.access_count = 5
    ep.store(item)
    stats = MemoryConsolidation(promote_threshold=3).consolidate(
        WorkingMemory(4), ep, sem)
    assert stats.promoted >= 1
    assert sem.query("blue") is not None
