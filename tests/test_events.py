from core.events import (Event, EventBus, LearningEvent, ObservationEvent,
                         ThoughtEvent)


def test_publish_assigns_monotonic_ids():
    bus = EventBus()
    a = bus.publish(ObservationEvent(source="s"))
    b = bus.publish(ThoughtEvent(source="s"))
    assert (a.id, b.id) == (0, 1)


def test_pump_dispatches_by_type():
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(ObservationEvent, lambda e: seen.append("obs"))
    bus.subscribe(ThoughtEvent, lambda e: seen.append("thought"))
    bus.publish(ObservationEvent(source="s"))
    bus.publish(ThoughtEvent(source="s"))
    drained = bus.pump()
    assert len(drained) == 2
    assert sorted(seen) == ["obs", "thought"]
    assert bus.processed == 2


def test_subscribing_to_base_receives_all_events():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(Event, seen.append)
    bus.publish(ObservationEvent(source="s"))
    bus.publish(LearningEvent(source="s"))
    bus.pump()
    assert len(seen) == 2


def test_pump_is_empty_after_draining():
    bus = EventBus()
    bus.publish(ObservationEvent(source="s"))
    bus.pump()
    assert bus.pending == 0
    assert bus.pump() == []
