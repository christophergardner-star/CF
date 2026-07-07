"""Events and the in-process event bus.

Every subsystem communicates *exclusively* through events.  Modules never call
one another directly; they publish events and subscribe to the event *types*
they care about.  This is what keeps the kernel loosely coupled, event driven
and trivially extensible: a new module simply subscribes to the events it wants
and publishes the events it produces.

Dispatch is by type using the event's MRO, so subscribing to :class:`Event`
receives everything while subscribing to :class:`ObservationEvent` receives only
observations.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from time import time
from typing import Any, Callable, Mapping

Payload = Mapping[str, Any]
Handler = Callable[["Event"], None]


@dataclass(slots=True)
class Event:
    """Base class for everything that flows across the bus.

    ``id`` is assigned by the bus on :meth:`EventBus.publish`, giving a global,
    monotonically increasing ordering without any module-level global state.
    """

    source: str
    payload: Payload = field(default_factory=dict)
    priority: float = 0.5
    timestamp: float = field(default_factory=time)
    id: int = -1

    @property
    def kind(self) -> str:
        return type(self).__name__

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


# -- Concrete event vocabulary ------------------------------------------------
# These are deliberately thin: the semantics live in the payload so the schema
# stays open and new information can be threaded through without new classes.

@dataclass(slots=True)
class ObservationEvent(Event):
    """A raw or lightly processed percept produced by perception."""


@dataclass(slots=True)
class ThoughtEvent(Event):
    """An internal, symbolic thought produced by a cognitive module."""


@dataclass(slots=True)
class MemoryEvent(Event):
    """A memory operation: store request, recall request or recall result."""


@dataclass(slots=True)
class CuriosityEvent(Event):
    """Signals novelty / prediction error worth exploring."""


@dataclass(slots=True)
class LearningEvent(Event):
    """Signals that something was learned; typically lowers entropy."""


@dataclass(slots=True)
class PlannerEvent(Event):
    """A goal or plan emitted by the planner."""


@dataclass(slots=True)
class CritiqueEvent(Event):
    """An evaluation / value estimate emitted by the critic."""


@dataclass(slots=True)
class SleepEvent(Event):
    """A module announcing it is going to sleep."""


@dataclass(slots=True)
class WakeEvent(Event):
    """A module announcing it has woken up."""


@dataclass(slots=True)
class StructuralEvent(Event):
    """A structural change (spawn / prune / merge / reweight) took place."""


class EventBus:
    """A minimal synchronous publish/subscribe bus with a pump step.

    Publishing enqueues an event; :meth:`pump` drains the queue and dispatches
    each event to every handler subscribed to the event's type (or any of its
    base types).  Draining in an explicit step -- rather than dispatching on
    publish -- lets the organism control exactly when signals propagate, which
    keeps the tick loop deterministic and free of re-entrancy surprises.
    """

    def __init__(self) -> None:
        self._subscribers: dict[type[Event], list[Handler]] = defaultdict(list)
        self._queue: deque[Event] = deque()
        self._next_id: int = 0
        self._processed: int = 0

    def subscribe(self, event_type: type[Event], handler: Handler) -> None:
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: type[Event], handler: Handler) -> None:
        handlers = self._subscribers.get(event_type)
        if handlers and handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event) -> Event:
        event.id = self._next_id
        self._next_id += 1
        self._queue.append(event)
        return event

    def pump(self, max_events: int = 10_000) -> list[Event]:
        """Deliver queued events to their subscribers.

        Handlers may publish further events (e.g. a memory recall result); those
        are delivered within the same pump up to ``max_events`` to guarantee
        termination even if a handler misbehaves.
        """
        drained: list[Event] = []
        while self._queue and len(drained) < max_events:
            event = self._queue.popleft()
            drained.append(event)
            for etype in type(event).__mro__:
                handlers = self._subscribers.get(etype)
                if not handlers:
                    continue
                for handler in tuple(handlers):
                    handler(event)
        self._processed += len(drained)
        return drained

    @property
    def pending(self) -> int:
        return len(self._queue)

    @property
    def processed(self) -> int:
        return self._processed
