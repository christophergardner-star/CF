"""The abstract ``Module`` -- the base "organ" every cognitive module extends.

A module owns local state only (energy, entropy, temperature, confidence,
importance, connections, ...).  It never references another module: it receives
work as events in its inbox and produces work as events it publishes.  The
kernel drives execution through the :meth:`Module.step` template method, which
sequences observe -> think -> learn -> communicate while charging energy for
each action, so a tired module naturally does less.

Concrete modules implement :meth:`observe`, :meth:`think` and :meth:`learn`; the
lifecycle hooks (:meth:`communicate`, :meth:`sleep`, :meth:`wake`,
:meth:`report`) have sensible defaults that can be overridden.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

from core.energy import Action, EnergyBudget
from core.entropy import EntropyField
from core.events import Event, EventBus, SleepEvent, WakeEvent
from core.lifecycle import Lifecycle, LifecycleState
from core.metabolism import ClassicMetabolism, EnergyReservoir, Metabolism


class Module(ABC):
    """Base class for all cognitive modules.

    Attributes mirror the "organism" theme: each module has its own energy,
    entropy, temperature, confidence, importance and (Hebbian) connections.
    """

    #: Event types this module wants delivered to its inbox.
    subscriptions: tuple[type[Event], ...] = ()

    def __init__(self, name: str, bus: EventBus, *,
                 energy: EnergyBudget | None = None,
                 entropy: EntropyField | None = None,
                 metabolism: Metabolism | None = None,
                 base_priority: float = 0.5, importance: float = 0.5,
                 idle_sleep_ticks: int = 6,
                 options: dict[str, Any] | None = None) -> None:
        self.name = name
        self.bus = bus
        # A module's energy and entropy live inside its metabolism (the physical
        # substrate).  If none is supplied we default to the classic economics so
        # that plain ``energy=``/``entropy=`` construction keeps working.
        if metabolism is None:
            metabolism = ClassicMetabolism(energy or EnergyBudget(),
                                           entropy or EntropyField())
        self.metabolism = metabolism
        self.energy = metabolism.energy
        self.entropy = metabolism.entropy
        self.base_priority = base_priority
        self.importance = importance
        self.idle_sleep_ticks = idle_sleep_ticks
        self.options = options or {}

        self.temperature: float = self.entropy.temperature
        self.confidence: float = 0.5
        self.state: dict[str, Any] = {}
        self.connections: dict[str, float] = {}  # source-name -> Hebbian weight
        self.lifecycle = Lifecycle()
        self.rng = random.Random(hash((name, self.options.get("seed", 0))) & 0xFFFFFFFF)

        self._inbox: deque[Event] = deque(maxlen=256)
        for event_type in self.subscriptions:
            bus.subscribe(event_type, self._receive)

    # -- physical state (delegated to the metabolism) -------------------------
    @property
    def load(self) -> float:
        return self.metabolism.load

    @property
    def fidelity(self) -> float:
        """Information fidelity in [0, 1]; 1 = unimpaired, lower = noisier."""
        return self.metabolism.fidelity()

    @property
    def action_capacity(self) -> float:
        """Size of the currently reachable action manifold in [0, 1]."""
        return self.metabolism.capacity()

    def work(self, action: Action, scale: float = 1.0) -> bool:
        """Perform an action: spend energy and (thermodynamically) make heat."""
        return self.metabolism.do_work(action, scale)

    def can_afford(self, action: Action, scale: float = 1.0) -> bool:
        """An action is available only if it is both affordable (energy) *and*
        reachable (not squeezed out of the action manifold by metabolic load)."""
        return (self.metabolism.can_afford(action, scale)
                and self.metabolism.reachable(action))

    def recover(self, reservoir: EnergyReservoir | None, sleeping: bool,
                dt: float = 1.0) -> None:
        self.metabolism.recover(reservoir, sleeping, dt)

    # -- bus plumbing ---------------------------------------------------------
    def _receive(self, event: Event) -> None:
        """Bus handler: queue foreign events and strengthen the sender link."""
        if event.source == self.name:
            return
        self._inbox.append(event)
        self.connections[event.source] = self.connections.get(event.source, 0.0) + 0.05

    def emit(self, event: Event) -> None:
        self.bus.publish(event)

    def detach(self) -> None:
        """Unsubscribe from the bus (used when a module is pruned)."""
        for event_type in self.subscriptions:
            self.bus.unsubscribe(event_type, self._receive)

    def drain_inbox(self) -> list[Event]:
        events = list(self._inbox)
        self._inbox.clear()
        return events

    # -- scheduling signals ---------------------------------------------------
    @property
    def sleeping(self) -> bool:
        return self.lifecycle.sleeping

    @property
    def awake(self) -> bool:
        return self.lifecycle.awake

    def wants_to_act(self) -> bool:
        """The module's *drive* to act (inbox load and any spontaneous urge),
        independent of whether it currently *can*."""
        return len(self._inbox) > 0

    def has_work(self) -> bool:
        """Actionable work = a drive to act *and* the minimal cognitive action
        (THINK) still reachable.  When load collapses the manifold below THINK
        the module has nothing it can do, so it becomes eligible for sleep --
        rest emerges from action-space collapse, not from any load->sleep rule."""
        return self.wants_to_act() and self.metabolism.reachable(Action.THINK)

    def urgency(self) -> float:
        load = min(1.0, len(self._inbox) / 3.0)
        drive = 0.3 if self.entropy.is_high else 0.0
        return min(1.0, load + drive)

    # -- template method executed by the kernel -------------------------------
    def step(self, tick: int) -> "ModuleReport":
        observations = self.observe()
        thoughts: list[Event] = []
        acted = False
        if self.can_afford(Action.THINK):
            self.work(Action.THINK)
            thoughts = self.think(observations)
            acted = True
        if thoughts and self.can_afford(Action.LEARN):
            self.work(Action.LEARN)
            self.learn(thoughts)
        self.communicate(thoughts)
        self.regulate()
        # A module that could not even think this tick "stalls": it does not
        # refresh its activity, so its idle timer grows and it drifts toward
        # sleep.  Overload therefore ends in rest by collapsing the action set.
        if acted:
            self.lifecycle.mark_active(tick)
        return self.report()

    def regulate(self) -> None:
        """Homeostasis: let the metabolism couple load->entropy and relax it,
        update the derived temperature, and (under low fidelity) let information
        entropy miscalibrate confidence toward chance."""
        self.metabolism.regulate()
        self.temperature = 0.5 * self.temperature + 0.5 * self.entropy.temperature
        fidelity = self.fidelity
        if fidelity < 1.0:
            self.confidence += (0.5 - self.confidence) * (1.0 - fidelity) * 0.3

    # -- lifecycle hooks ------------------------------------------------------
    def communicate(self, events: list[Event]) -> None:
        for event in events:
            if self.work(Action.COMMUNICATE):
                self.emit(event)

    def sleep(self, tick: int) -> None:
        if self.lifecycle.transition_to(LifecycleState.SLEEPING, tick):
            self.emit(SleepEvent(source=self.name, payload={"tick": tick}))

    def wake(self, tick: int) -> None:
        if self.lifecycle.transition_to(LifecycleState.ACTIVE, tick):
            self.emit(WakeEvent(source=self.name, payload={"tick": tick}))

    def report(self) -> "ModuleReport":
        return ModuleReport(
            name=self.name,
            state=self.lifecycle.state.value,
            energy=self.energy.level,
            entropy=self.entropy.level,
            temperature=self.temperature,
            confidence=self.confidence,
            importance=self.importance,
            last_active=self.lifecycle.last_active,
            inbox=len(self._inbox),
            load=self.load,
            fidelity=self.fidelity,
        )

    # -- abstract cognition ---------------------------------------------------
    @abstractmethod
    def observe(self) -> list[Event]:
        """Gather input (typically ``self.drain_inbox()`` plus any sensing)."""

    @abstractmethod
    def think(self, observations: list[Event]) -> list[Event]:
        """Process observations and return events to publish."""

    @abstractmethod
    def learn(self, thoughts: list[Event]) -> None:
        """Update internal state / confidence from what was just produced."""


from dataclasses import dataclass  # placed after the class to keep imports tidy


@dataclass(slots=True)
class ModuleReport:
    name: str
    state: str
    energy: float
    entropy: float
    temperature: float
    confidence: float
    importance: float
    last_active: int
    inbox: int
    load: float = 0.0
    fidelity: float = 1.0

    def as_dict(self, tick: int = 0) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "energy": self.energy,
            "entropy": self.entropy,
            "temperature": self.temperature,
            "confidence": self.confidence,
            "importance": self.importance,
            "idle": max(0, tick - self.last_active),
            "inbox": self.inbox,
            "load": self.load,
            "fidelity": self.fidelity,
        }
