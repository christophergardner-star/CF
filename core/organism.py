"""The Organism -- the kernel that owns everything and drives the tick loop.

The organism is deliberately thin: it holds the shared resources (energy,
entropy, health, age), the event bus, the scheduler, the memory manager and the
list of modules, and it sequences one biological "tick".  All intelligence lives
in the modules; the organism just keeps them alive, fed and scheduled, exactly
like an operating-system kernel for continual intelligence.

Tick order (as specified):
    observe -> process events -> update energy -> update entropy ->
    wake modules -> run modules -> consolidate memory -> allow learning ->
    allow structural adaptation -> sleep modules -> record metrics
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Protocol

from core.adaptation import StructuralAdapter
from core.energy import EnergyBudget
from core.entropy import EntropyField
from core.events import (CuriosityEvent, Event, EventBus, LearningEvent,
                         ObservationEvent)
from core.metabolism import EnergyReservoir
from core.module import Module
from core.scheduler import Scheduler
from memory.manager import MemoryManager
from utils.metrics import Metrics


class World(Protocol):
    def step(self) -> None: ...


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class Organism:
    def __init__(self, *, name: str, bus: EventBus, scheduler: Scheduler,
                 memory: MemoryManager, energy: EnergyBudget, entropy: EntropyField,
                 metrics: Metrics, adapter: StructuralAdapter,
                 module_factory: Any, logger: logging.Logger | None = None,
                 modules: list[Module] | None = None, world: World | None = None,
                 reservoir: EnergyReservoir | None = None, thermo: bool = False,
                 diffusion: float = 0.05,
                 run_cost: float = 1.0, tick_interval: float = 0.35) -> None:
        self.name = name
        self.bus = bus
        self.scheduler = scheduler
        self.memory = memory
        self.energy = energy
        self.entropy = entropy
        self.metrics = metrics
        self.adapter = adapter
        self.factory = module_factory
        self.log = logger or logging.getLogger("organism")
        self.modules: list[Module] = list(modules or [])
        self.world = world
        self.reservoir = reservoir or EnergyReservoir()
        self.thermo = thermo
        self.diffusion = diffusion
        self.run_cost = run_cost
        self.tick_interval = tick_interval

        self.age = 0
        self.health = 1.0
        self.learning_rate = 0.0
        self._curiosity = 0.3
        self._learn_count = 0
        self._spawn_counter = 0
        self.last_influx = 0.0

    # -- registration ---------------------------------------------------------
    def register(self, module: Module) -> None:
        self.modules.append(module)

    # -- the tick -------------------------------------------------------------
    def tick(self) -> dict[str, Any]:
        self.age += 1
        tick = self.age
        self._learn_count = 0

        self._observe()                              # 1. observe
        drained = self.bus.pump()                    # 2. process events
        self._ingest(drained)
        self._update_energy()                        # 3. update energy
        self._update_entropy(drained)                # 4. update entropy

        decision = self.scheduler.plan(self.modules, self.energy, tick, self.run_cost)
        for module in decision.to_wake:              # 5. wake required modules
            module.wake(tick)

        for module in decision.to_run:               # 6. run active modules
            if not self.energy.spend(self.run_cost):
                break
            module.step(tick)

        idle = len(decision.to_run) <= 1             # 7. consolidate memories
        self.memory.consolidate(idle=idle, fidelity=self._mean_fidelity())

        self._ingest(self.bus.pump())                # 8. allow learning to flow

        changes = self.adapter.adapt(self, tick)     # 9. structural adaptation
        if changes:
            self.bus.pump()

        for module in decision.to_sleep:             # 10. sleep inactive modules
            module.sleep(tick)

        self.learning_rate = 0.8 * self.learning_rate + 0.2 * self._learn_count
        snapshot = self.snapshot(len(decision.to_run))
        self.metrics.record(snapshot)                # 11. record metrics
        return snapshot

    async def run(self, ticks: int,
                  on_tick: Callable[["Organism"], None] | None = None) -> None:
        for _ in range(ticks):
            self.tick()
            if on_tick is not None:
                on_tick(self)
            await asyncio.sleep(self.tick_interval)

    # -- tick helpers ---------------------------------------------------------
    def _observe(self) -> None:
        if self.world is not None:
            self.world.step()

    def _ingest(self, events: list[Event]) -> None:
        for event in events:
            if isinstance(event, LearningEvent):
                self._learn_count += 1
            elif isinstance(event, CuriosityEvent):
                signal = float(event.get("curiosity", event.priority))
                self._curiosity = 0.8 * self._curiosity + 0.2 * signal

    def _update_energy(self) -> None:
        # The organism's scheduling budget (attention/compute) regenerates.
        self.energy.regenerate()
        # Metabolic recovery: replenish the shared free-energy reservoir from the
        # environment, then let each module draw from it and cool down.  In
        # classic mode the reservoir is ignored and modules regenerate freely.
        self.last_influx = self.reservoir.replenish() if self.thermo else 0.0
        source = self.reservoir if self.thermo else None
        for module in self.modules:
            module.recover(source, module.sleeping)
        if self.thermo:
            self._diffuse_load()

    def _diffuse_load(self) -> None:
        """Metabolic load diffuses locally: a hot module heats the modules it is
        wired to (heat flows along the learned communication pathways)."""
        if len(self.modules) < 2:
            return
        index = {m.name: m for m in self.modules}
        deltas: dict[Module, float] = {}
        for module in self.modules:
            load = module.metabolism.load
            if load <= 0.0:
                continue
            neighbours = [(index[n], w) for n, w in module.connections.items()
                          if n in index and index[n] is not module]
            total = sum(w for _, w in neighbours)
            if total <= 0.0:
                continue
            outflow = load * self.diffusion
            for target, weight in neighbours:
                deltas[target] = deltas.get(target, 0.0) + outflow * (weight / total)
            deltas[module] = deltas.get(module, 0.0) - outflow
        for module, delta in deltas.items():
            module.metabolism.receive_load(delta)

    def _mean_fidelity(self) -> float:
        if not self.modules:
            return 1.0
        return sum(m.fidelity for m in self.modules) / len(self.modules)

    def metabolic_total(self) -> float:
        """Total energy currently inside the system (for conservation checks)."""
        return self.reservoir.level + sum(m.energy.level + m.metabolism.load
                                          for m in self.modules)

    def total_dissipated(self) -> float:
        return sum(m.metabolism.dissipated for m in self.modules)

    def _update_entropy(self, drained: list[Event]) -> None:
        if self.modules:
            mean = sum(m.entropy.level for m in self.modules) / len(self.modules)
        else:
            mean = self.entropy.baseline
        self.entropy.level = _clamp(0.7 * self.entropy.level + 0.3 * mean)
        novelties = [float(e.get("novelty", 0.0)) for e in drained
                     if isinstance(e, ObservationEvent)]
        if novelties:
            self.entropy.perturb(max(novelties) * 0.2)
        self.entropy.relax(0.5)

    # -- AdaptableHost protocol (used by the structural adapter) --------------
    def curiosity_pressure(self) -> float:
        return self._curiosity

    def spawn_module(self, kind: str) -> Module | None:
        self._spawn_counter += 1
        name = f"{kind}-{self._spawn_counter}"
        try:
            module = self.factory.spawn(kind, name)
        except Exception as exc:  # pragma: no cover - defensive
            self.log.warning("spawn failed for %s: %s", kind, exc)
            return None
        module.wake(self.age)
        self.register(module)
        return module

    def retire_module(self, name: str) -> bool:
        for module in self.modules:
            if module.name == name:
                module.detach()
                self.modules.remove(module)
                return True
        return False

    def announce(self, event: Event) -> None:
        self.bus.publish(event)

    # -- readouts -------------------------------------------------------------
    def snapshot(self, active_count: int | None = None) -> dict[str, Any]:
        active = sum(1 for m in self.modules if m.awake)
        sleeping = sum(1 for m in self.modules if m.sleeping)
        balance = 1.0 - min(1.0, abs(self.entropy.level - self.entropy.baseline) * 2)
        mean_load = (sum(m.load for m in self.modules) / len(self.modules)
                     if self.modules else 0.0)
        fidelity = self._mean_fidelity()
        self.health = _clamp(0.55 * self.energy.fraction + 0.25 * balance
                             + 0.2 * fidelity)
        return {
            "name": self.name,
            "tick": self.age,
            "age": self.age,
            "energy": self.energy.level,
            "energy_fraction": self.energy.fraction,
            "entropy": self.entropy.level,
            "curiosity": self._curiosity,
            "learning_rate": self.learning_rate,
            "health": self.health,
            "active": active if active_count is None else active,
            "sleeping": sleeping,
            "events": self.bus.processed,
            "memories": self.memory.count,
            "thermo": self.thermo,
            "load": mean_load,
            "fidelity": fidelity,
            "reservoir": self.reservoir.level if self.thermo else 0.0,
            "metabolic_total": self.metabolic_total() if self.thermo else 0.0,
        }

    def reports(self) -> list[dict[str, Any]]:
        return [m.report().as_dict(self.age) for m in self.modules]
