"""The scheduler: decides which modules run, wake or sleep each tick.

The decision blends several signals -- a module's base priority and importance,
how urgently it wants to run (inbox load + entropy drive), how recently it was
active, and how much energy it has -- and then gates the ranked list against the
organism's remaining energy budget (never spending below a reserve).  Modules
with nothing useful to do are allowed to sleep and recover.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.energy import EnergyBudget
from core.module import Module


@dataclass(slots=True)
class ScheduleDecision:
    to_run: list[Module] = field(default_factory=list)
    to_wake: list[Module] = field(default_factory=list)
    to_sleep: list[Module] = field(default_factory=list)


class Scheduler:
    def __init__(self, energy_reserve: float = 0.12, wake_threshold: float = 0.35,
                 sleep_after_idle: int = 6, max_active: int | None = None,
                 logger: logging.Logger | None = None) -> None:
        self.energy_reserve = energy_reserve
        self.wake_threshold = wake_threshold
        self.sleep_after_idle = sleep_after_idle
        self.max_active = max_active
        self.log = logger or logging.getLogger("scheduler")

    def score(self, module: Module, tick: int) -> float:
        recency = 1.0 / (1.0 + module.lifecycle.idle_for(tick))
        entropy_bonus = 0.5 * module.entropy.level
        energy_factor = 0.4 + 0.6 * module.energy.fraction
        return (module.base_priority * 0.9
                + module.importance * 0.4
                + module.urgency() * 1.2
                + entropy_bonus
                + recency * 0.3) * energy_factor

    def plan(self, modules: list[Module], organism_energy: EnergyBudget,
             tick: int, run_cost: float) -> ScheduleDecision:
        decision = ScheduleDecision()

        for module in modules:
            has_work = module.has_work()
            if module.sleeping:
                if (has_work and not module.energy.is_low(0.15)
                        and self.score(module, tick) >= self.wake_threshold):
                    decision.to_wake.append(module)
            else:  # awake
                idle = module.lifecycle.idle_for(tick) >= self.sleep_after_idle
                if module.energy.is_low(0.1) or (not has_work and idle):
                    decision.to_sleep.append(module)

        sleeping = set(id(m) for m in decision.to_sleep)
        waking = set(id(m) for m in decision.to_wake)
        candidates = [m for m in modules
                      if id(m) not in sleeping and (m.awake or id(m) in waking)]
        candidates.sort(key=lambda m: self.score(m, tick), reverse=True)

        reserve = organism_energy.capacity * self.energy_reserve
        available = organism_energy.level - reserve
        for module in candidates:
            if available < run_cost:
                break
            if module.energy.is_low(0.08):
                continue
            decision.to_run.append(module)
            available -= run_cost
            if self.max_active and len(decision.to_run) >= self.max_active:
                break

        return decision
