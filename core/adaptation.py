"""Structural adaptation -- the organism reshaping itself over time.

No neural networks here: adaptation is *simulated* at the module-graph level.
The adapter can

* re-weight communication links (Hebbian weights decay; strong links are logged);
* spawn a new module when curiosity stays high (a fresh "hypothesis" organ);
* prune dynamic modules that have gone idle for too long; and
* merge redundant dynamic modules.

Every structural change is logged and announced as a :class:`StructuralEvent`.
The adapter talks to the organism only through the small ``AdaptableHost``
protocol, so it never depends on the concrete kernel.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from core.events import Event, StructuralEvent
from core.module import Module

if TYPE_CHECKING:  # pragma: no cover
    pass


@runtime_checkable
class AdaptableHost(Protocol):
    """The slice of the organism the adapter is allowed to touch."""

    modules: list[Module]

    def curiosity_pressure(self) -> float: ...
    def spawn_module(self, kind: str) -> Module | None: ...
    def retire_module(self, name: str) -> bool: ...
    def announce(self, event: Event) -> None: ...


class StructuralAdapter:
    #: modules whose name starts with this prefix are considered "dynamic" and
    #: are therefore eligible for pruning/merging (the core stack never is).
    DYNAMIC_PREFIX = "hypothesis"

    def __init__(self, enabled: bool = True, prune_idle_ticks: int = 35,
                 spawn_curiosity: float = 0.7, max_dynamic: int = 3,
                 weight_decay: float = 0.02, merge_idle: int = 12,
                 cooldown: int = 8, logger: logging.Logger | None = None) -> None:
        self.enabled = enabled
        self.prune_idle_ticks = prune_idle_ticks
        self.spawn_curiosity = spawn_curiosity
        self.max_dynamic = max_dynamic
        self.weight_decay = weight_decay
        self.merge_idle = merge_idle
        self.cooldown = cooldown
        self.log = logger or logging.getLogger("adaptation")
        self._last_spawn = -(10 ** 9)

    def adapt(self, host: AdaptableHost, tick: int) -> list[str]:
        if not self.enabled:
            return []
        changes: list[str] = []
        self._reweight(host, tick, changes)
        self._merge(host, tick, changes)
        self._prune(host, tick, changes)
        self._spawn(host, tick, changes)
        return changes

    # -- Hebbian re-weighting -------------------------------------------------
    def _reweight(self, host: AdaptableHost, tick: int, changes: list[str]) -> None:
        strongest: tuple[str, str, float] | None = None
        for module in host.modules:
            for source, weight in list(module.connections.items()):
                weight *= (1.0 - self.weight_decay)
                if weight < 0.01:
                    del module.connections[source]
                else:
                    module.connections[source] = weight
                    if strongest is None or weight > strongest[2]:
                        strongest = (source, module.name, weight)
        # Occasionally log the dominant pathway as a structural signal.
        if strongest and tick % 20 == 0 and strongest[2] > 0.5:
            src, dst, w = strongest
            msg = f"reweight {src}->{dst} = {w:.2f}"
            changes.append(msg)
            host.announce(StructuralEvent(source="adaptation",
                                          payload={"change": "reweight", "edge": [src, dst],
                                                   "weight": w}))

    # -- merging --------------------------------------------------------------
    def _merge(self, host: AdaptableHost, tick: int, changes: list[str]) -> None:
        dynamic = [m for m in host.modules
                   if m.name.startswith(self.DYNAMIC_PREFIX)
                   and m.lifecycle.idle_for(tick) >= self.merge_idle]
        if len(dynamic) < 2:
            return
        dynamic.sort(key=lambda m: m.importance, reverse=True)
        survivor, absorbed = dynamic[0], dynamic[1]
        # Fold the absorbed module's learned links + importance into the survivor.
        for source, weight in absorbed.connections.items():
            survivor.connections[source] = survivor.connections.get(source, 0.0) + weight
        survivor.importance = min(1.0, survivor.importance + 0.1)
        survivor.lifecycle.last_active = tick  # spare the survivor from pruning
        if host.retire_module(absorbed.name):
            msg = f"merge {absorbed.name} -> {survivor.name}"
            self.log.info("[adaptation] %s", msg)
            changes.append(msg)
            host.announce(StructuralEvent(
                source="adaptation",
                payload={"change": "merge", "from": absorbed.name,
                         "into": survivor.name}))

    # -- pruning --------------------------------------------------------------
    def _prune(self, host: AdaptableHost, tick: int, changes: list[str]) -> None:
        for module in list(host.modules):
            if not module.name.startswith(self.DYNAMIC_PREFIX):
                continue
            if module.lifecycle.idle_for(tick) >= self.prune_idle_ticks:
                if host.retire_module(module.name):
                    msg = f"prune {module.name} (idle {module.lifecycle.idle_for(tick)})"
                    self.log.info("[adaptation] %s", msg)
                    changes.append(msg)
                    host.announce(StructuralEvent(
                        source="adaptation",
                        payload={"change": "prune", "module": module.name}))

    # -- spawning -------------------------------------------------------------
    def _spawn(self, host: AdaptableHost, tick: int, changes: list[str]) -> None:
        dynamic = sum(1 for m in host.modules
                      if m.name.startswith(self.DYNAMIC_PREFIX))
        if dynamic >= self.max_dynamic:
            return
        if tick - self._last_spawn < self.cooldown:
            return
        if host.curiosity_pressure() < self.spawn_curiosity:
            return
        module = host.spawn_module(self.DYNAMIC_PREFIX)
        if module is None:
            return
        self._last_spawn = tick
        msg = f"spawn {module.name} (curiosity {host.curiosity_pressure():.2f})"
        self.log.info("[adaptation] %s", msg)
        changes.append(msg)
        host.announce(StructuralEvent(source="adaptation",
                                      payload={"change": "spawn", "module": module.name}))
