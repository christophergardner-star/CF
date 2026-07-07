"""Module discovery and construction -- the plugin seam.

New modules become usable by the kernel without touching it: either register a
builtin key here or point a :class:`~utils.config.ModuleConfig` ``type`` at a
dotted ``package.module:ClassName`` path.  The :class:`ModuleFactory` performs
dependency injection (bus, freshly built energy/entropy budgets and any
per-type extras such as perception's sensor), keeping construction in one place.
"""
from __future__ import annotations

import importlib

from core.energy import EnergyBudget
from core.entropy import EntropyField
from core.events import EventBus
from core.metabolism import build_metabolism
from core.module import Module
from modules.critic import CriticModule
from modules.curiosity import CuriosityModule
from modules.hypothesis import HypothesisModule
from modules.language import LanguageModule
from modules.perception import PerceptionModule
from modules.planner import PlannerModule
from utils.config import EnergyConfig, EntropyConfig, ModuleConfig

BUILTIN_MODULES: dict[str, type[Module]] = {
    "perception": PerceptionModule,
    "curiosity": CuriosityModule,
    "language": LanguageModule,
    "planner": PlannerModule,
    "critic": CriticModule,
    "hypothesis": HypothesisModule,
}


def resolve(type_name: str) -> type[Module]:
    """Resolve a module type to a class (builtin key or dotted path)."""
    if ":" in type_name:
        module_path, _, class_name = type_name.partition(":")
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    elif type_name in BUILTIN_MODULES:
        cls = BUILTIN_MODULES[type_name]
    else:
        raise KeyError(f"Unknown module type: {type_name!r}")
    if not issubclass(cls, Module):
        raise TypeError(f"{type_name!r} does not resolve to a Module subclass")
    return cls


class ModuleFactory:
    def __init__(self, bus: EventBus, injectors: dict[str, dict] | None = None,
                 metabolism: str = "classic", thermo_params: dict | None = None,
                 seed: int = 0) -> None:
        self.bus = bus
        #: per-type extra keyword arguments (e.g. {"perception": {"sensor": fn}})
        self.injectors = injectors or {}
        self.metabolism = metabolism            # "classic" | "thermodynamic"
        self.thermo_params = thermo_params or {}
        self.seed = seed

    def build(self, spec: ModuleConfig) -> Module:
        cls = resolve(spec.type)
        energy = EnergyBudget(spec.energy.capacity, spec.energy.level, spec.energy.regen_rate)
        entropy = EntropyField(spec.entropy.level, spec.entropy.baseline,
                               spec.entropy.relaxation)
        metabolism = build_metabolism(self.metabolism, energy, entropy, self.thermo_params)
        options = dict(spec.options)
        options.setdefault("seed", self.seed)
        extra = dict(self.injectors.get(spec.type, {}))
        return cls(spec.name, self.bus, metabolism=metabolism,
                   base_priority=spec.base_priority, importance=spec.importance,
                   idle_sleep_ticks=spec.idle_sleep_ticks, options=options,
                   **extra)

    def spawn(self, kind: str, name: str) -> Module:
        """Build a dynamic module with reasonable defaults (used by adaptation)."""
        spec = ModuleConfig(
            name=name, type=kind, base_priority=0.55, importance=0.4,
            idle_sleep_ticks=8,
            energy=EnergyConfig(32.0, 32.0, 2.6),
            entropy=EntropyConfig(level=0.6, baseline=0.4, relaxation=0.05))
        return self.build(spec)
