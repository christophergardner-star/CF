"""Typed configuration for the whole organism.

Configuration is plain data (nested dataclasses) with sane defaults.  It is the
single place ``main.py`` reads from when wiring the kernel together, which keeps
construction explicit and dependency-injected rather than magical.  Nothing in
``core`` imports this module -- config flows *in* as primitives, so the kernel
stays independent of how it is configured.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EnergyConfig:
    capacity: float = 100.0
    level: float = 100.0
    regen_rate: float = 3.0


@dataclass
class EntropyConfig:
    level: float = 0.5
    baseline: float = 0.3
    relaxation: float = 0.05


@dataclass
class ModuleConfig:
    """Description of a single module to build. ``type`` is a builtin key
    (see :mod:`modules.registry`) or a dotted ``package.module:Class`` path."""

    name: str
    type: str
    base_priority: float = 0.5
    importance: float = 0.5
    idle_sleep_ticks: int = 6
    energy: EnergyConfig = field(default_factory=lambda: EnergyConfig(32.0, 32.0, 2.6))
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    working_capacity: int = 7
    episodic_capacity: int = 512
    replay_k: int = 3
    promote_threshold: float = 3.0


@dataclass
class SchedulerConfig:
    energy_reserve: float = 0.12
    wake_threshold: float = 0.35
    sleep_after_idle: int = 6
    max_active: int | None = None


@dataclass
class AdaptationConfig:
    enabled: bool = True
    prune_idle_ticks: int = 35
    spawn_curiosity: float = 0.7
    max_dynamic: int = 3
    weight_decay: float = 0.02
    merge_idle: int = 12


@dataclass
class ThermoConfig:
    """The thermodynamic substrate.  Off by default: the classic economics is
    the characterised control (Organism A); turning this on gives the coupled
    three-field physics (Organism B)."""

    enabled: bool = False
    reservoir_capacity: float = 1500.0
    reservoir_level: float = 1500.0
    influx: float = 16.0
    uptake: float = 6.0
    heat_fraction: float = 0.6
    load_decay: float = 0.08
    load_to_entropy: float = 0.6
    fidelity_gain: float = 0.85
    action_ceiling: float = 0.9
    diffusion: float = 0.05

    def params(self) -> dict[str, float]:
        return {"heat_fraction": self.heat_fraction, "load_decay": self.load_decay,
                "uptake": self.uptake, "load_to_entropy": self.load_to_entropy,
                "fidelity_gain": self.fidelity_gain, "action_ceiling": self.action_ceiling}


@dataclass
class OrganismConfig:
    name: str = "kernel-0"
    energy: EnergyConfig = field(default_factory=lambda: EnergyConfig(260.0, 260.0, 7.0))
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    run_cost: float = 1.0
    tick_interval: float = 0.35


@dataclass
class Config:
    organism: OrganismConfig = field(default_factory=OrganismConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    thermo: ThermoConfig = field(default_factory=ThermoConfig)
    modules: list[ModuleConfig] = field(default_factory=list)
    ticks: int = 80
    seed: int = 7

    @classmethod
    def default(cls) -> "Config":
        return cls(modules=default_modules())

    @classmethod
    def thermodynamic(cls) -> "Config":
        """Default stack running on the thermodynamic substrate (Organism B)."""
        config = cls.default()
        config.thermo.enabled = True
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_modules() -> list[ModuleConfig]:
    """The default cognitive stack: perception -> curiosity -> language ->
    planner -> critic, forming a closed explore/learn loop."""
    return [
        ModuleConfig(name="perception", type="perception",
                     base_priority=0.85, importance=0.7),
        ModuleConfig(name="curiosity", type="curiosity",
                     base_priority=0.6, importance=0.6,
                     entropy=EntropyConfig(level=0.6, baseline=0.45)),
        ModuleConfig(name="language", type="language",
                     base_priority=0.55, importance=0.6),
        ModuleConfig(name="planner", type="planner",
                     base_priority=0.5, importance=0.65),
        ModuleConfig(name="critic", type="critic",
                     base_priority=0.5, importance=0.7),
    ]
