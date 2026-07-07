"""Experiment scaffolding: build an organism, observe it, measure it.

These are thin helpers so the actual experiments in ``run_experiment.py`` read
like a lab protocol rather than plumbing.  A "condition" is (config +
perturbations); running it yields an :class:`~lab.observatory.Observatory` (the
raw record) and the organism (for graph/structure measures).
"""
from __future__ import annotations

import logging
from typing import Iterable

from lab import metrics
from lab.observatory import Observatory
from main import build_organism
from utils.config import Config

_QUIET = logging.getLogger("lab")
_QUIET.addHandler(logging.NullHandler())
_QUIET.propagate = False


def fresh_config(ticks: int, seed: int = 0) -> Config:
    config = Config.default()
    config.ticks = ticks
    config.seed = seed
    config.organism.tick_interval = 0.0
    return config


def thermo_config(ticks: int, seed: int = 0) -> Config:
    """Organism B: the coupled three-field thermodynamic substrate."""
    config = fresh_config(ticks, seed)
    config.thermo.enabled = True
    return config


def unlimited_config(ticks: int, seed: int = 0) -> Config:
    """Organism A: classic economics with an effectively unlimited energy feed
    and no metabolic load -- the control for the constraint experiment."""
    config = fresh_config(ticks, seed)
    for spec in config.modules:
        spec.energy.regen_rate = 1000.0
        spec.energy.level = spec.energy.capacity
    return config


def run_condition(config: Config, ticks: int,
                  perturbations: Iterable | None = None) -> tuple[Observatory, object]:
    organism = build_organism(config, _QUIET)
    observatory = Observatory(organism)
    observatory.run(ticks, perturbations)
    return observatory, organism


def measure(config: Config, ticks: int,
            perturbations: Iterable | None = None) -> dict:
    observatory, organism = run_condition(config, ticks, perturbations)
    return metrics.full_report(observatory.frames, organism, observatory)


def window(observatory: Observatory, lo: int, hi: int):
    """Frames whose tick is in ``[lo, hi)`` -- for pre/post comparisons."""
    return [f for f in observatory.frames if lo <= f.tick < hi]
