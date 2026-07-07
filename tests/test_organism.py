import logging

from main import build_organism
from utils.config import Config


def _small_config() -> Config:
    config = Config.default()
    config.ticks = 30
    config.organism.tick_interval = 0.0
    return config


def test_organism_ticks_without_error_and_records_metrics():
    organism = build_organism(_small_config(), logging.getLogger("test"))
    for _ in range(30):
        snapshot = organism.tick()
    assert len(organism.metrics) == 30
    assert snapshot["tick"] == 30
    assert 0.0 <= snapshot["health"] <= 1.0


def test_organism_processes_events_and_forms_memories():
    organism = build_organism(_small_config(), logging.getLogger("test"))
    for _ in range(30):
        organism.tick()
    assert organism.bus.processed > 0
    assert organism.memory.count > 0


def test_core_modules_survive_a_run():
    organism = build_organism(_small_config(), logging.getLogger("test"))
    for _ in range(30):
        organism.tick()
    names = {m.name for m in organism.modules}
    assert {"perception", "curiosity", "language", "planner", "critic"} <= names


def test_spawn_and_retire_module():
    organism = build_organism(_small_config(), logging.getLogger("test"))
    module = organism.spawn_module("hypothesis")
    assert module is not None and module in organism.modules
    assert organism.retire_module(module.name) is True
    assert module not in organism.modules


def test_structural_adaptation_merges_idle_dynamic_modules():
    organism = build_organism(_small_config(), logging.getLogger("test"))
    a = organism.spawn_module("hypothesis")
    b = organism.spawn_module("hypothesis")
    assert a is not None and b is not None
    a.lifecycle.last_active = 0
    b.lifecycle.last_active = 0
    before = len(organism.modules)
    changes = organism.adapter.adapt(organism, tick=20)  # idle 20 >= merge_idle 12
    assert any(c.startswith("merge") for c in changes)
    assert len(organism.modules) == before - 1
