"""Tests for the STRATA cortex organ and its coupling to the kernel."""
import asyncio
import logging

import pytest

np = pytest.importorskip("numpy")

from core.energy import EnergyBudget
from core.entropy import EntropyField
from core.events import EventBus, LearningEvent, ObservationEvent, ThoughtEvent
from core.metabolism import ThermodynamicMetabolism
from modules.strata_cortex import StrataCortexModule
from strata.network import StrataConfig, StrataNetwork


def make_cortex() -> tuple[EventBus, StrataCortexModule]:
    bus = EventBus()
    module = StrataCortexModule(
        "cortex", bus,
        energy=EnergyBudget(capacity=1000.0, level=1000.0, regen_rate=50.0))
    return bus, module


def drive(bus: EventBus, module: StrataCortexModule, tokens: list[str]) -> None:
    for t, token in enumerate(tokens):
        bus.publish(ObservationEvent(source="perception",
                                     payload={"token": token}))
        bus.pump()
        module.step(t)
        # emulate the kernel's recovery phase (organism._update_energy);
        # without it the module starves and -- correctly -- stops thinking
        module.recover(None, module.sleeping)


def test_cortex_learns_to_predict_a_cycle():
    bus, module = make_cortex()
    pattern = ["sun", "rain", "wind", "snow"]
    drive(bus, module, [pattern[t % 4] for t in range(300)])

    before_hits, before_all = (module.state["hits"],
                               module.state["hits"] + module.state["misses"])
    drive(bus, module, [pattern[t % 4] for t in range(300, 400)])
    hits = module.state["hits"] - before_hits
    scored = module.state["hits"] + module.state["misses"] - before_all
    assert scored > 0
    assert hits / scored > 0.6            # far above the 1/4 chance level


def test_cortex_narrates_and_announces_growth():
    bus, module = make_cortex()
    heard: list[str] = []
    bus.subscribe(ThoughtEvent, lambda e: heard.append("thought"))
    bus.subscribe(LearningEvent, lambda e: heard.append("learning"))

    drive(bus, module, [["a", "b", "c"][t % 3] for t in range(60)])
    bus.pump()                            # deliver the cortex's own events
    assert "thought" in heard             # it narrates predictions
    assert "learning" in heard            # bootstrap column growth announced


def test_zero_plasticity_freezes_slow_weights_but_not_fast():
    net = StrataNetwork(StrataConfig(d_in=24, code_dim=128, seed=0))
    rng = np.random.default_rng(0)
    stream = rng.standard_normal((4, 24))
    stream /= np.linalg.norm(stream, axis=1, keepdims=True)
    for t in range(80):
        net.step(stream[t % 4], plasticity=1.0)

    # cascade diffusion may move mass between levels, but with plasticity 0
    # no new information may enter: per-chain level sums are conserved
    def chain_sums():
        return [
            (col.C.levels.sum(), col.D.levels.sum())
            for col in net.router.columns
        ]

    before = chain_sums()
    fast_before = net.fast.F.copy()
    for t in range(80, 160):
        net.step(stream[t % 4] + 0.5 * rng.standard_normal(24),
                 plasticity=0.0)
    for (c0, d0), (c1, d1) in zip(before, chain_sums()):
        assert np.isclose(c0, c1)
        assert np.isclose(d0, d1)
    assert not np.allclose(fast_before, net.fast.F)   # episodic capture lives


def test_network_reports_synaptic_heat():
    net = StrataNetwork(StrataConfig(d_in=24, code_dim=128, seed=0))
    rng = np.random.default_rng(1)
    stream = rng.standard_normal((4, 24))
    stream /= np.linalg.norm(stream, axis=1, keepdims=True)

    heats = [net.step(stream[t % 4], plasticity=1.0)["heat"]
             for t in range(60)]
    assert max(heats) > 0.0                       # learning does work

    # a converged, predictable stream at zero plasticity does no synaptic
    # work at all: no slow updates, and nothing novel for the fast weights
    calm = [net.step(stream[t % 4], plasticity=0.0)["heat"]
            for t in range(60, 100)]
    assert max(calm[5:]) == 0.0


def test_thermo_cortex_learning_generates_load_and_throttles():
    """The closed loop: plasticity -> heat -> load -> fidelity -> plasticity."""
    bus = EventBus()
    metabolism = ThermodynamicMetabolism(
        EnergyBudget(capacity=32.0, level=32.0, regen_rate=0.0),
        EntropyField())
    module = StrataCortexModule("cortex", bus, metabolism=metabolism)

    rng = np.random.default_rng(2)
    max_load, min_fidelity = 0.0, 1.0
    for t in range(200):
        # a volatile stream (fresh symbols) keeps prediction error high,
        # so the substrate keeps doing heavy synaptic work
        token = f"sym-{rng.integers(0, 12)}"
        bus.publish(ObservationEvent(source="perception",
                                     payload={"token": token}))
        bus.pump()
        module.step(t)
        module.recover(None, module.sleeping)
        max_load = max(max_load, module.load)
        min_fidelity = min(min_fidelity, module.fidelity)

    assert max_load > 1.0                 # learning heated the module
    assert min_fidelity < 0.9             # ... which degraded fidelity,
    #                                       throttling its own plasticity


def test_organism_runs_with_cortex():
    from main import build_organism
    from utils.config import Config, ModuleConfig

    config = Config.default()
    config.organism.tick_interval = 0.0
    config.modules.append(ModuleConfig(
        name="cortex", type="modules.strata_cortex:StrataCortexModule",
        base_priority=0.6, importance=0.65))
    organism = build_organism(config, logging.getLogger("test"))
    asyncio.run(organism.run(30))

    cortex = next(m for m in organism.modules if m.name == "cortex")
    assert cortex.state["steps"] > 0                 # it thought
    assert cortex.state["columns"] >= 1              # the substrate is alive
