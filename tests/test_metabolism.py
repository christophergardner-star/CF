import logging

from core.energy import Action, EnergyBudget
from core.entropy import EntropyField
from core.events import EventBus, ObservationEvent
from core.lifecycle import LifecycleState
from core.metabolism import (ClassicMetabolism, EnergyReservoir,
                            ThermodynamicMetabolism)
from core.module import Module
from lab.experiment import fresh_config, thermo_config
from main import build_organism


class _Tiny(Module):
    subscriptions = ()

    def observe(self):
        return self.drain_inbox()

    def think(self, observations):
        return []

    def learn(self, thoughts):
        pass


# -- reservoir ----------------------------------------------------------------
def test_reservoir_draw_is_conservative():
    r = EnergyReservoir(level=10, capacity=10, influx=0)
    assert r.draw(4) == 4 and r.level == 6
    assert r.draw(100) == 6 and r.level == 0  # cannot hand out what it lacks


def test_reservoir_replenish_caps_at_capacity():
    r = EnergyReservoir(level=8, capacity=10, influx=5)
    assert r.replenish() == 2 and r.level == 10  # only the room is filled


# -- classic control ----------------------------------------------------------
def test_classic_metabolism_has_no_load_and_full_fidelity():
    m = ClassicMetabolism(EnergyBudget(20, 20, 3), EntropyField(level=0.9))
    assert m.do_work(Action.THINK)
    assert m.load == 0.0
    assert m.fidelity() == 1.0  # entropy never degrades fidelity in the control


# -- thermodynamic substrate --------------------------------------------------
def test_work_turns_energy_into_heat_and_conserves_it():
    energy = EnergyBudget(20, 20, 0)
    m = ThermodynamicMetabolism(energy, EntropyField(level=0.0, baseline=0.0),
                                heat_fraction=0.6)
    before = energy.level + m.load + m.dissipated
    assert m.do_work(Action.LEARN)  # cost 2.0
    after = energy.level + m.load + m.dissipated
    assert abs(before - after) < 1e-9   # nothing created or destroyed
    assert m.load > 0.0                 # some energy retained as load


def test_sleeping_cools_load_faster():
    awake = ThermodynamicMetabolism(EnergyBudget(50, 50), EntropyField(), load_decay=0.1)
    rested = ThermodynamicMetabolism(EnergyBudget(50, 50), EntropyField(), load_decay=0.1)
    awake.load = rested.load = 10.0
    awake.recover(None, sleeping=False)
    rested.recover(None, sleeping=True)
    assert rested.load < awake.load


def test_load_raises_entropy_and_degrades_fidelity():
    m = ThermodynamicMetabolism(EnergyBudget(50, 50), EntropyField(level=0.1, baseline=0.1),
                                load_to_entropy=0.6, fidelity_gain=0.85)
    m.load = 40.0
    f_before = m.fidelity()
    for _ in range(10):
        m.regulate()
    assert m.entropy.level > 0.1
    assert m.fidelity() < f_before


# -- the conservation law at the organism level -------------------------------
def test_thermodynamic_organism_conserves_energy():
    config = thermo_config(20, seed=1)
    config.adaptation.enabled = False  # spawning/pruning changes the mass balance
    organism = build_organism(config, logging.getLogger("t"))
    for _ in range(20):
        before, diss_before = organism.metabolic_total(), organism.total_dissipated()
        organism.tick()
        delta = organism.metabolic_total() - before
        expected = organism.last_influx - (organism.total_dissipated() - diss_before)
        assert abs(delta - expected) < 1e-6   # energy never appears from nowhere


def test_classic_organism_reports_no_load():
    organism = build_organism(fresh_config(15), logging.getLogger("t"))
    for _ in range(15):
        snap = organism.tick()
    assert snap["load"] == 0.0
    assert snap["fidelity"] == 1.0


# -- the action manifold (load deforms which actions are reachable) -----------
def test_classic_reaches_every_action_regardless_of_entropy():
    m = ClassicMetabolism(EnergyBudget(10, 10, 3), EntropyField(level=1.0))
    assert m.capacity() == 1.0
    assert all(m.reachable(a) for a in Action)


def test_load_shrinks_manifold_dropping_complex_actions_first():
    m = ThermodynamicMetabolism(EnergyBudget(20, 20, 0), EntropyField(),
                                action_ceiling=0.9)
    assert m.capacity() == 1.0 and m.reachable(Action.LEARN)
    m.load = 20 * 0.9 * 0.5                  # capacity ~0.5
    assert not m.reachable(Action.LEARN)     # complexity 0.8 -> gone
    assert not m.reachable(Action.PLAN)      # complexity 0.6 -> gone
    assert m.reachable(Action.THINK)         # complexity 0.4 -> still reachable
    assert m.reachable(Action.COMMUNICATE)


def test_action_collapse_stalls_a_module():
    bus = EventBus()
    m = ThermodynamicMetabolism(EnergyBudget(10, 10, 0), EntropyField())
    mod = _Tiny("tiny", bus, metabolism=m)
    mod._inbox.append(ObservationEvent(source="world"))
    m.load = 9.5                              # capacity ~0 -> even THINK unreachable
    assert mod.wants_to_act() and not mod.has_work()  # drive, but nothing reachable
    mod.step(tick=5)
    # A module that cannot even think does not refresh its activity -> it drifts
    # to sleep, so rest can arise from action-space collapse (not just energy).
    assert mod.lifecycle.state is not LifecycleState.ACTIVE
