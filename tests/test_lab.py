import logging

from lab import metrics
from lab.experiment import fresh_config, run_condition
from lab.observatory import Frame, Observatory
from lab.perturbations import (AblateSemantic, LesionModule, StarveEnergy,
                              SuppressSleep)
from main import build_organism


def _frame(tick, states, curiosity=0.5, semantic=0):
    return Frame(tick=tick, energy=100, energy_fraction=0.5, entropy=0.5,
                 curiosity=curiosity, learning_rate=0.3, health=0.8,
                 active=sum(1 for s in states.values() if s == "active"),
                 sleeping=sum(1 for s in states.values() if s == "sleeping"),
                 events_delta=1, event_counts={}, memories={"semantic": semantic},
                 module_states=states, module_energy={}, module_entropy={},
                 module_confidence={})


def test_sleep_cycles_counts_onsets():
    frames = [
        _frame(1, {"a": "active"}),
        _frame(2, {"a": "sleeping"}),   # onset 1
        _frame(3, {"a": "active"}),
        _frame(4, {"a": "sleeping"}),   # onset 2
    ]
    cycles = metrics.sleep_cycles(frames)
    assert cycles["a"]["sleep_onsets"] == 2
    assert cycles["a"]["fraction_asleep"] == 0.5


def test_novel_concept_rate_counts_gains_only():
    frames = [_frame(i, {"a": "active"}, semantic=s)
              for i, s in enumerate([0, 1, 3, 3, 2, 4], start=1)]
    stats = metrics.novel_concept_rate(frames)
    assert stats["total_new"] == 5  # +1 +2 +0 +0 +2 ; the drop is ignored
    assert stats["final"] == 4


def test_phi_proxy_is_zero_for_single_module():
    frames = [_frame(i, {"a": "active"}) for i in range(1, 6)]
    assert metrics.integrated_information_proxy(frames) == 0.0


def test_lesion_removes_module():
    org = build_organism(fresh_config(5), logging.getLogger("t"))
    obs = Observatory(org)
    obs.run(5, [LesionModule(at_tick=2, label="x", name="planner")])
    assert all("planner" not in f.module_states for f in obs.frames if f.tick >= 3)


def test_ablation_reduces_semantic_memory():
    org = build_organism(fresh_config(40), logging.getLogger("t"))
    for _ in range(30):
        org.tick()
    before = len(org.memory.semantic)
    AblateSemantic(at_tick=0, label="a", fraction=0.5, seed=1).apply(org)
    assert len(org.memory.semantic) <= before


def test_starve_cuts_energy():
    org = build_organism(fresh_config(5), logging.getLogger("t"))
    level_before = org.energy.level
    StarveEnergy(at_tick=0, label="s", factor=0.1).apply(org)
    assert org.energy.level < level_before


def test_suppress_sleep_keeps_all_awake():
    org = build_organism(fresh_config(40), logging.getLogger("t"))
    obs = Observatory(org)
    obs.run(40, [SuppressSleep(at_tick=1, label="no-sleep")])
    total_asleep = sum(f.sleeping for f in obs.frames)
    assert total_asleep == 0


def test_observatory_records_one_frame_per_tick_and_survival():
    obs, org = run_condition(fresh_config(25), 25)
    assert len(obs.frames) == 25
    survival = metrics.memory_survival(obs)
    assert survival["tracked"] > 0
    report = metrics.full_report(obs.frames, org, obs)
    assert set(report) >= {"phi_proxy", "sleep_cycles", "memory_survival"}
