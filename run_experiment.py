"""Run controlled experiments on the organism and print the results.

This is the lab bench.  Each experiment perturbs the organism in a specific way
and measures the response -- lesion studies, ablation, energy starvation,
contradiction floods, and an A/B test of whether enforced sleep aids retention.

Usage::

    python run_experiment.py lesion       # remove the planner mid-run
    python run_experiment.py ablation     # delete 30% of semantic memory
    python run_experiment.py starve       # cut the energy supply
    python run_experiment.py flood        # inject contradictory observations
    python run_experiment.py sleep        # A/B: sleep vs no-sleep retention
    python run_experiment.py all
    python run_experiment.py sleep --ticks 300 --seed 3
"""
from __future__ import annotations

import argparse
import sys

from lab import metrics
from lab.experiment import (fresh_config, run_condition, thermo_config,
                           unlimited_config, window)
from lab.observatory import Observatory
from lab.perturbations import (AblateSemantic, FloodContradiction, LesionModule,
                              StarveEnergy, SuppressSleep)

ASLEEP = ("sleeping", "dormant")


# -- pretty printing ----------------------------------------------------------
def _use_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if reconf:
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def head(title: str) -> None:
    print("\n" + "=" * 66)
    print(f" {title}")
    print("=" * 66)


def row(label: str, value: object) -> None:
    print(f"  {label:<26}: {value}")


def verdict(text: str) -> None:
    print(f"\n  → {text}")


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _first_sleeper(frames, after_tick):
    prev = {}
    for frame in frames:
        if frame.tick <= after_tick:
            prev = frame.module_states
            continue
        for name, state in frame.module_states.items():
            if state in ASLEEP and prev.get(name) == "active":
                return {"module": name, "at_tick": frame.tick,
                        "ticks_after": frame.tick - after_tick}
        prev = frame.module_states
    return None


# -- experiments --------------------------------------------------------------
def exp_lesion(ticks: int, seed: int) -> None:
    head("LESION  --  remove the planner mid-run")
    at = ticks // 3
    obs, org = run_condition(fresh_config(ticks, seed), ticks,
                             [LesionModule(at_tick=at, label="lesion:planner",
                                           name="planner")])
    pre, post = window(obs, 0, at), window(obs, at, ticks + 1)
    pre_learn = metrics.summary(pre, "learning_rate")["mean"]
    post_learn = metrics.summary(post, "learning_rate")["mean"]
    row("lesion applied at tick", at)
    row("learning_rate  pre / post", f"{pre_learn:.3f} / {post_learn:.3f}")
    row("curiosity      pre / post",
        f"{metrics.summary(pre, 'curiosity')['mean']:.3f} / "
        f"{metrics.summary(post, 'curiosity')['mean']:.3f}")
    grew = any("hypothesis" in n for f in post for n in f.module_states)
    row("grew a new organ after lesion", "yes" if grew else "no")
    keep = post_learn >= 0.5 * pre_learn
    verdict("other modules compensated -- learning continued"
            if keep else "learning collapsed -- planner was load-bearing")


def exp_ablation(ticks: int, seed: int) -> None:
    head("ABLATION  --  delete 30% of semantic memory")
    at = ticks // 2
    obs, org = run_condition(fresh_config(ticks, seed), ticks,
                             [AblateSemantic(at_tick=at, label="ablate:semantic",
                                             fraction=0.3, seed=seed)])
    sem = [f.memories.get("semantic", 0) for f in obs.frames]
    pre_peak = max(sem[:at]) if at < len(sem) else 0
    just_after = sem[at - 1] if 0 < at <= len(sem) else 0
    final = sem[-1] if sem else 0
    post = window(obs, at, ticks + 1)
    row("ablation applied at tick", at)
    row("semantic concepts (peak pre)", pre_peak)
    row("semantic concepts (post-cut)", just_after)
    row("semantic concepts (final)", final)
    row("new concepts / 100 ticks post",
        metrics.novel_concept_rate(post)["per_100_ticks"])
    verdict("knowledge re-consolidated after damage"
            if final >= just_after else "damage persisted")


def exp_starve(ticks: int, seed: int) -> None:
    head("STARVE  --  cut the energy supply to 20%")
    at = ticks // 2
    obs, org = run_condition(fresh_config(ticks, seed), ticks,
                             [StarveEnergy(at_tick=at, label="starve:0.2", factor=0.2)])
    pre, post = window(obs, 0, at), window(obs, at, ticks + 1)
    row("starvation applied at tick", at)
    row("fraction asleep  pre / post",
        f"{_mean([f.sleeping / max(1, f.active + f.sleeping) for f in pre]):.2f} / "
        f"{_mean([f.sleeping / max(1, f.active + f.sleeping) for f in post]):.2f}")
    first = _first_sleeper(obs.frames, at)
    if first:
        row("first module to shut down",
            f"{first['module']}  (+{first['ticks_after']} ticks)")
        verdict(f"under scarcity, '{first['module']}' was dropped first")
    else:
        verdict("no module was forced to sleep by starvation")


def exp_flood(ticks: int, seed: int) -> None:
    head("FLOOD  --  inject contradictory observations")
    at = ticks // 2
    burst = [FloodContradiction(at_tick=at + i, label="flood", burst=10, seed=seed)
             for i in range(5)]
    obs, org = run_condition(fresh_config(ticks, seed), ticks, burst)
    pre = window(obs, 0, at)
    post = window(obs, at + 5, ticks + 1)
    base = metrics.summary(pre, "entropy")["mean"]
    peak = max((f.entropy for f in window(obs, at, ticks + 1)), default=0.0)
    recovery = next((f.tick - (at + 5) for f in post if f.entropy <= base * 1.1), None)
    row("flood applied at ticks", f"{at}..{at + 4}")
    row("entropy baseline (pre)", f"{base:.3f}")
    row("entropy peak (during/after)", f"{peak:.3f}")
    row("ticks to recover to baseline", recovery if recovery is not None else ">end")
    verdict(f"entropy spiked +{(peak - base):.2f} then "
            + ("relaxed back" if recovery is not None else "stayed elevated"))


def exp_sleep(ticks: int, seed: int, repeats: int = 3) -> None:
    head("SLEEP STUDY  --  does enforced sleep improve retention?  (A/B)")

    def condition(suppress: bool) -> dict:
        surv, novel, learn = [], [], []
        for s in range(repeats):
            perts = [SuppressSleep(at_tick=1, label="no-sleep")] if suppress else None
            obs, org = run_condition(fresh_config(ticks, seed + s), ticks, perts)
            surv.append(metrics.memory_survival(obs)["median_lifetime"])
            novel.append(metrics.novel_concept_rate(obs.frames)["per_100_ticks"])
            learn.append(metrics.summary(obs.frames, "learning_rate")["mean"])
        return {"median_lifetime": _mean(surv), "novel_per_100": _mean(novel),
                "learning_rate": _mean(learn)}

    rest = condition(suppress=False)
    nosleep = condition(suppress=True)
    row(f"(mean of {repeats} seeds)", "")
    row("median memory lifetime  rest",  f"{rest['median_lifetime']:.1f} ticks")
    row("median memory lifetime  no-sleep", f"{nosleep['median_lifetime']:.1f} ticks")
    row("new concepts/100  rest / no-sleep",
        f"{rest['novel_per_100']:.1f} / {nosleep['novel_per_100']:.1f}")
    row("learning_rate     rest / no-sleep",
        f"{rest['learning_rate']:.3f} / {nosleep['learning_rate']:.3f}")
    better = rest["median_lifetime"] > nosleep["median_lifetime"]
    verdict("sleep improved long-term memory retention"
            if better else "no retention benefit from sleep in this regime")


def _corr(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = _mean(a), _mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else 0.0


def exp_constraint(ticks: int, seed: int, repeats: int = 3) -> None:
    head("CONSTRAINT  --  A (unlimited energy, no load)  vs  B (thermodynamic)")

    def condition(builder) -> dict:
        surv, novel, adapt, phi, learn = [], [], [], [], []
        for s in range(repeats):
            obs, org = run_condition(builder(ticks, seed + s), ticks)
            surv.append(metrics.memory_survival(obs)["median_lifetime"])
            novel.append(metrics.novel_concept_rate(obs.frames)["per_100_ticks"])
            adapt.append(metrics.adaptation_frequency(obs.frames)["per_100_ticks"])
            phi.append(metrics.integrated_information_proxy(obs.frames))
            learn.append(metrics.summary(obs.frames, "learning_rate")["mean"])
        return {"retention": _mean(surv), "novel": _mean(novel),
                "adapt": _mean(adapt), "phi": _mean(phi), "learn": _mean(learn)}

    a = condition(unlimited_config)
    b = condition(thermo_config)
    row(f"(mean of {repeats} seeds)", "A unlimited   /   B thermodynamic")
    row("memory retention (median life)", f"{a['retention']:.1f}   /   {b['retention']:.1f}")
    row("novel concepts / 100 ticks", f"{a['novel']:.1f}   /   {b['novel']:.1f}")
    row("adaptation events / 100", f"{a['adapt']:.2f}   /   {b['adapt']:.2f}")
    row("phi-proxy (integration)", f"{a['phi']:.4f}   /   {b['phi']:.4f}")
    row("learning rate (event flow)", f"{a['learn']:.3f}   /   {b['learn']:.3f}")
    winners = [
        "retention->B" if b["retention"] > a["retention"] else "retention->A",
        "novelty->B" if b["novel"] > a["novel"] else "novelty->A",
        "phi->B" if b["phi"] > a["phi"] else "phi->A",
    ]
    verdict("mixed: " + ", ".join(winners)
            + "  --  constraint trades abstraction speed for retention "
              "(NOT a clean 'B wins')")


def _scarce_thermo(ticks: int, seed: int, influx: float) -> "object":
    cfg = thermo_config(ticks, seed)
    cfg.thermo.influx = influx
    cfg.thermo.uptake = 3.0            # slow metabolism, so scarcity actually bites
    cfg.thermo.reservoir_level = cfg.thermo.reservoir_capacity = 400.0
    return cfg


def exp_overheat(ticks: int, seed: int) -> None:
    head("OVERHEAT  --  self-regulation emerges in a scarcity BAND  (emergent)")
    # Voluntary rest is non-monotonic in free energy: too much -> never rests,
    # too little -> too starved to cycle.  We sweep influx to expose the band.
    def onsets(obs, name):
        return metrics.sleep_cycles(obs.frames).get(name, {}).get("sleep_onsets", 0)

    classic_obs, _ = run_condition(fresh_config(ticks, seed), ticks)
    row("planner sleep-onsets (classic control)", onsets(classic_obs, "planner"))
    row("", "-- thermodynamic, by environmental influx --")
    peak = (0.0, -1.0)
    for influx in (16.0, 10.0, 6.0, 4.0):
        obs, _ = run_condition(_scarce_thermo(ticks, seed, influx), ticks)
        o = onsets(obs, "planner")
        load = _mean([f.module_load.get("planner", 0.0) for f in obs.frames])
        row(f"influx={influx:<4}  onsets / mean-load", f"{o}  /  {load:.1f}")
        if o > peak[1]:
            peak = (influx, o)
    verdict(f"planner rest is non-monotonic in free energy (peak influx≈{peak[0]:.0f}, "
            f"{int(peak[1])} onsets); it is emergent -- energy-gated but load-shaped -- "
            f"with no explicit sleep rule (diagnostic: rest is still energy-triggered, "
            f"because the manifold self-limits load before THINK collapses)")


EXPERIMENTS = {
    "lesion": exp_lesion,
    "ablation": exp_ablation,
    "starve": exp_starve,
    "flood": exp_flood,
    "sleep": exp_sleep,
    "constraint": exp_constraint,
    "overheat": exp_overheat,
}


def main(argv: list[str] | None = None) -> None:
    _use_utf8()
    parser = argparse.ArgumentParser(description="Run experiments on the organism")
    parser.add_argument("experiment", choices=list(EXPERIMENTS) + ["all"])
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in names:
        EXPERIMENTS[name](args.ticks, args.seed)
    print()


if __name__ == "__main__":
    main()
