"""Derived measures -- turning recorded frames into research results.

Everything here is a pure function of the recorded data (no numpy, no side
effects), so measures are cheap to compute and trivial to unit test.

A note on honesty: :func:`integrated_information_proxy` is an *approximation*
inspired by IIT, not a computation of Phi.  It rewards a system whose modules
are both differentiated (they vary) and integrated (they co-vary) while active.
Treat it as a comparative index across runs, not an absolute quantity.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from lab.observatory import Frame, Observatory

ASLEEP = ("sleeping", "dormant")


# -- tiny stats helpers -------------------------------------------------------
def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def series(frames: Sequence[Frame], attr: str) -> list[float]:
    return [float(getattr(f, attr)) for f in frames]


def summary(frames: Sequence[Frame], attr: str) -> dict[str, float]:
    xs = series(frames, attr)
    return {"mean": round(_mean(xs), 3), "std": round(_std(xs), 3),
            "min": round(min(xs), 3) if xs else 0.0,
            "max": round(max(xs), 3) if xs else 0.0}


# -- structural / activity measures ------------------------------------------
def module_activity(frames: Sequence[Frame]) -> tuple[list[str], dict[str, list[float]]]:
    names = sorted({n for f in frames for n in f.module_states})
    activity = {n: [1.0 if f.module_states.get(n) == "active" else 0.0 for f in frames]
                for n in names}
    return names, activity


def integrated_information_proxy(frames: Sequence[Frame]) -> float:
    """Comparative index of integration x differentiation x activity (see module docstring)."""
    names, activity = module_activity(frames)
    if len(names) < 2:
        return 0.0
    corrs = [abs(_corr(activity[names[i]], activity[names[j]]))
             for i in range(len(names)) for j in range(i + 1, len(names))]
    integration = _mean(corrs)
    differentiation = _mean([_std(activity[n]) for n in names])
    active_frac = _mean([f.active / max(1, f.active + f.sleeping) for f in frames])
    return round(integration * differentiation * active_frac, 4)


def sleep_cycles(frames: Sequence[Frame]) -> dict[str, dict[str, float]]:
    names, _ = module_activity(frames)
    result: dict[str, dict[str, float]] = {}
    for name in names:
        states = [f.module_states.get(name) for f in frames]
        onsets = sum(1 for i in range(1, len(states))
                     if states[i] in ASLEEP and states[i - 1] == "active")
        asleep = sum(1 for s in states if s in ASLEEP)
        result[name] = {
            "sleep_onsets": onsets,
            "fraction_asleep": round(asleep / len(states), 3) if states else 0.0,
            "mean_period": round(len(states) / onsets, 1) if onsets else 0.0,
        }
    return result


def curiosity_oscillation(frames: Sequence[Frame]) -> dict[str, float]:
    xs = series(frames, "curiosity")
    out = {"amplitude": round(_std(xs), 3), "mean": round(_mean(xs), 3), "period": 0.0}
    n = len(xs)
    if n < 6:
        return out
    m = _mean(xs)
    d = [x - m for x in xs]
    den = sum(v * v for v in d)
    if den <= 0:
        return out
    acf = [sum(d[i] * d[i + k] for i in range(n - k)) / den for k in range(n // 2)]
    dipped = False
    best_k, best = 0, 0.0
    for k in range(1, len(acf) - 1):
        if acf[k] < 0:
            dipped = True
        if dipped and acf[k] > acf[k - 1] and acf[k] >= acf[k + 1] and acf[k] > best:
            best, best_k = acf[k], k
    out["period"] = float(best_k)
    return out


def novel_concept_rate(frames: Sequence[Frame]) -> dict[str, float]:
    counts = [f.memories.get("semantic", 0) for f in frames]
    gains = sum(max(0, counts[i] - counts[i - 1]) for i in range(1, len(counts)))
    return {"total_new": gains,
            "per_100_ticks": round(100 * gains / len(frames), 2) if frames else 0.0,
            "final": counts[-1] if counts else 0}


def adaptation_frequency(frames: Sequence[Frame]) -> dict[str, float]:
    total = sum(len(f.structural) for f in frames)
    return {"events": total,
            "per_100_ticks": round(100 * total / len(frames), 2) if frames else 0.0}


def interaction_graph(organism) -> list[tuple[str, str, float]]:
    """Directed edges source -> module weighted by the learned (Hebbian) link.

    Since communicating costs energy, this doubles as an energy-flow graph:
    strong edges are the pathways most of the organism's energy travels along.
    """
    edges = [(src, m.name, round(w, 3))
             for m in organism.modules for src, w in m.connections.items()]
    edges.sort(key=lambda e: e[2], reverse=True)
    return edges


def learning_efficiency(frames: Sequence[Frame]) -> dict[str, float]:
    """A first, deliberately crude probe of the conjectured relationship

        useful learning  ~  free energy / (metabolic load x information entropy)

    Reported as new concepts produced per unit of (load x entropy).  Only
    meaningful on the thermodynamic substrate; classic runs have zero load."""
    mean_load = _mean(series(frames, "load"))
    mean_entropy = _mean(series(frames, "entropy"))
    useful = novel_concept_rate(frames)["total_new"]
    if mean_load < 1e-6:
        return {"applicable": 0.0, "useful_learning": useful,
                "mean_load": round(mean_load, 3), "mean_entropy": round(mean_entropy, 3)}
    cost = mean_load * mean_entropy
    return {"applicable": 1.0, "useful_learning": useful,
            "mean_load": round(mean_load, 3), "mean_entropy": round(mean_entropy, 3),
            "efficiency": round(useful / cost, 4) if cost > 0 else 0.0}


def memory_survival(observatory: Observatory) -> dict[str, float]:
    lifetimes = observatory.memory_lifetimes()
    if not lifetimes:
        return {"tracked": 0}
    all_life = sorted(lt for lt, _ in lifetimes)
    died = [lt for lt, d in lifetimes if d]
    return {
        "tracked": len(lifetimes),
        "died": len(died),
        "survivors": len(lifetimes) - len(died),
        "median_lifetime": all_life[len(all_life) // 2],
        "mean_lifetime_of_dead": round(_mean(died), 1) if died else 0.0,
    }


# -- one call to rule them all ------------------------------------------------
def full_report(frames: Sequence[Frame], organism=None,
                observatory: Observatory | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ticks": len(frames),
        "energy": summary(frames, "energy"),
        "entropy": summary(frames, "entropy"),
        "curiosity": summary(frames, "curiosity"),
        "learning_rate": summary(frames, "learning_rate"),
        "health": summary(frames, "health"),
        "phi_proxy": integrated_information_proxy(frames),
        "novel_concepts": novel_concept_rate(frames),
        "curiosity_oscillation": curiosity_oscillation(frames),
        "sleep_cycles": sleep_cycles(frames),
        "adaptation": adaptation_frequency(frames),
    }
    if any(f.load > 0 for f in frames):
        report["load"] = summary(frames, "load")
        report["fidelity"] = summary(frames, "fidelity")
        report["learning_efficiency"] = learning_efficiency(frames)
    if organism is not None:
        report["interaction_graph"] = interaction_graph(organism)[:8]
    if observatory is not None:
        report["memory_survival"] = memory_survival(observatory)
    return report
