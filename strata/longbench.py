"""The long-sequence bench: the experiment that settles STRATA's claims.

A schedule of many tasks organised into *families*: each family has a shared
prototype direction (so its symbols are mutually correlated -- hard for a
fixed random encoder) and successive *variants* within a family share most of
their vocabulary and transitions (so forward transfer is possible in
principle).  The bench measures, per condition:

  retention     frozen-circuit error on every task after the whole schedule,
                as a function of how long ago the task was learned
  capacity      columns spawned vs. tasks seen (diversity- or task-scaling?)
  transfer      ticks-to-criterion on a family's first variant vs. its later
                variants (later should be faster if transfer is real)

Encoder conditions:

  fixed          random projection, never learns (the prototype so far)
  naive          competitive Hebbian, no consolidation (drift unprotected)
  consolidated   competitive Hebbian + per-unit consolidation + recruitment

Run with:  python -m strata.longbench  [--families 8] [--variants 3]
           [--ticks-per-task 500] [--seed 0]
"""
from __future__ import annotations

import argparse
import copy

import numpy as np

from strata.network import StrataConfig, StrataNetwork


class Task:
    """A noisy cyclic sequence of symbol vectors."""

    def __init__(self, pattern: np.ndarray, seed: int, noise: float = 0.02):
        self.pattern = pattern / np.linalg.norm(pattern, axis=1, keepdims=True)
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def sample(self, t: int) -> np.ndarray:
        clean = self.pattern[t % len(self.pattern)]
        return clean + self.noise * self.rng.standard_normal(len(clean))


def make_schedule(
    d: int,
    n_families: int,
    variants: int,
    length: int = 12,
    substitutions: int = 3,
    family_pull: float = 0.6,
    seed: int = 0,
    rank: int | None = None,
) -> list[Task]:
    """Families of related tasks.  Every symbol of a family is pulled toward
    the family prototype (correlated inputs: a fixed random encoder gives
    them overlapping codes).  Variant k+1 keeps the cycle of variant k but
    substitutes a few symbols -- most transitions carry over, so a system
    with real forward transfer learns later variants faster.

    With ``rank`` set, each family's symbol deviations are confined to a
    ``rank``-dimensional subspace (orthogonal to the prototype): after the
    slow pathway removes the family mean, the residuals live on a low-rank
    manifold that a random projection scatters inefficiently but a learned
    dictionary can carve up -- the bench where representation learning must
    prove it *beats* random, not just matches it."""
    rng = np.random.default_rng(seed)
    tasks: list[Task] = []

    for f in range(n_families):
        prototype = rng.standard_normal(d)
        prototype /= np.linalg.norm(prototype)
        basis = None
        if rank is not None:
            raw = rng.standard_normal((d, rank))
            raw -= np.outer(prototype, prototype @ raw)   # orthogonal to p
            basis, _ = np.linalg.qr(raw)

        def family_symbol() -> np.ndarray:
            if basis is None:
                fresh = rng.standard_normal(d)
            else:
                fresh = basis @ rng.standard_normal(basis.shape[1])
            fresh /= np.linalg.norm(fresh)
            v = family_pull * prototype + (1.0 - family_pull**2) ** 0.5 * fresh
            return v / np.linalg.norm(v)

        pattern = np.array([family_symbol() for _ in range(length)])
        for k in range(variants):
            if k > 0:
                pattern = pattern.copy()
                for pos in rng.choice(length, size=substitutions, replace=False):
                    pattern[pos] = family_symbol()
            tasks.append(Task(pattern, seed=seed + 1000 * f + k))
    return tasks


def frozen_error(net: StrataNetwork, task: Task, cycles: int = 8) -> float:
    """Evaluate a task on a plasticity-frozen copy of the network.

    Fast-weight capture is also disabled (it would one-shot rebind the task
    during the probe itself) and so are centroid claims (a re-centering slow
    expectation would likewise mask true structural retention)."""
    probe = copy.deepcopy(net)
    probe.router.spawn_threshold = 10.0          # unreachable
    probe.cfg.eta_fast = 0.0
    probe.router.adaptive = False
    probe.router.theta_claim = 2.0               # unreachable
    L = len(task.pattern)
    errs = []
    for t in range(cycles * L):
        report = probe.step(task.sample(t), plasticity=0.0)
        if report["error"] is not None and t >= (cycles // 2) * L:
            errs.append(report["error"])
    return float(np.mean(errs))


def run_condition(
    config: StrataConfig,
    tasks: list[Task],
    ticks_per_task: int,
    criterion: float = 0.15,
) -> dict:
    net = StrataNetwork(config)
    ttc: list[int] = []                          # ticks-to-criterion per task
    columns_curve: list[int] = []
    for task in tasks:
        window: list[float] = []
        reached = ticks_per_task
        for t in range(ticks_per_task):
            report = net.step(task.sample(t))
            if report["error"] is not None:
                window.append(report["error"])
                if len(window) > 24:
                    window.pop(0)
                if (reached == ticks_per_task and len(window) == 24
                        and float(np.mean(window)) < criterion):
                    reached = t
        ttc.append(reached)
        columns_curve.append(len(net.router.columns))

    retention = [frozen_error(net, task) for task in tasks]
    return {"ttc": ttc, "columns": columns_curve, "retention": retention,
            "net": net}


def summarise(name: str, result: dict, variants: int) -> str:
    ttc = np.array(result["ttc"], dtype=float)
    retention = np.array(result["retention"])
    n = len(ttc)
    first = ttc[0::variants].mean()              # each family's first variant
    later = np.concatenate(
        [ttc[k::variants] for k in range(1, variants)]).mean()
    old = retention[: n // 3].mean()             # learned longest ago
    recent = retention[-n // 3:].mean()
    return (f"{name:<13}"
            f"{first:>9.0f}{later:>9.0f}{first / max(later, 1.0):>7.1f}x"
            f"{old:>10.3f}{recent:>9.3f}"
            f"{result['columns'][-1]:>7d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--ticks-per-task", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=None,
                        help="confine family residuals to this rank")
    parser.add_argument("--code-dim", type=int, default=128)
    parser.add_argument("--criterion", type=float, default=0.15)
    parser.add_argument("--conjunctive", type=int, default=0,
                        help="fixed conjunctive expansion size (0 = off)")
    parser.add_argument("--grow-from", type=int, default=None,
                        help="start the dictionary at this size and grow")
    parser.add_argument("--adaptive", action="store_true",
                        help="statistics-tracked routing thresholds")
    args = parser.parse_args()

    d = 24
    n_tasks = args.families * args.variants

    def base(**overrides) -> StrataConfig:
        return StrataConfig(
            d_in=d, code_dim=args.code_dim, seed=args.seed, max_columns=48,
            conjunctive_dim=args.conjunctive,
            encoder_grow_from=args.grow_from,
            adaptive_bands=args.adaptive,
            **overrides)

    conditions = {
        "fixed": base(),
        "fixed+mp": base(encode_mode="mp"),
        "naive": base(encoder_lr=0.15, encoder_kappa=0.0),
        "consolidated": base(encoder_lr=0.15, encoder_kappa=0.2),
        "consol.+mp": base(encoder_lr=0.15, encoder_kappa=0.2,
                           encode_mode="mp"),
        "cons+mp+dec": base(encoder_lr=0.15, encoder_kappa=0.2,
                            encode_mode="mp", encoder_usage_decay=2e-4),
    }

    structure = ("isotropic residuals" if args.rank is None
                 else f"rank-{args.rank} residual subspace per family")
    print(f"\n{n_tasks} tasks ({args.families} families x {args.variants} "
          f"variants), {args.ticks_per_task} ticks each, correlated symbols "
          f"(family pull 0.6), {structure}\n")
    header = (f"{'condition':<13}{'TTC v1':>9}{'TTC v2+':>9}{'fwd':>8}"
              f"{'ret old':>10}{'ret new':>9}{'cols':>7}")
    print(header)
    print("-" * len(header))
    for name, config in conditions.items():
        tasks = make_schedule(
            d, args.families, args.variants, seed=args.seed + 7,
            rank=args.rank)
        result = run_condition(config, tasks, args.ticks_per_task,
                               criterion=args.criterion)
        print(summarise(name, result, args.variants))
    print(
        "\nTTC v1 / v2+ : ticks-to-criterion on each family's first variant vs"
        "\n               its later variants; 'fwd' > 1 means forward transfer."
        "\nret old/new  : frozen-circuit error on the oldest / newest third of"
        "\n               tasks after the full schedule (forgetting-with-age)."
        "\ncols         : columns after all tasks "
        "(diversity-scaling if << tasks)."
    )


if __name__ == "__main__":
    main()
