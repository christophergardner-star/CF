"""The STRATA continual-learning bench.

Streams a sequence of *tasks* (each a noisy cyclic pattern of unit vectors)
through the network with no task labels and no replay buffer, in the schedule
``A -> B -> C -> A``, then asks the only question that matters for continual
learning: **how much of task A survived learning B and C?**

Four conditions strip the layered defenses one at a time:

    full          routing + cascade + fast weights + sparse codes
    -routing      one column for everything (structural isolation removed)
    -cascade      ... and single-timescale weights (power-law memory removed)
    dense         ... and dense codes, no fast weights (a vanilla online
                  regressor -- the catastrophic-forgetting control)

Run with:  python -m strata.demo  [--ticks-per-phase 600] [--seed 0]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from strata.network import StrataConfig, StrataNetwork


class CyclicTask:
    """A deterministic cycle of unit vectors observed under small noise."""

    def __init__(
        self,
        d: int,
        length: int,
        seed: int,
        noise: float = 0.02,
        pattern: np.ndarray | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        if pattern is None:
            pattern = rng.standard_normal((length, d))
        self.pattern = pattern / np.linalg.norm(pattern, axis=1, keepdims=True)
        self.noise = noise
        self.rng = rng

    def sample(self, t: int) -> np.ndarray:
        clean = self.pattern[t % len(self.pattern)]
        return clean + self.noise * self.rng.standard_normal(len(clean))


def make_interference_tasks(
    d: int, n_unique: int, n_shared: int, task_seeds: list[int], shared_seed: int
) -> list[CyclicTask]:
    """Tasks that *collide*: a shared vocabulary appears in every task's cycle
    but with task-specific successors, so no single static input->next map can
    fit two tasks at once.  This is what makes forgetting mandatory for an
    unprotected learner (disjoint vocabularies can be fit jointly by one
    linear map whenever capacity suffices -- measured, not assumed)."""
    shared = np.random.default_rng(shared_seed).standard_normal((n_shared, d))
    tasks = []
    for seed in task_seeds:
        rng = np.random.default_rng(seed)
        unique = rng.standard_normal((n_unique, d))
        pattern = np.concatenate([unique, shared])
        pattern = pattern[rng.permutation(len(pattern))]   # task-specific order
        tasks.append(CyclicTask(d, len(pattern), seed=seed, pattern=pattern))
    return tasks


def run_condition(
    config: StrataConfig, schedule: list[tuple[str, CyclicTask, int]]
) -> dict:
    """Stream the schedule through a fresh network; return retention metrics."""
    net = StrataNetwork(config)
    log: list[tuple[int, str, float]] = []          # (phase_index, task, error)
    for phase, (name, task, ticks) in enumerate(schedule):
        for t in range(ticks):
            report = net.step(task.sample(t))
            if report["error"] is not None:
                log.append((phase, name, report["error"]))

    def mean_err(
        phase: int,
        first: int | None = None,
        last: int | None = None,
        skip: int = 0,
    ):
        errs = [e for p, _, e in log if p == phase][skip:]
        window = errs[:first] if first else errs[-last:]
        return float(np.mean(window))

    revisit_phase = len(schedule) - 1
    return {
        "A learned": mean_err(0, last=100),
        "A revisit": mean_err(revisit_phase, first=60),
        # recall after the routing context has re-oriented (~12 ticks):
        # separates re-orientation latency from genuine forgetting
        "A recall": mean_err(revisit_phase, first=48, skip=12),
        "A relearned": mean_err(revisit_phase, last=100),
        "columns": len(net.router.columns),
        "spawns": net.router.spawn_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks-per-phase", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # d=24: random unit inputs need enough ambient dimension to be mutually
    # distinguishable; the encoder cannot separate what the input space
    # itself does not separate (measured: at d=8 no routing threshold exists)
    d, ticks = 24, args.ticks_per_phase

    def build_schedule() -> list[tuple[str, CyclicTask, int]]:
        """Fresh task objects so every condition sees the identical stream."""
        seeds = [args.seed + 1, args.seed + 2, args.seed + 3]
        a1, b, c = make_interference_tasks(
            d, n_unique=10, n_shared=3, task_seeds=seeds, shared_seed=args.seed + 99
        )
        (a2,) = make_interference_tasks(
            d, n_unique=10, n_shared=3, task_seeds=seeds[:1], shared_seed=args.seed + 99
        )
        return [("A", a1, ticks), ("B", b, ticks), ("C", c, ticks), ("A", a2, ticks)]

    base = StrataConfig(d_in=d, code_dim=128, seed=args.seed)
    conditions = {
        "full STRATA": base,
        "-routing": replace(base, use_routing=False),
        "-cascade": replace(base, use_routing=False, use_cascade=False),
        "dense": replace(
            base,
            use_routing=False,
            use_cascade=False,
            use_fast=False,
            use_sparse=False,
        ),
    }

    header = (
        f"{'condition':<14}{'A learned':>11}{'A revisit':>11}"
        f"{'A recall':>10}{'A relearned':>13}{'cols':>6}{'spawns':>8}"
    )
    print(f"\nSchedule: A({ticks}) B({ticks}) C({ticks}) A({ticks})   "
          f"(squared next-step prediction error; noise floor ~{d * 0.02**2:.4f})\n")
    print(header)
    print("-" * len(header))
    for name, config in conditions.items():
        result = run_condition(config, build_schedule())
        print(
            f"{name:<14}"
            f"{result['A learned']:>11.4f}"
            f"{result['A revisit']:>11.4f}"
            f"{result['A recall']:>10.4f}"
            f"{result['A relearned']:>13.4f}"
            f"{result['columns']:>6d}"
            f"{result['spawns']:>8d}"
        )
    print(
        "\n'A revisit' = error on task A in the first 60 ticks after 1200 ticks"
        "\nof other tasks (includes ~12 ticks of routing re-orientation)."
        "\n'A recall'  = the same window minus the re-orientation transient:"
        "\nthis is the forgetting metric proper."
    )


if __name__ == "__main__":
    main()
