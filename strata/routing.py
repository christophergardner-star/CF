"""Centroid routing over columns, with novelty-driven spawning.

Routing is kernel-based (cosine to a centroid), not a learned softmax gate --
there is nothing to collapse -- and it compares against a constant number of
modules, never across sequence positions, so it costs O(1) per step.

Columns outside the active set receive exactly zero learning: structural
isolation is the first and cheapest line of defense against forgetting.
A new column is spawned only when *no* existing column claims the input for a
sustained stretch, measured by a *leaky* orphan level rather than a
consecutive-tick streak (a single familiar tick inside a novel regime must not
reset the evidence).  The spawn is warm-started from the nearest column so old
knowledge seeds the new task (forward transfer).

Routing keys are expected to be slow *context* signals (an EMA of recent
codes), so columns form at task granularity rather than one per input pattern.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from strata.column import LiquidColumn


class ColumnRouter:
    """Owns the population of columns and decides who is in the circuit."""

    def __init__(
        self,
        factory: Callable[[int], LiquidColumn],
        top_k: int = 2,
        theta_route: float = 0.5,
        theta_claim: float = 0.8,
        spawn_rate: float = 0.15,
        spawn_threshold: float = 0.8,
        max_columns: int = 16,
        adaptive: bool = False,
    ) -> None:
        self.factory = factory
        self.top_k = top_k
        self.theta_route = theta_route
        self.theta_claim = theta_claim
        self.spawn_rate = spawn_rate
        self.spawn_threshold = spawn_threshold
        self.max_columns = max_columns
        self.adaptive = adaptive
        # background similarity statistics (updated on orphaned ticks): the
        # floor no adaptive claim band may ever cross -- without it, a
        # column that begins claiming noise widens its own band and
        # absorption returns *through the statistics* (threshold collapse)
        self.bg_mean = 0.2
        self.bg_var = 0.02
        #: set by the host each tick; background statistics must not update
        #: during regime transitions -- boundary swings inflate bg_mean,
        #: which raises the adaptive floor and route thresholds, which
        #: orphans sibling regimes, which spawns storms (measured: 48
        #: columns for 48 tasks at 16 families).  Statistics are identity,
        #: and identity freezes during transitions.
        self.stats_frozen = False
        self.columns: list[LiquidColumn] = []
        self.orphan_level = 0.0
        self.spawn_count = 0

    def route_threshold(self, column: LiquidColumn) -> float:
        if not self.adaptive:
            return self.theta_route
        return 0.5 * (self.claim_threshold(column)
                      + self.bg_mean + self.bg_var ** 0.5)

    def claim_threshold(self, column: LiquidColumn) -> float:
        if not self.adaptive:
            return self.theta_claim
        floor = self.bg_mean + 2.0 * self.bg_var ** 0.5
        return max(column.claim_mean - 2.0 * column.claim_var ** 0.5, floor)

    def route(
        self,
        code: np.ndarray,
        prior: np.ndarray | None = None,
    ) -> list[tuple[LiquidColumn, float, bool]]:
        """Return the active set as ``(column, gate, learns)`` triples.

        ``learns`` is False for a fallback column (nobody cleared the routing
        threshold): it still predicts, so errors are visible, but it must not
        learn -- otherwise every regime boundary corrupts the nearest old
        column during the orphan window before a spawn fires.

        ``prior`` (optional, one value per column, in [0, 1]) is a top-down
        expectation from a higher layer.  It biases *inference only*, with a
        bounded bonus: a prior may promote an already-plausible column, but
        can never make an alien column routable (prior capture guard).
        """
        if not self.columns:
            self._spawn(code, parent=None)
        affinities = np.array([c.affinity(code) for c in self.columns])
        if prior is not None and len(prior) == len(self.columns):
            bonus = np.array([
                0.5 * max(self.claim_threshold(c) - self.route_threshold(c), 0.0)
                for c in self.columns
            ])
            affinities = affinities + bonus * np.clip(prior, 0.0, 1.0)
        thresholds = np.array([self.route_threshold(c) for c in self.columns])
        best = float(affinities.max())

        orphaned = 1.0 if not np.any(affinities >= thresholds) else 0.0
        if self.adaptive and orphaned and not self.stats_frozen:
            delta = best - self.bg_mean
            self.bg_mean += 0.02 * delta
            self.bg_var += 0.02 * (delta * delta - self.bg_var)
        self.orphan_level += (orphaned - self.orphan_level) * self.spawn_rate
        if (
            self.orphan_level > self.spawn_threshold
            and len(self.columns) < self.max_columns
        ):
            parent = self.columns[int(affinities.argmax())]
            self._spawn(code, parent)
            affinities = np.append(affinities, [1.0])
            thresholds = np.append(thresholds, [0.0])

        order = np.argsort(affinities)[::-1][: self.top_k]
        chosen = [i for i in order if affinities[i] >= thresholds[i]]
        learns = bool(chosen)
        chosen = chosen or [int(order[0])]
        weights = np.array([max(affinities[i], 1e-6) ** 2 for i in chosen])
        weights /= weights.sum()
        return [
            (self.columns[i], float(w), learns) for i, w in zip(chosen, weights)
        ]

    def _spawn(self, code: np.ndarray, parent: LiquidColumn | None) -> None:
        column = self.factory(len(self.columns))
        column.centroid = code.copy()
        if parent is not None:
            column.warm_start_from(parent)
        self.columns.append(column)
        self.spawn_count += 1
        self.orphan_level = 0.0
