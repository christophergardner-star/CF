"""Depth: a two-layer STRATA stack with event-driven ascent.

The interface law: a layer transmits upward only what it cannot predict,
pooled to its own timescale; it transmits downward only priors on inference,
never targets for learning.

* **Upward**: Layer 2 does not tick on Layer 1's clock.  When L1's CUSUM
  fires, the surprise burst (trigger -> return to calm) is pooled into one
  event ``u = surprise-weighted mean of (code ⊕ α·error)`` and L2 steps once.
  Boring ticks transmit nothing; temporal abstraction falls out.
* **Downward**: L2's prediction of the *next* event decodes (against each L1
  column's code signature) to a prior over L1's columns.  The prior biases
  L1's routing with a bounded bonus (prior-capture guard, in the router) and,
  at the moment a transition is detected, seeds L1's context toward the
  predicted column's centroid -- attacking the measured re-orientation
  transient directly.  Priors never touch plastic weights: a hallucinated
  expectation cannot consolidate itself.

Maturity synchronisation: events are emitted *after* L1's learn+re-encode
step (keys are born settled), and L2's slow learning is gated by the maturity
of the L1 features in the event -- a layer consolidates only at the rate of
the least mature thing it consumes.

Run the depth bench:  python -m strata.stack  [--cycles 4] [--ticks 300]
It measures the revisit transient (error in the first 40 ticks of each phase)
for a plain L1 vs. the stack, on a repeating A->B->C regime cycle.
"""
from __future__ import annotations

import argparse

import numpy as np

from strata.network import StrataConfig, StrataNetwork


class StrataStack:
    """Layer 1 on the input stream; Layer 2 on Layer 1's surprise events."""

    def __init__(
        self,
        l1_config: StrataConfig,
        error_weight: float = 0.5,
        timescale_ratio: int = 4,
    ) -> None:
        self.l1 = StrataNetwork(l1_config)
        self.error_weight = error_weight
        u_dim = l1_config.code_dim + l1_config.d_in
        self.l2 = StrataNetwork(StrataConfig(
            d_in=u_dim,
            code_dim=l1_config.code_dim,
            seed=l1_config.seed + 77,
            # Granularity/data trade-off, measured: L2 *can* differentiate
            # regime-arrival types (context_rho 0.85 + spawn_threshold 0.55
            # gives 2-3 columns tracking schedule structure), but at tens
            # of events per run, splitting experience across columns
            # starves each successor map and priors degrade (31% -> 10%
            # transient reduction).  At this event budget the best L2 is
            # the least differentiated one: a slow mixture context and a
            # single column.  Differentiation should pay only on schedules
            # long enough for each L2 column to see hundreds of events --
            # which is exactly why hierarchical timescales are geometric.
            context_rho=0.12,
            theta_route=0.7,
            theta_claim=0.9,
            transition_ticks=3,     # L2 ticks are events, not clock ticks
            cascade_tau1=l1_config.cascade_tau1 * timescale_ratio ** 2,
            max_columns=l1_config.max_columns,
        ))
        self._burst: list[tuple[float, np.ndarray]] = []
        self._in_burst = False
        self._settling = 0          # post-calm ticks still being pooled
        self.settle_window = 15
        self._prior: np.ndarray | None = None
        self._prior_column: int | None = None
        self.events = 0

    # ------------------------------------------------------------------ tick

    def step(self, x: np.ndarray, plasticity: float = 1.0) -> dict:
        prior = self._column_prior()
        report = self.l1.step(x, plasticity=plasticity, routing_prior=prior)

        transition = report["transition"]
        if transition and not self._in_burst:
            self._in_burst = True
            self._burst = []                     # a fresh boundary begins
            self._seed_context()                 # top-down jump-start
        if self._in_burst and not transition:
            # calm returned: now pool the *arriving* regime's early codes.
            # Pooling the burst itself was measured to feed L2 "what
            # confusion looks like" -- every boundary identical -- so L2
            # could never differentiate schedules.  The event must describe
            # what arrived, not what the transition felt like.
            self._in_burst = False
            self._settling = self.settle_window
        if self._settling > 0:
            self._pool(report)
            self._settling -= 1
            if self._settling == 0:              # settled: emit upward
                self._emit(plasticity)
        report["l2_columns"] = len(self.l2.router.columns)
        report["l2_events"] = self.events
        return report

    # ------------------------------------------------------------- interface

    def _pool(self, report: dict) -> None:
        """Accumulate the surprise burst (post-settlement codes only)."""
        pending = self.l1.pending
        if pending is None or report["error"] is None:
            return
        error = np.zeros(self.l1.cfg.d_in)
        if self.l1.pending is not None:
            # the residual error direction of the *previous* prediction is
            # part of what L1 could not explain
            error = pending.prediction - pending.mu
        phi = np.concatenate([pending.code, self.error_weight * error])
        norm = np.linalg.norm(phi)
        if norm > 1e-12:
            self._burst.append((max(report["novelty"], 1e-3), phi / norm))

    def _emit(self, plasticity: float) -> None:
        if not self._burst:
            return
        weights = np.array([w for w, _ in self._burst])
        vectors = np.array([v for _, v in self._burst])
        u = (weights[:, None] * vectors).sum(axis=0) / weights.sum()
        # maturity synchronisation: consolidate upward knowledge only as
        # fast as the L1 features inside the event have settled
        maturity = self.l1.encoder.support_maturity(
            np.abs(u[: self.l1.cfg.code_dim]))
        self.l2.step(u, plasticity=plasticity * maturity)
        self._burst = []
        self.events += 1
        self._decode_prior()

    def _decode_prior(self) -> None:
        """L2's predicted next event -> a prior over L1's columns."""
        self._prior = None
        self._prior_column = None
        if self.l2.pending is None:
            return
        predicted_code = self.l2.pending.prediction[: self.l1.cfg.code_dim]
        norm = np.linalg.norm(predicted_code)
        if norm < 0.1:                           # no confident expectation
            return
        prior = []
        for column in self.l1.router.columns:
            sig_norm = np.linalg.norm(column.code_signature)
            if sig_norm < 1e-9:
                prior.append(0.0)
            else:
                prior.append(max(0.0, float(
                    predicted_code @ column.code_signature) / (norm * sig_norm)))
        self._prior = np.array(prior)
        best = int(self._prior.argmax())
        if self._prior[best] > 0.6:
            self._prior_column = best

    def _column_prior(self) -> np.ndarray | None:
        if self._prior is None:
            return None
        if len(self._prior) != len(self.l1.router.columns):
            return None                          # columns spawned since
        return self._prior

    def _seed_context(self) -> None:
        """At a detected transition, jump-start L1's slow context toward the
        predicted regime's centroid (inference bias only; no weights)."""
        if self._prior_column is None:
            return
        if self._prior_column >= len(self.l1.router.columns):
            return
        centroid = self.l1.router.columns[self._prior_column].centroid
        self.l1.context = 0.5 * self.l1.context + 0.5 * centroid


# ------------------------------------------------------------------- bench

def revisit_transients(net_step, tasks, cycles: int, ticks: int) -> list[float]:
    """Mean error over the first 40 ticks of each phase, cycles >= 2."""
    transients = []
    for cycle in range(cycles):
        for task in tasks:
            window = []
            for t in range(ticks):
                report = net_step(task.sample(t))
                if report["error"] is not None and t < 40:
                    window.append(report["error"])
            if cycle >= 1:                       # skip first (learning) cycle
                transients.append(float(np.mean(window)))
    return transients


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from strata.demo import make_interference_tasks

    def tasks():
        return make_interference_tasks(
            24, n_unique=10, n_shared=3, task_seeds=[1, 2, 3],
            shared_seed=args.seed + 99)

    config = StrataConfig(d_in=24, code_dim=128, seed=args.seed)

    plain = StrataNetwork(config)
    plain_t = revisit_transients(plain.step, tasks(), args.cycles, args.ticks)

    stack = StrataStack(StrataConfig(d_in=24, code_dim=128, seed=args.seed))
    stack_t = revisit_transients(stack.step, tasks(), args.cycles, args.ticks)

    print(f"\nregime cycle A->B->C x{args.cycles}, {args.ticks} ticks/phase "
          f"(revisit transient = mean error, first 40 ticks, cycles >= 2)\n")
    print(f"plain L1 : transient {np.mean(plain_t):.3f}  "
          f"(n={len(plain_t)} phases)")
    print(f"stack    : transient {np.mean(stack_t):.3f}  "
          f"(L2 events={stack.events}, L2 columns="
          f"{len(stack.l2.router.columns)})")
    print(f"\nreduction: {100 * (1 - np.mean(stack_t) / np.mean(plain_t)):.0f}%"
          f"  (positive = top-down priors are earning their complexity)")


if __name__ == "__main__":
    main()
