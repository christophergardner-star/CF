"""A column: a liquid state-space reservoir with cascade-protected readouts.

The recurrence is diagonal and leaky, with *liquid* time constants -- the leak
of every channel depends on the current input, and novelty stretches the
integration window ("dwell on what you don't understand").  Because the decay
factor always lies in (0, 1), the state is bounded for all time: explosion is
impossible by construction.

The reservoir itself is fixed (liquid-state-machine style); everything the
column *learns* lives in two readouts, each backed by a CascadeChain:

* ``C`` reads the recurrent state (temporal context),
* ``D`` reads the sparse code directly (static associations -- this is where
  fast-weight content gets distilled during consolidation).

Both learn with a local delta rule; no backprop-through-time anywhere.
"""
from __future__ import annotations

import numpy as np

from strata.cascade import CascadeChain


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


class LiquidColumn:
    """One routable module: fixed liquid reservoir + plastic cascaded readouts."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        key_dim: int,
        n_state: int = 32,
        tau_min: float = 1.5,
        novelty_dwell: float = 2.0,
        cascade_levels: int = 4,
        cascade_tau1: float = 8.0,
        conjunctive_dim: int = 0,
        conjunctive_s: int = 12,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.tau_min = tau_min
        self.novelty_dwell = novelty_dwell
        self.W_B = rng.standard_normal((n_state, d_in)) / np.sqrt(d_in)
        self.w_tau = rng.standard_normal((n_state, d_in)) * 0.5
        self.tau_bias = rng.uniform(-1.0, 2.0, n_state)
        self.h = np.zeros(n_state)
        self.C = CascadeChain((d_out, n_state), cascade_levels, cascade_tau1)
        self.D = CascadeChain((d_out, key_dim), cascade_levels, cascade_tau1)
        # the column's slow expectation of its inputs (input space).  It is
        # both the routing identity (affinity = cosine to it) and the slow
        # pathway of the two-pathway code: downstream, inputs are encoded as
        # *deviations* from the claiming column's expectation, so the shared
        # component of a regime never reaches the sparse dictionary at all
        self.centroid = np.zeros(d_in)
        # a fixed random sparse *conjunctive* expansion of code x state,
        # read by its own cascade: kWTA over random mixtures manufactures
        # symbol-in-temporal-context features, lifting effective key rank
        # beyond a low-rank input manifold.  Fixed = nothing rotates, so
        # readout churn is impossible and maturity plumbing is m(q) = m(z)
        self.conjunctive_s = conjunctive_s
        self.R = None
        self.Q = None
        if conjunctive_dim > 0:
            self.R = rng.standard_normal(
                (conjunctive_dim, key_dim + n_state)) / np.sqrt(key_dim + n_state)
            self.Q = CascadeChain(
                (d_out, conjunctive_dim), cascade_levels, cascade_tau1)
        # code-space signature (EMA of claimed codes): the upward identity a
        # higher layer decodes its priors against
        self.code_signature = np.zeros(key_dim)
        # per-column claim statistics for adaptive routing bands
        self.claim_mean = 0.8
        self.claim_var = 0.01
        self.usage = 0

    def step(self, x: np.ndarray, novelty: float) -> np.ndarray:
        """Advance the liquid state one tick; returns the new state."""
        tau = self.tau_min + _softplus(self.w_tau @ x + self.tau_bias) * (
            1.0 + self.novelty_dwell * novelty
        )
        decay = np.exp(-1.0 / tau)
        self.h = decay * self.h + (1.0 - decay) * (self.W_B @ x)
        return self.h

    def _conjunct(self, code: np.ndarray, state: np.ndarray) -> np.ndarray:
        pre = self.R @ np.concatenate([code, state])
        q = np.zeros(len(pre))
        winners = np.argpartition(pre, -self.conjunctive_s)[-self.conjunctive_s:]
        q[winners] = np.maximum(pre[winners], 0.0)
        norm = np.linalg.norm(q)
        return q / norm if norm > 1e-12 else q

    def predict(self, code: np.ndarray) -> np.ndarray:
        prediction = self.C.effective @ self.h + self.D.effective @ code
        if self.Q is not None:
            prediction = prediction + self.Q.effective @ self._conjunct(
                code, self.h)
        return prediction

    def learn(
        self,
        state: np.ndarray,
        code: np.ndarray,
        error: np.ndarray,
        lr_state: float,
        lr_code: float,
    ) -> float:
        """Delta-rule update of both readouts (surface cascade level only).

        Returns the total Frobenius norm of the weight change -- an update is
        an outer product, so ``|outer(e, s)|_F = |e||s|`` comes for free --
        letting a host account for the metabolic heat of plasticity.
        """
        self.C.write(lr_state * np.outer(error, state))
        self.D.write(lr_code * np.outer(error, code))
        err_norm = float(np.linalg.norm(error))
        moved = err_norm * (
            lr_state * float(np.linalg.norm(state))
            + lr_code * float(np.linalg.norm(code))
        )
        if self.Q is not None:
            q = self._conjunct(code, state)
            self.Q.write(lr_code * np.outer(error, q))
            moved += lr_code * err_norm * float(np.linalg.norm(q))
        return moved

    def distill(self, key: np.ndarray, value: np.ndarray, lr: float) -> float:
        """Consolidate a fast-weight association into the static readout."""
        error = value - self.D.effective @ key
        self.D.write(lr * np.outer(error, key))
        return lr * float(np.linalg.norm(error)) * float(np.linalg.norm(key))

    def consolidate(self, dt: float = 1.0) -> None:
        self.C.diffuse(dt)
        self.D.diffuse(dt)
        if self.Q is not None:
            self.Q.diffuse(dt)

    def affinity(self, key: np.ndarray) -> float:
        """True cosine to the centroid (scale-blind on both sides -- the
        routing context and the centroid live at physical, unnormalised
        scale because the centroid doubles as the residual reference)."""
        denom = float(np.linalg.norm(key)) * float(np.linalg.norm(self.centroid))
        if denom < 1e-12:
            return 0.0
        return float(key @ self.centroid) / denom

    def update_centroid(self, code: np.ndarray, rho: float = 0.05) -> None:
        self.centroid = (1.0 - rho) * self.centroid + rho * code
        self.usage += 1

    def update_signature(self, code: np.ndarray, rho: float = 0.05) -> None:
        self.code_signature = (1.0 - rho) * self.code_signature + rho * code

    def update_claim_stats(self, affinity: float, rho: float = 0.02) -> None:
        delta = affinity - self.claim_mean
        self.claim_mean += rho * delta
        self.claim_var += rho * (delta * delta - self.claim_var)

    def warm_start_from(self, other: "LiquidColumn") -> None:
        """Inherit knowledge on a fresh, fully plastic cascade (forward transfer)."""
        self.C.load_effective(other.C.effective)
        self.D.load_effective(other.D.effective)
        if self.Q is not None and other.Q is not None:
            self.R = other.R          # same conjunctive basis: features align
            self.Q.load_effective(other.Q.effective)
        self.claim_mean = other.claim_mean
        self.claim_var = other.claim_var
