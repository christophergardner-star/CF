"""Cascade consolidation: every slow weight is a chain of coupled variables.

Following Benna & Fusi, a parameter is not one number but the surface of a
diffusion chain ``u_1 .. u_m`` whose timescales grow geometrically
(``tau_i = tau_1 * g**(i-1)``).  Learning writes only to the surface ``u_1``;
repeated, consistent updates diffuse into deep, slow levels and become nearly
immovable, while one-off updates decay back out.  The chain converts the
exponential forgetting of a single-timescale weight into power-law retention,
and the deep levels act as an anchor that pulls the surface back toward old
knowledge when new pressure stops ("savings" on relearning).
"""
from __future__ import annotations

import numpy as np


class CascadeChain:
    """A tensor-shaped weight backed by ``m`` diffusion levels."""

    def __init__(
        self,
        shape: tuple[int, ...],
        levels: int = 4,
        tau1: float = 8.0,
        growth: float = 4.0,
    ) -> None:
        if levels < 1:
            raise ValueError("need at least one level")
        self.levels = np.zeros((levels,) + shape)
        self.taus = tau1 * growth ** np.arange(levels)

    @property
    def effective(self) -> np.ndarray:
        """The weight actually used by the forward pass (the surface level)."""
        return self.levels[0]

    def write(self, delta: np.ndarray) -> None:
        """Apply a learning update to the surface level only."""
        self.levels[0] += delta

    def diffuse(self, dt: float = 1.0) -> None:
        """One step of the coupled dynamics between neighbouring levels."""
        u = self.levels
        m = u.shape[0]
        if m == 1:
            return
        flow = np.zeros_like(u)
        for i in range(m - 1):
            # symmetric exchange between level i and i+1, paced by the
            # *slower* of the two so deep levels change slowly
            rate = dt / self.taus[i + 1]
            diff = (u[i] - u[i + 1]) * rate
            flow[i] -= diff
            flow[i + 1] += diff
        self.levels += flow

    def load_effective(self, weights: np.ndarray) -> None:
        """Warm-start: place inherited knowledge on the (plastic) surface only."""
        self.levels[:] = 0.0
        self.levels[0] = weights.copy()
