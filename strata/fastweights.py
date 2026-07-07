"""Fast Hebbian weights: the one-shot episodic surface layer.

A single matrix ``F`` binds sparse keys to values with a *delta rule*
(self-limiting, unlike plain Hebb): novel experience is captured instantly,
with no gradient and no replay buffer -- the memory *is* ``F``.  Writes are
gated by novelty upstream, and the store decays slowly so consolidation has a
window (not a guarantee) to distill its content into slow weights.

A small ring of recently written keys ("anchors") is kept so consolidation can
replay the store's own content later; this is a few keys, not raw data.
"""
from __future__ import annotations

import numpy as np


class FastWeightMemory:
    """Delta-rule associative memory mapping sparse keys to value vectors."""

    def __init__(self, key_dim: int, value_dim: int, max_anchors: int = 32) -> None:
        self.F = np.zeros((value_dim, key_dim))
        self.max_anchors = max_anchors
        self.anchors: list[np.ndarray] = []

    def read(self, key: np.ndarray) -> np.ndarray:
        return self.F @ key

    def confidence(self, key: np.ndarray) -> float:
        """How strongly the store responds to this key (0 = never seen)."""
        return float(np.linalg.norm(self.read(key)))

    def write(self, key: np.ndarray, value: np.ndarray, rate: float) -> float:
        """Bind ``key -> value`` by the delta rule; convergent and bounded.

        Returns the Frobenius norm of the weight change (the synaptic "work"
        done), so a host system can account for the heat of plasticity.
        """
        if rate <= 0.0:
            return 0.0
        error = value - self.F @ key
        self.F += rate * np.outer(error, key)
        self._remember_anchor(key)
        return rate * float(np.linalg.norm(error)) * float(np.linalg.norm(key))

    def decay(self, factor: float = 0.999) -> None:
        self.F *= factor

    def _remember_anchor(self, key: np.ndarray) -> None:
        for stored in self.anchors:
            if float(stored @ key) > 0.95:      # already have this pattern
                return
        self.anchors.append(key.copy())
        if len(self.anchors) > self.max_anchors:
            self.anchors.pop(0)
