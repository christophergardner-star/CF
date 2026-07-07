"""The protocol a system must implement to be diagnosable.

Deliberately minimal: flat parameter access, an evaluation, a training
phase whose update magnitude can be scaled, per-layer activations
enumerated from the architecture alone (Amendment 1: the layer set must
not depend on any probe's output), and cloning so probes cannot
contaminate one another.
"""
from __future__ import annotations

import copy
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ProbeTarget(Protocol):
    def get_params(self) -> np.ndarray:
        """A flat copy of every learnable parameter."""
        ...

    def set_params(self, flat: np.ndarray) -> None:
        """Restore parameters from a flat vector (inverse of get_params)."""
        ...

    def evaluate(self, task: Any) -> float:
        """Performance on ``task`` in [0, 1]; higher is better."""
        ...

    def train(self, task: Any, update_scale: float = 1.0) -> None:
        """One training phase on ``task``; ``update_scale`` multiplies the
        magnitude of every parameter update (the timescale intervention)."""
        ...

    def activations(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        """Per-layer activations for ``inputs`` (rows = samples), keyed by
        layer name.  The layer set is enumerated from the architecture:
        every module with learnable parameters plus the penultimate
        representation, per Amendment 1."""
        ...


def clone(target: ProbeTarget) -> ProbeTarget:
    """Probes run on copies; the diagnosed system is never mutated."""
    return copy.deepcopy(target)
