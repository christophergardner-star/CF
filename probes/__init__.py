"""Interventional probes for adjudicating the STRATA failure taxonomy.

This package implements the registered criteria of
``paper/preregistration.md`` (including Amendment 1): the three exclusion
tests that decide whether a forgetting event lies outside the trichotomy of
causes (substrate overlap, timescale collision, basis correlation), and the
orchestrator that runs all three on one event.

The probes are architecture-agnostic: anything implementing the
:class:`~probes.target.ProbeTarget` protocol can be diagnosed --- the point
being that the *instrument*, not any particular system, is the transferable
artifact under test.
"""
from probes.cka import linear_cka
from probes.exclusions import (
    basis_exclusion,
    diagnose,
    substrate_exclusion,
    timescale_exclusion,
)
from probes.target import ProbeTarget

__all__ = [
    "ProbeTarget",
    "linear_cka",
    "substrate_exclusion",
    "timescale_exclusion",
    "basis_exclusion",
    "diagnose",
]
