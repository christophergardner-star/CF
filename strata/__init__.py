"""STRATA: a Stratified-Plasticity State-Space Network prototype.

A minimal vertical slice of the STRATA continual-learning architecture:

    sparse kWTA encoder -> centroid routing -> liquid SSM columns
        -> fast Hebbian weights -> cascade (Benna-Fusi) consolidation

Everything learns online with *local* rules (delta rule + eligibility of a
diagonal recurrence); there is no backprop-through-time and no replay buffer.
This package is deliberately self-contained: it depends on numpy but touches
nothing in the organism kernel.
"""
from strata.cascade import CascadeChain
from strata.column import LiquidColumn
from strata.encoder import SparseEncoder
from strata.fastweights import FastWeightMemory
from strata.network import StrataConfig, StrataNetwork
from strata.routing import ColumnRouter

__all__ = [
    "CascadeChain",
    "ColumnRouter",
    "FastWeightMemory",
    "LiquidColumn",
    "SparseEncoder",
    "StrataConfig",
    "StrataNetwork",
]
