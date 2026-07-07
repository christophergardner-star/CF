"""Linear centered CKA, exactly as registered in Amendment 1.

No RBF kernel, no whitening, no substitutes.  Given feature matrices
X (n x p) and Y (n x q) --- here, one task's held-out samples represented
at one layer versus the other task's --- linear CKA is

    CKA(X, Y) = ||Yc' Xc||_F^2 / (||Xc' Xc||_F * ||Yc' Yc||_F)

with Xc, Yc column-centered.  Values lie in [0, 1]; 0 = no linear
representational overlap.
"""
from __future__ import annotations

import numpy as np


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Paired linear CKA: valid only when rows of x and y are
    representations of the SAME inputs.  For two different tasks' data the
    pairing is arbitrary and this quantity provably reduces to its chance
    floor regardless of true subspace alignment -- use
    :func:`covariance_alignment` there instead."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA requires the same number of samples")
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    cross = float(np.linalg.norm(yc.T @ xc) ** 2)
    denom = (float(np.linalg.norm(xc.T @ xc))
             * float(np.linalg.norm(yc.T @ yc)))
    if denom < 1e-24:
        return 0.0
    return cross / denom


def covariance_alignment(x: np.ndarray, y: np.ndarray) -> float:
    """Pairing-free representational overlap between two datasets at one
    layer: tr(Sa Sb) / (|Sa|_F |Sb|_F) over centered feature covariances.
    Equals 0 for activity confined to disjoint feature subspaces and 1 for
    identical second-order structure; invariant to sample pairing and to
    sample count mismatch."""
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    sa = xc.T @ xc / max(len(xc) - 1, 1)
    sb = yc.T @ yc / max(len(yc) - 1, 1)
    denom = float(np.linalg.norm(sa)) * float(np.linalg.norm(sb))
    if denom < 1e-24:
        return 0.0
    return float(np.trace(sa @ sb)) / denom
