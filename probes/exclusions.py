"""The three interventional exclusions and the event orchestrator.

Registered logic (paper/preregistration.md): a forgetting event lies
OUTSIDE the trichotomy iff all three exclusions pass on the same event.
Attribution is interventional: each test either reverses/prevents the loss
(condition attributed) or fails to (condition excluded).  Attribution by
elimination does not exist in this code, by construction.

Registered constants, not editable after first external data:
event threshold 20% relative drop; substrate recovery < 50%; timescale
slowdown 10x with forgetting reduction < 50%; basis CKA < 0.1 at every
architecturally-enumerated layer.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from probes.cka import covariance_alignment
from probes.target import ProbeTarget, clone

EVENT_DROP = 0.20
RECOVERY_BAR = 0.50
SLOWDOWN = 0.10
REDUCTION_BAR = 0.50
CKA_BAR = 0.10


def substrate_exclusion(
    target_after_a: ProbeTarget, task_a: Any, task_b: Any
) -> dict:
    """Train on B, then restore every parameter to its pre-B value.
    If that recovers < 50% of the lost performance, the loss is not
    carried by overwritten substrate (exclusion passes)."""
    probe = clone(target_after_a)
    params_before = probe.get_params()
    acc_before = probe.evaluate(task_a)
    probe.train(task_b)
    acc_after = probe.evaluate(task_a)
    lost = acc_before - acc_after
    if lost <= 1e-12:
        return {"event": False, "passes": False,
                "acc_before": acc_before, "acc_after": acc_after}
    probe.set_params(params_before)
    acc_restored = probe.evaluate(task_a)
    recovery = (acc_restored - acc_after) / lost
    return {
        "event": lost / max(acc_before, 1e-12) > EVENT_DROP,
        "acc_before": acc_before,
        "acc_after": acc_after,
        "recovery_fraction": float(recovery),
        "passes": bool(recovery < RECOVERY_BAR),
    }


def timescale_exclusion(
    target_after_a: ProbeTarget, task_a: Any, task_b: Any
) -> dict:
    """Rerun the interfering phase with updates scaled down 10x.  If
    forgetting shrinks by < 50%, the loss is not a function of unprotected
    update speed (exclusion passes)."""
    fast = clone(target_after_a)
    acc_before = fast.evaluate(task_a)
    fast.train(task_b)
    forgetting_full = acc_before - fast.evaluate(task_a)

    slow = clone(target_after_a)
    slow.train(task_b, update_scale=SLOWDOWN)
    forgetting_slow = acc_before - slow.evaluate(task_a)

    if forgetting_full <= 1e-12:
        return {"event": False, "passes": False,
                "forgetting_full": forgetting_full}
    reduction = 1.0 - forgetting_slow / forgetting_full
    return {
        "event": forgetting_full / max(acc_before, 1e-12) > EVENT_DROP,
        "forgetting_full": float(forgetting_full),
        "forgetting_slow": float(forgetting_slow),
        "reduction_fraction": float(reduction),
        "passes": bool(reduction < REDUCTION_BAR),
    }


def basis_exclusion(
    target_after_a: ProbeTarget,
    heldout_a: np.ndarray,
    heldout_b: np.ndarray,
) -> dict:
    """Per-layer linear CKA between the tasks' representations, measured on
    the pre-interference model (Amendment 1).  If every layer in the
    architecturally-enumerated set is below 0.1, the loss cannot be carried
    by correlated bases (exclusion passes, given a forgetting event)."""
    acts_a = target_after_a.activations(heldout_a)
    acts_b = target_after_a.activations(heldout_b)
    # pairing-free overlap: paired CKA on unrelated samples provably
    # reduces to its chance floor regardless of true subspace alignment
    # (measured by this instrument's own tests before any external run)
    per_layer = {name: covariance_alignment(acts_a[name], acts_b[name])
                 for name in acts_a}
    # transparency, not logic: an empirical null per layer -- alignment
    # after destroying feature correspondence by column permutation.
    # Layers whose null reaches the registered bar are flagged vacuous:
    # there the exclusion cannot pass and must not be read as evidence.
    rng = np.random.default_rng(0)
    null_floor = {}
    for name in acts_a:
        nulls = []
        for _ in range(3):
            perm = rng.permutation(acts_b[name].shape[1])
            nulls.append(covariance_alignment(
                acts_a[name], acts_b[name][:, perm]))
        null_floor[name] = float(np.mean(nulls))
    max_layer = max(per_layer, key=per_layer.get)
    return {
        "per_layer": per_layer,
        "null_floor": null_floor,
        "vacuous_layers": [name for name, floor in null_floor.items()
                           if floor >= CKA_BAR],
        "max_cka": float(per_layer[max_layer]),
        "max_layer": max_layer,          # must be reported either way
        "passes": bool(all(v < CKA_BAR for v in per_layer.values())),
    }


def diagnose(
    target_after_a: ProbeTarget,
    task_a: Any,
    task_b: Any,
    heldout_a: np.ndarray,
    heldout_b: np.ndarray,
) -> dict:
    """Run all three exclusions on one candidate forgetting event.

    ``outside_trichotomy`` is True only when a genuine event (>20%
    relative drop) passes all three exclusions --- the registered (c)
    criterion.  Everything else is attribution: whichever exclusions
    *fail* name the conditions carrying the loss."""
    substrate = substrate_exclusion(target_after_a, task_a, task_b)
    timescale = timescale_exclusion(target_after_a, task_a, task_b)
    basis = basis_exclusion(target_after_a, heldout_a, heldout_b)
    event = bool(substrate["event"])
    attributed = []
    if event:
        if not substrate["passes"]:
            attributed.append("substrate")
        if not timescale["passes"]:
            attributed.append("timescale")
        if not basis["passes"]:
            attributed.append("basis")
    return {
        "event": event,
        "substrate": substrate,
        "timescale": timescale,
        "basis": basis,
        "attributed_to": attributed,
        "outside_trichotomy": bool(
            event and substrate["passes"] and timescale["passes"]
            and basis["passes"]
        ),
    }
